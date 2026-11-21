import {
  CloudSpeechApiKey,
  type CloudSpeechApiKeyProvider,
  CloudSpeechError,
  CloudSpeechModel,
  CloudSpeechModelId,
  SettingsSecretValue,
  isSecretReference,
  type SecretReference,
  type ServerSettingsError,
  type CloudSpeechKeys,
  type CloudSpeechProvider,
  EnvironmentSpeechTranscriptionResult,
  type SpeechTranscriptionOptions,
} from "@t3tools/contracts";
import * as SecretResolver from "@t3tools/secret-provider-core/server/SecretResolver";
import * as HostProcess from "@t3tools/shared/HostProcess";
import { readOpenCodeGoApiKeys } from "@t3tools/provider-opencode/server/openCodeGoCredentials";
import { cloudSpeechLanguages, effectiveSpeechLanguage } from "@t3tools/shared/speech";
import * as Context from "effect/Context";
import * as SecureRandom from "effect/crypto/SecureRandom";
import * as Effect from "effect/Effect";
import * as Exit from "effect/Exit";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Redacted from "effect/Redacted";
import * as Schema from "effect/Schema";
import { FetchHttpClient, HttpClient, HttpClientRequest, HttpClientResponse } from "effect/http";
import { expandHomePath } from "../os-jank.ts";
import * as ServerSettings from "../serverSettings.ts";
import { requireSettingsSecret } from "../secrets/settingsSecretValue.ts";
import { applySpeechAliases } from "./customWords.ts";
import { removeSpeechFillerWords } from "./fillerWords.ts";
import { decodeSpeechPcm, SpeechInvalidAudioError } from "./SpeechService.ts";

const endpoints = {
  openai: { url: "https://api.openai.com/v1/audio/transcriptions", model: "gpt-transcribe" },
  xai: { url: "https://api.x.ai/v1/stt", model: "grok-voice-transcribe-2.0" },
  openrouter: { url: "https://openrouter.ai/api/v1/audio/transcriptions" },
};
const OPENROUTER_CATALOG_URL =
  "https://openrouter.ai/api/v1/models?output_modalities=transcription&zdr=true";
const CODEX_TRANSCRIPTION_URL = "https://chatgpt.com/backend-api/transcribe";
// OpenCode Go has no transcription endpoint; an audio-capable Gemini model is
// the only speech-to-text path on the subscription.
const OPENCODE_GO_TRANSCRIPTION_URL =
  "https://opencode.ai/inference/go/google/v1beta/models/gemini-4-argon:generateContent";
const isApiKey = Schema.is(CloudSpeechApiKey);
const isModelId = Schema.is(CloudSpeechModelId);
const sameCredential = Schema.toEquivalence(SettingsSecretValue);
const OpenRouterCatalog = Schema.Struct({
  data: Schema.Array(Schema.Struct({ id: Schema.String, name: Schema.optionalKey(Schema.String) })),
});
const OpenCodeGoTranscription = Schema.Struct({
  candidates: Schema.optionalKey(
    Schema.Array(
      Schema.Struct({
        content: Schema.optionalKey(
          Schema.Struct({
            parts: Schema.optionalKey(
              Schema.Array(
                Schema.Struct({
                  text: Schema.optionalKey(Schema.String),
                  thought: Schema.optionalKey(Schema.Boolean),
                }),
              ),
            ),
          }),
        ),
      }),
    ),
  ),
});
const decodeCodexTokens = Schema.decodeUnknownEffect(
  Schema.fromJsonString(
    Schema.Struct({
      auth_mode: Schema.Literal("chatgpt"),
      tokens: Schema.Struct({
        access_token: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9._~+/-]+=*$/)),
        account_id: Schema.String.check(Schema.isPattern(/^[A-Za-z0-9_-]{1,256}$/)),
      }),
    }),
  ),
);

function wave(samples: Float32Array) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const tag = (at: number, value: string) =>
    new Uint8Array(buffer, at, 4).set(new TextEncoder().encode(value));
  tag(0, "RIFF");
  view.setUint32(4, buffer.byteLength - 8, true);
  tag(8, "WAVE");
  tag(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16_000, true);
  view.setUint32(28, 32_000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  tag(36, "data");
  view.setUint32(40, buffer.byteLength - 44, true);
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index]!;
    view.setInt16(44 + index * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return buffer;
}

function audioBody(
  provider: "openai" | "xai" | "codex",
  audio: ArrayBuffer,
  language: string,
  options: SpeechTranscriptionOptions,
) {
  const form = new FormData();
  if (provider !== "codex") form.set("model", endpoints[provider].model);
  if (language !== "auto") form.set(provider === "xai" ? "language" : "languages[]", language);
  if (provider !== "codex") {
    for (const { term } of options.speechCustomWords) {
      if (provider === "openai" && /[<>\r\n]/.test(term)) continue;
      form.append(provider === "openai" ? "keywords[]" : "keyterm", term);
    }
  }
  if (provider === "xai") {
    if (language !== "auto") form.set("format", "true");
    form.set(
      "filler_words",
      String(!options.speechRemoveFillerWords || Boolean(options.speechCorrectionWord)),
    );
    if (!options.speechSilenceFiltering) form.set("vad_threshold", "0");
  }
  // xAI ignores options that follow the file part.
  form.set("file", new Blob([audio], { type: "audio/wav" }), "dictation.wav");
  return HttpClientRequest.bodyFormData(form);
}

function openRouterBody(model: string, audio: ArrayBuffer, language: string) {
  return HttpClientRequest.bodyJsonUnsafe({
    model,
    input_audio: { data: Buffer.from(audio).toString("base64"), format: "wav" },
    ...(language !== "auto" ? { language } : {}),
    provider: { zdr: true },
  });
}

function openCodeGoBody(audio: ArrayBuffer, language: string, options: SpeechTranscriptionOptions) {
  const hints = options.speechCustomWords
    .map(({ term }) => term.replace(/[<>\r\n]/g, " "))
    .join(", ");
  const instructions = [
    "Transcribe the speech in this audio verbatim. Return only the transcript, without commentary or formatting.",
    language === "auto" ? "Detect the spoken language." : `The speech is in ${language}.`,
    hints ? `Spell these terms exactly when spoken: ${hints}.` : "",
  ]
    .filter(Boolean)
    .join(" ");
  return HttpClientRequest.bodyJsonUnsafe({
    contents: [
      {
        role: "user",
        parts: [
          { text: instructions },
          {
            inline_data: {
              mime_type: "audio/wav",
              data: Buffer.from(audio).toString("base64"),
            },
          },
        ],
      },
    ],
    generationConfig: { maxOutputTokens: 8192 },
  });
}

export class CloudSpeech extends Context.Service<
  CloudSpeech,
  {
    readonly configured: (
      includeCodex?: boolean,
    ) => Effect.Effect<CloudSpeechKeys, ServerSettingsError>;
    /** OpenRouter's public transcription catalog, cached briefly on this server. */
    readonly models: Effect.Effect<ReadonlyArray<CloudSpeechModel>, CloudSpeechError>;
    readonly setKey: (
      provider: CloudSpeechApiKeyProvider,
      key: Redacted.Redacted<string> | SecretReference | null,
    ) => Effect.Effect<void, ServerSettingsError>;
    readonly transcribe: (
      provider: CloudSpeechProvider,
      pcm: Uint8Array,
      options: SpeechTranscriptionOptions,
      model?: string | undefined,
    ) => Effect.Effect<string, CloudSpeechError | SpeechInvalidAudioError | ServerSettingsError>;
  }
>()("t3/speech/CloudSpeech") {}

export const layer = Layer.effect(
  CloudSpeech,
  Effect.gen(function* () {
    const settings = yield* ServerSettings.ServerSettingsService;
    const resolver = yield* SecretResolver.SecretResolver;
    const http = yield* HttpClient.HttpClient;
    const subscriptionHttp = HttpClient.withScope(http);
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const env = yield* HostProcess.Environment;
    const hostHome = yield* HostProcess.HomeDirectory;
    const codexHome = env.T3CODE_VOICE_CODEX_HOME?.trim() || env.CODEX_HOME?.trim();
    const codexAuthPath = path.join(
      codexHome ? yield* expandHomePath(codexHome) : path.join(hostHome, ".codex"),
      "auth.json",
    );
    const readCodexTokens = fs.readFileString(codexAuthPath).pipe(
      Effect.flatMap(decodeCodexTokens),
      Effect.map(({ tokens }) => tokens),
      Effect.option,
    );
    const credentials = settings.getSettings.pipe(
      Effect.map((value) => value.cloudSpeechCredentials),
    );
    const openCodeGoSession = `ses_${SecureRandom.uuidV4Unsafe().replaceAll("-", "")}`;
    const readOpenCodeGoKeys = readOpenCodeGoApiKeys({
      environment: env,
      homeDirectory: hostHome,
    }).pipe(
      Effect.orElseSucceed(() => []),
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(Path.Path, path),
    );
    const configured = Effect.fn("CloudSpeech.configured")(function* (
      includeHostCredentials = false,
    ) {
      const { openai, xai, openrouter } = yield* credentials;
      return {
        openai: Boolean(openai),
        xai: Boolean(xai),
        openrouter: Boolean(openrouter),
        ...(includeHostCredentials
          ? {
              codex: Option.isSome(yield* readCodexTokens),
              opencodeGo: (yield* readOpenCodeGoKeys).length > 0,
            }
          : {}),
      };
    });
    const execute = (request: HttpClientRequest.HttpClientRequest) =>
      http
        .execute(request)
        .pipe(
          Effect.provideService(FetchHttpClient.RequestInit, { redirect: "error" }),
          Effect.provideService(HttpClient.TracerPropagationEnabled, false),
        );
    const fetchCatalog = Effect.fn("CloudSpeech.catalog")(function* () {
      const unavailable = () =>
        new CloudSpeechError({ provider: "openrouter", reason: "unavailable" });
      const response = yield* execute(HttpClientRequest.get(OPENROUTER_CATALOG_URL)).pipe(
        Effect.mapError(unavailable),
      );
      if (response.status < 200 || response.status >= 300) return yield* unavailable();
      const { data } = yield* HttpClientResponse.schemaBodyJson(OpenRouterCatalog)(response).pipe(
        Effect.mapError(unavailable),
      );
      return data.flatMap(({ id, name }) =>
        isModelId(id) ? [{ id, name: name?.trim() || id }] : [],
      );
    });
    // The catalog is public, so the cache holds no credential or consent state.
    const models = yield* Effect.cachedWithTTL(
      fetchCatalog().pipe(
        Effect.timeoutOrElse({
          duration: "10 seconds",
          orElse: () =>
            Effect.fail(new CloudSpeechError({ provider: "openrouter", reason: "unavailable" })),
        }),
      ),
      (exit) => (Exit.isSuccess(exit) ? "10 minutes" : 0),
    );
    const setKey = Effect.fn("CloudSpeech.setKey")(function* (
      provider: CloudSpeechApiKeyProvider,
      key: Redacted.Redacted<string> | SecretReference | null,
    ) {
      yield* settings.updateSettings({
        cloudSpeechCredentials: {
          [provider]: Redacted.isRedacted(key) ? Redacted.value(key) : (key ?? ""),
        },
      });
    });
    const transcribe = Effect.fn("CloudSpeech.transcribe")(
      function* (
        provider: CloudSpeechProvider,
        pcm: Uint8Array,
        options: SpeechTranscriptionOptions,
        model?: string,
      ) {
        const api =
          provider === "codex" || provider === "opencode-go"
            ? undefined
            : { provider, source: (yield* credentials)[provider] };
        if (api && !api.source)
          return yield* new CloudSpeechError({ provider, reason: "missing-key" });
        if (provider === "openrouter" && (!model || !isModelId(model)))
          return yield* new CloudSpeechError({ provider, reason: "model-unavailable" });
        const samples = yield* Effect.try({
          try: () => decodeSpeechPcm(pcm),
          catch: () =>
            new SpeechInvalidAudioError({
              byteLength: pcm.byteLength,
              message: "Invalid PCM audio.",
            }),
        });
        const language = effectiveSpeechLanguage(cloudSpeechLanguages, options.speechLanguage);
        let request: HttpClientRequest.HttpClientRequest;
        if (provider === "codex") {
          const tokens = yield* readCodexTokens;
          if (Option.isNone(tokens))
            return yield* new CloudSpeechError({ provider, reason: "missing-key" });
          request = HttpClientRequest.post(CODEX_TRANSCRIPTION_URL).pipe(
            HttpClientRequest.bearerToken(Redacted.make(tokens.value.access_token)),
            HttpClientRequest.setHeaders({
              "chatgpt-account-id": tokens.value.account_id,
              originator: "Codex Desktop",
              "user-agent": "T3Code",
            }),
            audioBody("codex", wave(samples), language, options),
          );
        } else if (provider === "opencode-go") {
          const [key] = yield* readOpenCodeGoKeys;
          if (!key) return yield* new CloudSpeechError({ provider, reason: "missing-key" });
          request = HttpClientRequest.post(OPENCODE_GO_TRANSCRIPTION_URL).pipe(
            HttpClientRequest.bearerToken(Redacted.make(key)),
            HttpClientRequest.setHeaders({
              "x-opencode-session": openCodeGoSession,
              "user-agent": "T3Code",
            }),
            openCodeGoBody(wave(samples), language, options),
          );
        } else {
          const { provider, source } = api!;
          if (provider === "openrouter" && !(yield* models).some((entry) => entry.id === model))
            return yield* new CloudSpeechError({ provider, reason: "model-unavailable" });
          // A vault that locked since the last recording must not authorize a new upload.
          if (isSecretReference(source)) yield* resolver.invalidate([source]);
          const unreadable = () => new CloudSpeechError({ provider, reason: "unreadable-key" });
          const key = yield* requireSettingsSecret(resolver, source).pipe(
            Effect.mapError(unreadable),
          );
          if (!isApiKey(key)) return yield* unreadable();
          const audio = wave(samples);
          request = HttpClientRequest.post(endpoints[provider].url).pipe(
            HttpClientRequest.bearerToken(Redacted.make(key)),
            provider === "openrouter"
              ? openRouterBody(model!, audio, language)
              : audioBody(provider, audio, language, options),
          );
          // A catalog fetch or vault lookup may have outlived the credential it started with.
          const current = (yield* credentials)[provider];
          if (!current || !sameCredential(source, current))
            return yield* new CloudSpeechError({
              provider,
              reason: current ? "key-changed" : "missing-key",
            });
        }
        const unavailable = () => new CloudSpeechError({ provider, reason: "unavailable" });
        const response = yield* (
          provider === "codex" || provider === "opencode-go"
            ? subscriptionHttp.execute(request).pipe(
                Effect.provideService(FetchHttpClient.RequestInit, {
                  redirect: "error",
                  credentials: "omit",
                }),
                // Neither the OAuth credential nor the account ID belongs in HTTP spans.
                Effect.provideService(HttpClient.TracerDisabledWhen, () => true),
              )
            : execute(request)
        ).pipe(Effect.mapError(unavailable));
        if (response.status < 200 || response.status >= 300)
          return yield* new CloudSpeechError({
            provider,
            reason:
              response.status === 429
                ? "rate-limit"
                : [401, 402, 403].includes(response.status)
                  ? "rejected-key"
                  : response.status === 404 && provider === "openrouter"
                    ? "model-unavailable"
                    : "unavailable",
          });
        let text: string;
        if (provider === "opencode-go") {
          const { candidates } = yield* HttpClientResponse.schemaBodyJson(OpenCodeGoTranscription)(
            response,
          ).pipe(Effect.mapError(unavailable));
          text =
            candidates
              ?.flatMap(({ content }) => content?.parts ?? [])
              .filter((part) => !part.thought)
              .map((part) => part.text ?? "")
              .join("") ?? "";
        } else {
          const result = yield* HttpClientResponse.schemaBodyJson(
            EnvironmentSpeechTranscriptionResult,
          )(response).pipe(Effect.mapError(unavailable));
          text = result.text;
        }
        const corrected = applySpeechAliases(text, options.speechCustomWords);
        return (
          options.speechRemoveFillerWords
            ? removeSpeechFillerWords(corrected, language, options.speechCorrectionWord)
            : corrected
        ).trim();
      },
      (effect, provider) =>
        effect.pipe(
          Effect.scoped,
          Effect.timeoutOrElse({
            duration: "2 minutes",
            orElse: () => Effect.fail(new CloudSpeechError({ provider, reason: "unavailable" })),
          }),
        ),
    );
    return CloudSpeech.of({ configured, models, setKey, transcribe });
  }),
);
