import { it } from "@effect/vitest";
import * as NodeServices from "@effect/platform-node/NodeServices";
import { DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS, SecretProviderId } from "@t3tools/contracts";
import * as SecretResolver from "@t3tools/secret-provider-core/server/SecretResolver";
import * as HostProcess from "@t3tools/shared/HostProcess";
import * as Effect from "effect/Effect";
import * as Fiber from "effect/Fiber";
import * as FileSystem from "effect/FileSystem";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import * as Redacted from "effect/Redacted";
import * as FetchHttpClient from "effect/http/FetchHttpClient";
import { expect, vi } from "vite-plus/test";
import * as ServerSettings from "../serverSettings.ts";
import * as CloudSpeech from "./CloudSpeech.ts";

it.effect.each(["openai", "xai"] as const)("sends only to %s using its own saved key", (provider) =>
  Effect.gen(function* () {
    const reference = {
      provider: SecretProviderId.make("work"),
      reference: "op://Work/Voice/key",
    };
    let resolved: string | undefined;
    let cached: string | undefined;
    const read = vi.fn(() => Effect.sync(() => [(cached ??= resolved)]));
    const upstream = vi.fn<typeof fetch>(async () => Response.json({ text: "hello" }));
    const service = yield* CloudSpeech.CloudSpeech.pipe(
      Effect.provide(
        CloudSpeech.layer.pipe(
          Layer.provide(FetchHttpClient.layer),
          Layer.provide(NodeServices.layer),
          Layer.provide(ServerSettings.ServerSettingsService.layerTest()),
          Layer.provide(
            Layer.mock(SecretResolver.SecretResolver)({
              read,
              label: () => Effect.succeed("Work"),
              invalidate: () =>
                Effect.sync(() => {
                  cached = undefined;
                }),
            }),
          ),
        ),
      ),
    );
    const transcribe = (pcm = new Uint8Array(new Float32Array([0, 0.5]).buffer)) =>
      service
        .transcribe(provider, pcm, {
          ...DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS,
          speechLanguage: "en",
          speechCustomWords: [{ term: "T3 Code", aliases: [] }],
        })
        .pipe(Effect.provideService(FetchHttpClient.Fetch, upstream));
    yield* service.setKey(provider === "openai" ? "xai" : "openai", Redacted.make("other"));
    expect((yield* Effect.flip(transcribe()))._tag).toBe("CloudSpeechError");
    yield* service.setKey(provider, Redacted.make("chosen"));
    expect(yield* service.configured()).toEqual({ openai: true, xai: true, openrouter: false });
    yield* Effect.flip(transcribe(new Uint8Array(3)));
    expect(upstream).not.toHaveBeenCalled();
    expect(yield* transcribe()).toBe("hello");
    const [url, init] = upstream.mock.calls[0]!;
    expect(String(url)).toBe(
      {
        openai: "https://api.openai.com/v1/audio/transcriptions",
        xai: "https://api.x.ai/v1/stt",
      }[provider],
    );
    expect(init?.redirect).toBe("error");
    expect(new Headers(init?.headers).get("authorization")).toBe("Bearer chosen");
    const form = init?.body as FormData;
    expect(form.get("model")).toBe(
      provider === "openai" ? "gpt-transcribe" : "grok-voice-transcribe-2.0",
    );
    expect(form.getAll(provider === "openai" ? "keywords[]" : "keyterm")).toEqual(["T3 Code"]);
    expect(form.get(provider === "openai" ? "languages[]" : "language")).toBe("en");
    expect([...form.keys()].at(-1)).toBe("file");
    const wav = new DataView(yield* Effect.promise(() => (form.get("file") as Blob).arrayBuffer()));
    expect(wav.getUint32(24, true)).toBe(16_000);
    expect(wav.getInt16(46, true)).toBe(16384);
    upstream.mockResolvedValueOnce(new Response("sensitive upstream diagnostic", { status: 401 }));
    expect(yield* Effect.flip(transcribe())).toMatchObject({ provider, reason: "rejected-key" });
    yield* service.setKey(provider, null);
    expect((yield* service.configured())[provider]).toBe(false);
    yield* Effect.flip(transcribe());
    expect(upstream).toHaveBeenCalledTimes(2);
    yield* service.setKey(provider, reference);
    expect((yield* service.configured())[provider]).toBe(true);
    expect(read).not.toHaveBeenCalled();
    resolved = "resolved-fixture";
    expect(yield* transcribe()).toBe("hello");
    expect(read).toHaveBeenCalledWith([reference]);
    expect(new Headers(upstream.mock.calls[2]?.[1]?.headers).get("authorization")).toBe(
      "Bearer resolved-fixture",
    );
    resolved = undefined;
    expect(yield* Effect.flip(transcribe())).toMatchObject({
      provider,
      reason: "unreadable-key",
    });
    expect(upstream).toHaveBeenCalledTimes(3);
  }),
);

it.effect("uses only the selected host Codex login and sends no model or vocabulary", () =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3-voice-codex-" });
    const selected = path.join(root, "voice");
    const ambient = path.join(root, "ambient");
    yield* fs.makeDirectory(selected);
    yield* fs.makeDirectory(ambient);
    const authPath = path.join(selected, "auth.json");
    const tokens = { access_token: "fixture.oauth.token", account_id: "fixture-account" };
    const login = { auth_mode: "chatgpt", tokens };
    yield* fs.writeFileString(path.join(ambient, "auth.json"), JSON.stringify(login));
    const service = yield* CloudSpeech.CloudSpeech.pipe(
      Effect.provide(
        CloudSpeech.layer.pipe(
          Layer.provide(FetchHttpClient.layer),
          Layer.provide(ServerSettings.ServerSettingsService.layerTest()),
          Layer.provide(Layer.mock(SecretResolver.SecretResolver)({})),
        ),
      ),
      Effect.provideService(HostProcess.Environment, {
        T3CODE_VOICE_CODEX_HOME: selected,
        CODEX_HOME: ambient,
      }),
    );
    yield* service.setKey("openai", Redacted.make("available-paid-key"));
    const upstream = vi.fn<typeof fetch>(async () => Response.json({ text: "alias" }));
    const transcribe = () =>
      service
        .transcribe(
          "codex",
          new Uint8Array(new Float32Array([0, 0.5]).buffer),
          {
            ...DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS,
            speechLanguage: "en",
            speechCustomWords: [{ term: "T3 Code", aliases: ["alias"] }],
          },
          "ignored-model",
        )
        .pipe(Effect.provideService(FetchHttpClient.Fetch, upstream));

    expect(yield* service.configured(true)).toMatchObject({ codex: false, openai: true });
    expect(yield* Effect.flip(transcribe())).toMatchObject({ reason: "missing-key" });
    for (const invalid of [
      { ...login, auth_mode: "apikey" },
      { ...login, tokens: { ...tokens, account_id: "bad\r\naccount" } },
      { ...login, tokens: { ...tokens, access_token: "bad\r\ntoken" } },
    ]) {
      yield* fs.writeFileString(authPath, JSON.stringify(invalid));
      expect(yield* Effect.flip(transcribe())).toMatchObject({ reason: "missing-key" });
    }
    expect(upstream).not.toHaveBeenCalled();

    yield* fs.writeFileString(authPath, JSON.stringify(login));
    expect(yield* service.configured()).not.toHaveProperty("codex");
    expect(yield* service.configured(true)).toMatchObject({ codex: true });
    expect(yield* transcribe()).toBe("T3 Code");
    const [url, init] = upstream.mock.calls[0]!;
    expect(String(url)).toBe("https://chatgpt.com/backend-api/transcribe");
    expect(init?.redirect).toBe("error");
    expect(init?.credentials).toBe("omit");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer fixture.oauth.token");
    expect(headers.get("chatgpt-account-id")).toBe("fixture-account");
    expect(headers.get("originator")).toBe("Codex Desktop");
    expect(headers.has("cookie")).toBe(false);
    const form = init?.body as FormData;
    expect([...form.keys()]).toEqual(["languages[]", "file"]);
    expect(form.get("languages[]")).toBe("en");
    const file = form.get("file") as File;
    expect(file.type).toBe("audio/wav");
    const wav = new DataView(yield* Effect.promise(() => file.arrayBuffer()));
    expect(wav.getUint32(24, true)).toBe(16_000);
    expect(wav.getInt16(46, true)).toBe(16384);

    upstream.mockResolvedValueOnce(new Response("sensitive upstream diagnostic", { status: 401 }));
    expect(yield* Effect.flip(transcribe())).toMatchObject({
      provider: "codex",
      reason: "rejected-key",
    });
    expect(upstream).toHaveBeenCalledTimes(2);
    expect(upstream.mock.calls[1]?.[1]?.signal?.aborted).toBe(true);
  }).pipe(Effect.provide(NodeServices.layer)),
);

it.effect("transcribes with the host OpenCode Go login through Gemini", () =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const path = yield* Path.Path;
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3-voice-opencode-" });
    yield* fs.makeDirectory(path.join(root, "opencode"));
    const service = yield* CloudSpeech.CloudSpeech.pipe(
      Effect.provide(
        CloudSpeech.layer.pipe(
          Layer.provide(FetchHttpClient.layer),
          Layer.provide(ServerSettings.ServerSettingsService.layerTest()),
          Layer.provide(Layer.mock(SecretResolver.SecretResolver)({})),
        ),
      ),
      Effect.provideService(HostProcess.Environment, { XDG_DATA_HOME: root }),
    );
    const upstream = vi.fn<typeof fetch>(async () =>
      Response.json({
        candidates: [
          { content: { parts: [{ text: "Hello " }, { text: "world" }] } },
          { content: { parts: [{ thought: true, text: "ignored" }] } },
        ],
      }),
    );
    const transcribe = () =>
      service
        .transcribe("opencode-go", new Uint8Array(new Float32Array([0, 0.5]).buffer), {
          ...DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS,
          speechLanguage: "en",
          speechCustomWords: [{ term: "T3 Code", aliases: [] }],
        })
        .pipe(Effect.provideService(FetchHttpClient.Fetch, upstream));

    expect(yield* service.configured(true)).toMatchObject({ opencodeGo: false });
    expect(yield* Effect.flip(transcribe())).toMatchObject({
      provider: "opencode-go",
      reason: "missing-key",
    });
    expect(upstream).not.toHaveBeenCalled();

    yield* fs.writeFileString(
      path.join(root, "opencode", "auth.json"),
      JSON.stringify({ "opencode-go": { type: "api", key: "go-key" } }),
    );
    expect(yield* service.configured()).not.toHaveProperty("opencodeGo");
    expect(yield* service.configured(true)).toMatchObject({ opencodeGo: true });
    expect(yield* transcribe()).toBe("Hello world");

    const [url, init] = upstream.mock.calls[0]!;
    expect(String(url)).toBe(
      "https://opencode.ai/inference/go/google/v1beta/models/gemini-4-argon:generateContent",
    );
    expect(init?.redirect).toBe("error");
    expect(init?.credentials).toBe("omit");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer go-key");
    expect(headers.get("x-opencode-session")).toMatch(/^ses_[a-f0-9]{32}$/);
    const body = JSON.parse(yield* Effect.promise(() => new Response(init?.body).text())) as {
      contents: Array<{
        parts: Array<{ text?: string; inline_data?: { mime_type: string; data: string } }>;
      }>;
    };
    const [textPart, audioPart] = body.contents[0]!.parts;
    expect(textPart?.text).toContain("The speech is in en.");
    expect(textPart?.text).toContain("T3 Code");
    expect(audioPart?.inline_data?.mime_type).toBe("audio/wav");
    const wav = Buffer.from(audioPart?.inline_data?.data ?? "", "base64");
    expect(wav.toString("ascii", 0, 4)).toBe("RIFF");
    expect(wav.readInt16LE(46)).toBe(16384);

    upstream.mockResolvedValueOnce(new Response("sensitive upstream diagnostic", { status: 403 }));
    expect(yield* Effect.flip(transcribe())).toMatchObject({
      provider: "opencode-go",
      reason: "rejected-key",
    });
  }).pipe(Effect.provide(NodeServices.layer)),
);

it.effect("uploads to OpenRouter only for a listed exact model, as zero-retention JSON WAV", () =>
  Effect.gen(function* () {
    const upstream = vi.fn<typeof fetch>(async (input) =>
      new URL(String(input)).pathname.endsWith("/models")
        ? Response.json({ data: [{ id: "acme/transcribe-1" }, { id: "acme/transcribe-2" }] })
        : Response.json({ text: "hello" }),
    );
    const service = yield* CloudSpeech.CloudSpeech.pipe(
      Effect.provide(
        CloudSpeech.layer.pipe(
          Layer.provide(FetchHttpClient.layer),
          Layer.provide(NodeServices.layer),
          Layer.provide(ServerSettings.ServerSettingsService.layerTest()),
          Layer.provide(Layer.mock(SecretResolver.SecretResolver)({})),
        ),
      ),
    );
    const uploads = () =>
      upstream.mock.calls.filter(([url]) => String(url).endsWith("/audio/transcriptions"));
    const transcribe = (model: string | undefined) =>
      service
        .transcribe(
          "openrouter",
          new Uint8Array(new Float32Array([0, 0.5]).buffer),
          { ...DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS, speechLanguage: "en" },
          model,
        )
        .pipe(Effect.provideService(FetchHttpClient.Fetch, upstream));
    yield* service.setKey("openrouter", Redacted.make("router-key"));
    expect(yield* Effect.flip(transcribe(undefined))).toMatchObject({ provider: "openrouter" });
    upstream.mockResolvedValueOnce(new Response(null, { status: 503 }));
    expect(yield* Effect.flip(transcribe("acme/transcribe-2"))).toMatchObject({
      reason: "unavailable",
    });
    expect(yield* Effect.flip(transcribe("acme/delisted"))).toMatchObject({
      provider: "openrouter",
    });
    expect(uploads()).toEqual([]);
    expect(yield* transcribe("acme/transcribe-2")).toBe("hello");
    const [[url, init]] = uploads() as [[URL, RequestInit]];
    expect(new URL(String(upstream.mock.calls[0]![0])).searchParams.get("zdr")).toBe("true");
    expect(String(url)).toBe("https://openrouter.ai/api/v1/audio/transcriptions");
    expect(init.redirect).toBe("error");
    const headers = new Headers(init.headers);
    expect(headers.get("authorization")).toBe("Bearer router-key");
    expect(headers.get("content-type")).toContain("application/json");
    const body = JSON.parse(yield* Effect.promise(() => new Response(init.body).text())) as {
      model: string;
      models?: unknown;
      provider: { zdr: boolean };
      input_audio: { format: string; data: string };
    };
    expect(body).toMatchObject({
      model: "acme/transcribe-2",
      provider: { zdr: true },
      input_audio: { format: "wav" },
    });
    expect(body.models).toBeUndefined();
    const wav = Buffer.from(body.input_audio.data, "base64");
    expect(wav.toString("ascii", 0, 4)).toBe("RIFF");
    expect(wav.readInt16LE(46)).toBe(16384);
  }),
);

it.effect("removing a key while a catalog read is pending prevents the upload", () =>
  Effect.gen(function* () {
    const service = yield* CloudSpeech.CloudSpeech.pipe(
      Effect.provide(
        CloudSpeech.layer.pipe(
          Layer.provide(FetchHttpClient.layer),
          Layer.provide(NodeServices.layer),
          Layer.provide(ServerSettings.ServerSettingsService.layerTest()),
          Layer.provide(Layer.mock(SecretResolver.SecretResolver)({})),
        ),
      ),
    );
    const catalogStarted = Promise.withResolvers<void>();
    const catalog = Promise.withResolvers<Response>();
    const uploads: string[] = [];
    const upstream: typeof fetch = async (input) => {
      if (new URL(String(input)).pathname.endsWith("/models")) {
        catalogStarted.resolve();
        return catalog.promise;
      }
      uploads.push(String(input));
      return Response.json({ text: "unexpected" });
    };
    yield* service.setKey("openrouter", Redacted.make("old-key"));
    const job = yield* service
      .transcribe(
        "openrouter",
        new Uint8Array(4),
        DEFAULT_SPEECH_TRANSCRIPTION_OPTIONS,
        "acme/voice",
      )
      .pipe(Effect.provideService(FetchHttpClient.Fetch, upstream), Effect.forkChild);
    yield* Effect.promise(() => catalogStarted.promise);
    yield* service.setKey("openrouter", null);
    catalog.resolve(Response.json({ data: [{ id: "acme/voice" }] }));
    const exit = yield* Fiber.await(job);
    expect(uploads).toEqual([]);
    expect(exit._tag).toBe("Failure");
  }),
);
