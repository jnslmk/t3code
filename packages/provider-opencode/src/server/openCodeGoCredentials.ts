import * as NodeSqlite from "node:sqlite";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Option from "effect/Option";
import * as Path from "effect/Path";
import * as Schema from "effect/Schema";

const AuthFile = Schema.Struct({ "opencode-go": Schema.optionalKey(Schema.Unknown) });
const ApiAuth = Schema.Struct({ type: Schema.Literal("api"), key: Schema.String });
const decodeAuthFile = Schema.decodeEffect(Schema.fromJsonString(AuthFile));
const decodeApiAuth = Schema.decodeUnknownOption(ApiAuth);
// OpenCode 2's database tags a stored key "key"; auth.json tags it "api".
const StoredKey = Schema.Struct({
  type: Schema.Literal("key"),
  key: Schema.String,
  metadata: Schema.optionalKey(
    Schema.NullOr(Schema.Struct({ server: Schema.optionalKey(Schema.Unknown) })),
  ),
});
const decodeStoredKey = Schema.decodeOption(Schema.fromJsonString(StoredKey));

// Select before decoding so an OAuth or self-hosted connection cannot fall
// through to another stored account. NULL-active rows predate the selection flag.
const readStoredApiKey = (databasePath: string, integrationId: "opencode" | "opencode-go") =>
  Effect.try({
    try: () => {
      const database = new NodeSqlite.DatabaseSync(databasePath, { readOnly: true });
      try {
        database.exec("PRAGMA busy_timeout = 100");
        const row = database
          .prepare(
            `SELECT value FROM credential
             WHERE integration_id = ? AND active IS NOT 0
             ORDER BY active DESC, time_created DESC, id DESC LIMIT 1`,
          )
          .get(integrationId);
        const stored =
          typeof row?.value === "string"
            ? Option.getOrUndefined(decodeStoredKey(row.value))
            : undefined;
        const server = stored?.metadata?.server;
        if (typeof server === "string") {
          const url = new URL(server);
          if (`${url.origin}${url.pathname.replace(/\/+$/, "")}` !== "https://opencode.ai/console")
            return undefined;
        }
        return stored?.key.trim();
      } finally {
        database.close();
      }
    },
    catch: () => undefined,
  }).pipe(Effect.orElseSucceed(() => undefined));

/**
 * API keys the host's OpenCode install uses for OpenCode Go. The Console API
 * accepts any of them; a key without a Go subscription fails upstream.
 */
export const readOpenCodeGoApiKeys = Effect.fn("readOpenCodeGoApiKeys")(function* (input: {
  readonly environment: NodeJS.ProcessEnv;
  readonly homeDirectory: string;
}) {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const env = input.environment;
  const dataHome =
    env.XDG_DATA_HOME ||
    path.join(env.HOME || env.USERPROFILE || input.homeDirectory, ".local", "share");
  const authPath = path.join(dataHome, "opencode", "auth.json");
  const contents =
    env.OPENCODE_AUTH_CONTENT ||
    (yield* fs.readFileString(authPath).pipe(
      Effect.catchTags({
        PlatformError: (error) =>
          error.reason._tag === "NotFound" ? Effect.succeed("{}") : Effect.fail(error),
      }),
    ));
  const auth = yield* decodeAuthFile(contents);
  const apiAuth = decodeApiAuth(auth["opencode-go"]);
  // OpenCode overlays stored API credentials after environment credentials.
  const goKey = (Option.isSome(apiAuth) ? apiAuth.value.key : env.OPENCODE_API_KEY)?.trim();
  // Console supplies Go credentials in OpenCode 2. Inline auth explicitly
  // overrides local credentials, including the Console connection.
  const databaseFile = env.OPENCODE_DB ?? "opencode.db";
  const databasePath = path.resolve(dataHome, "opencode", databaseFile);
  const storedKeys =
    env.OPENCODE_AUTH_CONTENT || databaseFile === ":memory:"
      ? []
      : [
          yield* readStoredApiKey(databasePath, "opencode"),
          // OpenCode 2 imports auth.json once, so a replaced Go key only exists here.
          yield* readStoredApiKey(databasePath, "opencode-go"),
        ];
  return [...new Set([...storedKeys, goKey].filter((key): key is string => Boolean(key)))];
});
