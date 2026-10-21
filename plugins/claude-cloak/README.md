# Claude Cloak for Rig (opt in)

This separate plugin adds `claude-cloak/*` catalog entries and
`/claude-cloak login | logout`. It does not replace `anthropic/*`, read or
change `ANTHROPIC_API_KEY`, install global HTTP middleware, select a model, or
start login automatically. It is **not enabled by default**. This unpublished
workspace plugin supports Linux only. Its public plugin type and runtime
dependencies are Linux-gated; it provides no Windows, macOS or WASM implementation.

## Enable and use

Only if you want subscription authentication, explicitly add the plugin from
this checkout:

```sh
rig plugin add claude_cloak::ClaudeCloakPlugin --path ./plugins/claude-cloak
```

For an existing disabled entry, use
`rig plugin enable claude_cloak::ClaudeCloakPlugin` instead.

Then reload Rig yourself. Run `/claude-cloak login` to open the browser,
complete authorization, and use `/model` to select a `claude-cloak/*`
entry. The entries copy the installed Rig catalog's Anthropic models; listing
is not proof that your subscription authorizes a particular model.

If a saved session selected an `anthropic-oauth/*` entry, select the corresponding
`claude-cloak/*` entry after reloading. Existing credentials remain usable; the
credential directory is unchanged.

Login uses browser OAuth PKCE and a state-checked loopback callback, preferring
port 54545 and falling back to an OS-assigned port. It requires Linux
`xdg-open`. Browser-open and callback waits are bounded. There is no manual
code-paste or headless flow. This plugin never imports OMP's credentials.

Credentials are stored at `$RIG_HOME/anthropic-oauth/credentials.json`, or
`~/.rig/anthropic-oauth/credentials.json`, in a private directory with mode
0700 and files with mode 0600. Token writes use atomic replacement. Refresh is
serialized across processes and preserves refresh tokens omitted by the
server. Rotated tokens are saved before optional account lookup, so cancelling
that lookup cannot discard a successful refresh. The token and account identity
are read as one locked snapshot.
`/claude-cloak logout` deletes local credentials, not the server grant or
device identity. It refuses while login or refresh owns the credential lock;
retry after that operation finishes. Already-dispatched requests are not
cancelled. Authentication errors omit token response bodies and browser URLs.

## Compatibility and provenance

The reference is OMP commit
`d5cd750e41fede277f9eb933455da92a0738915e`, particularly
`packages/ai/src/providers/anthropic.ts`,
`packages/ai/src/registry/oauth/anthropic.ts`, and
`packages/ai/src/registry/oauth/anthropic-constants.ts`.
Already billing-prefixed system prompts are rejected instead of being reused,
to prevent duplicated or stale attestation blocks.
The pinned request fingerprint is Claude Code **2.1.280**, TypeScript SDK
**0.112.1**. These are compatibility constants copied from that source, not
claims about the program executing the request. `src/profile.rs` names this
single target's coupled identity, beta flags, cache default, billing entrypoint,
checksum seeds and compression threshold. There is no current-TUI profile,
main/auxiliary bookkeeping, synthetic title call, or copied tool inventory.

The sanitized Claude Code 2.1.294 / SDK 0.128.0 capture report is separate
evidence: the pinned checksum matched 48/48 OMP controls but 0/7 actual TUI
requests. Original decoded bytes are required to investigate that difference;
redacted fixtures cannot establish a replacement algorithm. HTTP 200 does not
establish current Claude Code parity. Thinking display updates, global cache
scope and current-TUI lifecycle metadata remain unsupported.

Agent requests select the first human message through Conversation authorship,
using its stored text, not raw keystrokes, before injected context is serialized.
When an authored human message exists, tool results and plugin-authored context
cannot displace it as the fingerprint source. The agent saves the derived
three-hex suffix and its source message ID, not another copy of the prompt.
Saved suffixes belong to this pinned profile; changing the profile requires a
reviewed state migration. Compaction and resume retain the suffix; a fork
inherits it when it retains the initial human message.
A fork before that message resets the fingerprint. Every fork receives a new
session UUID.
An image-only or empty first human message uses the empty-text seed. Direct
provider calls and delegated requests without an authoritative human message
retain the pinned OMP wire-text fallback rather than inventing an empty human
fingerprint.

An omitted cache preference selects the profile's long retention; explicit none,
short and long choices are preserved. Kernel-built
requests defer their eager catalog cache default in Prepare Input, before
normal request editors run. A provider-bound adapter supplies the long default
and private identity only when the final connection is Claude Cloak. Routed
and cloned requests do not carry private identity parameters to other providers.
A causally linked fallback that shares its predecessor's started request keeps
an omitted cache preference omitted. Editing that request creates a separate
request value and preserves the caller's explicit preference.

The plugin reuses Rig's generic Anthropic completion implementation for
message conversion, reasoning, streaming, tools and usage. Its own HTTP
client is used only by this separate vendor. It refuses destinations other
than HTTPS `api.anthropic.com/v1/messages` before reading credentials and
never follows redirects. The wrapper supplies OAuth bearer authentication,
OMP-compatible headers and beta selection, billing and Claude Code system
blocks, OMP cache-marker placement and retention, thinking controls, ordered
JavaScript-compatible JSON serialization, and a final-byte cch XXHash64
attestation. Requests at the OMP size threshold are gzip-compressed after
attestation. Equivalent gzip data may have different compressed bytes because
the runtimes use different DEFLATE implementations.
Custom tool names receive OMP's `_` prefix on the wire; the handler removes
one prefix from streamed and completed replies before Rig dispatches tools.

Device metadata derives from a private installation seed and account UUID.
Each Rig agent has a saved session UUID, attached only to this provider's
requests. Direct provider calls outside an agent retain a process-local
fallback UUID. Device initialization can fail closed if another process is creating its first seed
at exactly the same time; retrying then reads the completed file.

This is not the OMP agent or Claude Code. It does not replicate their full
system prompt, built-in tools, telemetry, compression-rejection recovery,
retries, request scheduling, context editing, compaction, model discovery, plan/quota
UI, or OAuth status command. Normal Rig message encoding and capabilities
still apply. No claim is made that subscriptions permit third-party clients,
that every model is entitled, or that Anthropic accepts this fingerprint.
Use only an account you are authorized to access and review provider terms.

## Verification

Offline tests cover OAuth PKCE/state, private storage and locking, refresh
merging, destination guards, ordered billing attestation, tool-name reversal,
and vendor isolation. The checked-in corpus under `tests/fixtures/omp/`
contains 48 requests generated by the pinned OMP source. A localhost regression
sends the same inputs through Rig's actual provider and compares exact decoded
JSON bytes (including the checksum), application headers and compression mode.
It does not normalize away payload differences. Another 18 source-generated
schema fixtures test strict budgets and normalization, and a two-request
streaming test covers tool-call decoding and history replay. The comparison
scope and limitations are detailed below. Run from the workspace:

```sh
cargo nextest run --locked --profile local -p claude-cloak
cargo clippy --locked -p claude-cloak --all-targets -- -D warnings
cargo fmt -p claude-cloak --check
```

Normal checks never use stored credentials or make authenticated API calls.
An ignored live acceptance test requires explicit opt-in and an existing login:

```sh
CLAUDE_CLOAK_LIVE=1 \
CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5 \
cargo test --locked -p claude-cloak --lib real_api_three_probes_without_tokio_runtime -- --ignored --nocapture
```

This sends four billable requests across three probes: unary completion,
streamed echo-tool history replay, and explicit cache-point budgeting. It uses
synthetic prompts, never reads project files, and reports only status, marker
counts and usage. It does not deliberately interrupt a real token refresh or
force malformed server streams; those cases remain offline regressions.

## Request comparison method

The reference captures execute OMP's `streamAnthropic` with an injected fetch
that records the original destination and sends to a localhost HTTP server.
Rig executes its catalog connection, completion adapter, Anthropic encoder and
OAuth transport using test-only credentials, identity and sender. The sender
redirects to localhost after the production destination guard and request
shaping. Captures pin test-only platform headers to the reference Linux/arm64
profile, independent of the test host. Production headers use the host platform.
Both servers return synthetic responses; production restrictions remain unchanged.

Inputs use identical prompts, messages, fake access token, account UUID,
session UUID and installation seed. Comparisons use original URLs, application
headers and exact decoded JSON bytes. They do not normalize beta order, tool
schemas, cache controls, metadata, checksums or JSON property order. Only local
routing/framing headers (`host`, `content-length`, `transfer-encoding`) are
excluded. Compression mode must match. Compressed wire bytes are retained,
but different DEFLATE implementations can produce different gzip bytes.

### Coverage

The 48 request fixtures cover Sonnet 4.6 and budget-thinking Sonnet 4.5:

- Plain text, absent system prompt and requests above the compression threshold.
- Custom tools, auto/required/none/named choices, underscored names, the buffered
  yield tool and tool-result history.
- Adaptive and explicit budget thinking with output-token reservation.
- Disabled, short and long cache retention with multiple conversation turns.
- Unicode, multi-block text, JSON escapes, numeric schema keys and floats.
- Schema constraint spilling and strict eligibility for the covered schemas.
- Sampling with thinking and stop-sequence truncation.
- Cache boundaries at 19/20/21 and 29/30/31 messages with one/three system blocks.
- Required, named and disabled tools with adaptive/budget thinking.
- Serialized ASCII/emoji lengths of 1023/1024/1025 UTF-16 units and UTF-8 bytes.

Compression uses UTF-8 byte length because OMP's checksum wrapper supplies
bytes. The effort beta retains original thinking intent even when forced tools
remove thinking from the final body. Internal request metadata retains caller
intent before the generic encoder resolves it and is removed before sending.
Explicit cache points reserve the four-marker budget before automatic anchors
are added. Invalid marker counts or TTL ordering fail before sending.

The 18 schema cases cover strict allowlists, opt-outs, optional-property
budgets at 23/24/25, union budgets at 15/16/17, strict-tool limits at 19/20/21,
failed-candidate budget rollback, references, combinators, enums, constraints
and legacy schemas. Tests compare normalized schema structure and serialized
schema property order to OMP output. Tool wrapper streaming/cache fields are
excluded from these helper-level comparisons, not full-request comparisons.

`src/transport/roundtrip.rs` exercises two interleaved streamed tool calls
through the catalog/handler/transport, then builds a second request from their
decoded results. It checks names, IDs, fragmented arguments, success/error
results, ordering, caller metadata, unknown native fields and absence of doubled
prefixes across saved-history serialization. This is a synthetic
end-to-end regression, not an additional OMP differential capture.

### Reproduction and limits

The normal regression needs no OMP installation, credentials or external
network. It uses localhost servers and fails on body-byte or application-header
differences. Fixture provenance is in [the request corpus README](tests/fixtures/omp/README.md)
and [the schema corpus README](tests/fixtures/omp-schema/README.md).
The ignored request-capture helper accepts an explicit `OMP_PARITY_DIR` with
caller inputs and synthetic identity files, and emits Rig captures there.
External OMP tooling and temporary research artifacts are not shipped here.

Equality is established for these profiles, not every OMP feature.
OMP-specific developer/per-call-message provenance, synthetic continuation
pads, context editing, compaction and retry state are not fully replicated.
Rig's typed messages cannot preserve every arbitrary OMP input distinction,
such as scalar text versus a one-element text array. Other models, schemas and
provider options need their own differential coverage. ArkType AST/comment
conversion and exhaustive reference/legacy-draft collision behavior are not
replicated or exhaustively tested. Internal OMP `NO_STRICT`/`disableStrictTools`
switches are not exposed. Compression-rejection retry/latching is not copied.
Schema postprocessing deliberately traverses only schema-valued positions.
Unlike the pinned OMP walkers, it preserves literal instance data in `const`,
`default`, `enum` and examples, even when that data resembles a schema.

Browser login is independently tested but is not covered by dummy inference
comparisons. Manual-code/headless login, monthly warning UI, server acceptance,
subscription entitlements, TLS fingerprint equivalence and full agent-loop
equivalence are not claimed.
