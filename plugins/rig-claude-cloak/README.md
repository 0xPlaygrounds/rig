# rig-claude-cloak

An optional plugin for the rig coding agent. It adds `claude-cloak/*` catalog
entries and `/claude-cloak login | logout`. It does not replace `anthropic/*`,
read or change `ANTHROPIC_API_KEY`, install global HTTP middleware, select a
model, or start login automatically. It is **not enabled by default**. It
supports Linux and macOS. Its public plugin type and runtime dependencies are
gated to those platforms. Windows and WASM are unsupported.

## Enable and use

It ships with every rig release, at the launcher's version. Only if you want
subscription authentication, add it:

```sh
rig plugin add rig_claude_cloak::ClaudeCloakPlugin
```

The entry names the crate and no source, so the agent builds it at the
launcher's version from crates.io, or from the rig checkout the agent is built
from (`RIG_SOURCE`). The default `plugins.toml` lists it disabled, and there
`rig plugin add` enables that entry, as `/plugins` and
`rig plugin enable rig_claude_cloak::ClaudeCloakPlugin` do.

Then reload Rig yourself. Run `/claude-cloak login` to open the browser,
complete authorization, and use `/model` to select a `claude-cloak/*`
entry. The entries copy the installed Rig catalog's Anthropic models; listing
is not proof that your subscription authorizes a particular model.

If a saved session selected an `anthropic-oauth/*` entry, select the corresponding
`claude-cloak/*` entry after reloading. Existing credentials remain usable; the
credential directory is unchanged.

Login uses browser OAuth PKCE and a state-checked loopback callback, preferring
port 54545 and falling back to an OS-assigned port. It uses `xdg-open` on Linux
and `/usr/bin/open` on macOS. Browser-open and callback waits are bounded.
There is no manual code-paste or headless flow. This plugin never imports
OMP's credentials.

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

The initial implementation was adapted from OMP commit
`d5cd750e41fede277f9eb933455da92a0738915e`, particularly its Anthropic provider
and OAuth modules. The retained OMP corpus is a historical control. Its
`2.1.280` version was an offline fallback, with an updater and server-required
version adoption, rather than a permanent compatibility target.

The current request profile targets observed Claude Code **2.1.307**,
TypeScript SDK **0.128.0**, and runtime header **v26.3.0**. `src/profile.rs`
contains the coupled identity, beta flags, cache default, billing entrypoint,
checksum rules and compression threshold. These describe wire compatibility.
`tests/fixtures/claude-code/profile-2.1.307.json` records the structural TUI
reference and official binary digest.

The current checksum hashes the original compact JSON with its billing
placeholder zeroed. Its byte scanner omits model string values, fallback
arrays, unsigned `max_tokens` fields and fallback-credit tokens, including
adjacent field separators. It preserves the remaining bytes and property
order. The recipe was checked independently against the native exclusion
scanner and original primary and auxiliary TUI request bytes. Redacted or
reserialized bytes cannot establish checksum correctness. Already billing-
prefixed system prompts are rejected to prevent duplicated attestation blocks.

Agent calls use the primary beta profile even with tools and thinking off.
Thinking defaults to `display: "updates"` with the corresponding beta; an
explicit display remains authoritative. Rig does not generate Claude Code's
session-title calls, copy its tool inventory, inject optional billing lifecycle
identifiers or diagnostics, or mark its system prompt for global cache scope.
Those fields depend on the application's own prompt, request and lifecycle
state. The live comparison records these differences rather than asserting
whole-request equality. HTTP 200 alone does not establish agent-loop parity.

Agent requests select the first human message through Conversation authorship,
using its stored text, not raw keystrokes, before injected context is serialized.
When an authored human message exists, tool results and plugin-authored context
cannot displace it as the fingerprint source. The agent saves the derived
three-hex suffix, its source message ID, three selected UTF-16 units and the
profile version. It does not save another copy
of the prompt. The selected units allow a version update to recompute the
suffix after the original human message has been compacted. Compaction and
resume retain the seed; a fork inherits it when it retains that human message.
A fork before the source resets both suffix and seed. Every fork receives a
new session UUID.

Earlier saves contain only a suffix and source ID. On their first request
under the new profile, the first retained authored human supplies a new seed
and source ID. If no authored human remains, migration uses the empty-text
seed and clears the unavailable source ID. This starts a new profile provenance
boundary while retaining the OAuth session UUID. Later turns keep that seed.
An image-only or empty first human also uses the empty-text seed. Direct
provider calls and delegated requests without an authoritative human retain
the conversational wire-text fallback.

An omitted cache preference selects the profile's long retention. Explicit
`none` adds no automatic profile markers, including on the identity block.
Caller-supplied block markers and raw top-level `cache_control` are explicit
overrides and remain authoritative, so `none` is not a promise to erase those
controls. A raw root control delegates its moving breakpoint to the server;
Cloak adds no automatic profile markers alongside it. Explicit block markers
remain, and a final explicit block must have the root's TTL. With no explicit
controls, `none` sends no markers. Short and long retention govern automatic
placement; the final shaped body is checked for marker budget and TTL order.
These preferences control request markers, not the provider's internal retention
or privacy policy. This is a deliberate correction
to the pinned OMP identity-marker behavior.

Kernel-built requests defer their eager catalog cache default in Prepare Input, before
normal request editors run. A provider-bound adapter supplies the long default
and private identity only when the final connection is Claude Cloak. Routed
and cloned requests do not carry private identity parameters to other providers.
A causally linked fallback that shares its predecessor's started request keeps
an omitted cache preference omitted. Editing that request creates a separate
request value and preserves the caller's explicit preference.

The plugin reuses Rig's generic Anthropic completion implementation for
message conversion, reasoning, streaming, tools and usage. Its own HTTP
client is used only by this separate vendor. OAuth and inference use the same
workspace reqwest implementation and normal TLS verification; child-scoped
`SSL_CERT_FILE` trust configuration applies to both. It refuses destinations other
than HTTPS `api.anthropic.com/v1/messages` before reading credentials and
never follows redirects. The wrapper supplies OAuth bearer authentication,
the observed profile headers and beta selection, billing and Claude Code
identity blocks, explicit cache-marker placement and retention, thinking
controls, ordered JavaScript-compatible JSON and the current cch XXHash64
attestation. Gzip starts at 4096 UTF-16 units after attestation and is disabled
with proxy or custom-CA environment configuration. Proxy captures therefore
verify uncompressed application requests, not native TLS or gzip equivalence.
Different DEFLATE implementations can produce different compressed bytes.
Custom tool names receive OMP's `_` prefix on the wire; the handler removes
one prefix from streamed and completed replies before Rig dispatches tools.

Device metadata derives from a private installation seed and account UUID.
Each Rig agent has a saved session UUID, attached only to this provider's
requests. Direct provider calls outside an agent retain a process-local
fallback UUID. Device initialization can fail closed if another process is
creating its first seed at exactly the same time; retrying then reads the
completed file.

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
contains 48 requests generated by the historical OMP source. A localhost
regression sends the same inputs through Rig's actual provider. It validates
both the historical checksum and the current checksum, checks current profile
headers and human fingerprints, then compares the remaining decoded JSON bytes
and property order. Its declared projection replaces profile billing text,
adds current thinking display defaults, and removes only the historical
`cache_none` identity marker. Original fixture files remain unchanged.
A separate structural current-TUI control freezes current headers, beta sets
and thinking settings. Another
18 source-generated schema fixtures test strict budgets and normalization, and a two-request
streaming test covers tool-call decoding and history replay. The comparison
scope and limitations are detailed below. Run from the workspace:

```sh
cargo nextest run --locked --profile local -p rig-claude-cloak
cargo clippy --locked -p rig-claude-cloak --all-targets -- -D warnings
cargo fmt -p rig-claude-cloak --check
```

Normal checks never use stored credentials or make authenticated API calls.
An ignored live acceptance test requires explicit opt-in and an existing login:

```sh
CLAUDE_CLOAK_LIVE=1 \
CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5 \
cargo test --locked -p rig-claude-cloak --lib real_api_three_probes_without_tokio_runtime -- --ignored --nocapture
```

This sends four billable requests across three probes: unary completion,
streamed echo-tool history replay, and explicit cache-point budgeting. It uses
synthetic prompts, never reads project files, and reports only status, marker
counts and usage. It does not deliberately interrupt a real token refresh or
force malformed server streams; those cases remain offline regressions.

A portable, explicitly authorized actual-agent matrix and structural logging
observer are documented in [the acceptance harness README](tests/acceptance/README.md).
They support supplied builds rather than machine-specific paths, isolate runtime
state, cap requests and duration, and classify missing evidence as unknown.
Their offline parser, guard and sequencing tests run in the repository's tooling
check. Live fux/provider execution of that maintained runner still requires a
separately authorized run; process mocks do not establish live success.

A separate expiry probe takes up to 70 minutes and at most nine requests, with
16 output tokens per request. It uses independent short-expiry, long-window and
long-expiry prefixes so the six-minute read cannot extend the prefix observed
at 65 minutes. It aborts as inconclusive if eligibility, usage or timing cannot
be established. Its 4,096-read threshold is a conservative evidence criterion,
not a claimed model cache minimum. Only run it after approving account usage:

```sh
CLAUDE_CLOAK_LIVE_EXPIRY=1 \
CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5 \
cargo test --locked -p rig-claude-cloak --lib transport::expiry::real_cache_expiry -- --ignored --exact --nocapture
```

The ignored `oauth_client_uses_child_scoped_certificate_trust` regression uses
local TLS and ephemeral certificates, not an account or inference. It requires
`openssl` and verifies untrusted rejection and child-only CA trust in fresh
processes without changing global trust settings.

## Request comparison method

The reference captures execute OMP's `streamAnthropic` with an injected fetch
that records the original destination and sends to a localhost HTTP server.
Rig executes its catalog connection, completion adapter, Anthropic encoder and
OAuth transport using test-only credentials, identity and sender. The sender
redirects to localhost after the production destination guard and request
shaping. Captures pin test-only platform headers to the reference Linux/arm64
profile, independent of the test host. Production headers use the host platform.
Both servers return synthetic responses; production restrictions remain unchanged.

Historical inputs use identical prompts, messages, fake access token, account
UUID, session UUID and installation seed. URL and stable application headers
are compared directly. Profile-specific user agent, SDK, beta sets and
compression mode are checked against the current reference and current
compression policy. Both original and current billing fingerprints and
checksums are verified before profile text is projected away. The declared
thinking-display and cache-none corrections are the only other body projections;
tool schemas, cache topology, metadata and remaining JSON order are compared
as bytes. This establishes the preserved historical contract, not current
byte equality with OMP. Real TUI comparison and its limits are described in
`tests/acceptance/README.md`.

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

Historical compression fixtures retain their original 1024-byte OMP boundary.
Current compression tests use the 4096-UTF-16-unit threshold and verify proxy
and custom-CA suppression separately. The effort beta retains original thinking intent even when forced tools
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
