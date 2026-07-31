# Reviewing PR #2228

806 files, +78k/−70k, 63 commits. This is not reviewable line by line, and
pretending otherwise wastes a reviewer's attention on mechanical sweeps. This
map says what to actually read, what to skim, and what to take on evidence.

## What the PR does

Converts Rig from trait-and-generic polymorphism to data. Providers become
serde `Config` structs plus free `complete`/`open_stream` functions; the agent
runtime loses its model type parameter (`Agent<M>` → `Agent`); hooks become
records instead of trait impls; vector stores expose concrete inherent methods
with no shared trait; system instructions have exactly one representation.

## The evidence that substitutes for line-by-line review

**Cassette fixtures are byte-identical across the entire PR.**
`git diff --stat -- tests/cassettes` is empty while ~59k lines of crate code
were rewritten. Roughly 730 recorded provider exchanges replay unchanged
through a completely new dispatch path. If a provider's request construction
had drifted, these fail — they are the reason a change this size is defensible
at all.

Where that evidence does **not** reach, and what covers it instead:

| Gap | Cover |
| --- | --- |
| Cohere has no cassettes (suite is ignored) | `cohere/functions.rs` conversion test |
| Bedrock, Vertex AI, Gemini gRPC, Candle are outside the replay claim | one conversion test each, asserting placement *and* exclusion from message arrays |
| Span attributes are not in recorded bytes | telemetry unit tests in `telemetry/mod.rs` |

## Read these (the actual risk)

1. **`crates/rig-agent/src/provider.rs`** — the `ProviderConfig` enum and
   dispatch. This is the architectural centre. See the accepted tradeoff below.
2. **`crates/rig-core/src/completion/request.rs`** — `CompletionRequest` and
   its builder. The legacy `preamble` field was deleted; system instructions
   are canonical `Message::System` entries only.
3. **`crates/rig-agent/src/hooks.rs`** — hooks as records (`HookEntry`,
   `HookEvent`, `HookDecision`) with the fold semantics that replaced
   `HookStack`.
4. **`crates/rig-core/src/telemetry/mod.rs`** — `completion_span` and
   `system_instructions_json`. Both now derive from the request, so telemetry
   and the wire body cannot disagree.
5. **`crates/rig-core/src/providers/openai/responses_api/mod.rs`** —
   `SystemInstructionsPlacement`. The one genuinely configurable wire-placement
   policy; ChatGPT and Copilot depend on non-default variants.

## Skim these (mechanical sweeps, compiler- or script-driven)

- Every `crates/*/examples/` and `tests/providers/` change: builder adoption,
  `ProviderConfig` unwrapping, `Box::pin` removal, `HookEntry::sync` adoption.
- The `preamble: None,` deletions (90 lines) and `preamble: Some(..)`
  conversions (29 sites).
- Provider `if let Some(preamble)` branch removals — deletions, because each
  provider already handled system messages arriving through the history.

## Accepted risks (decided, not overlooked)

**`ProviderConfig` is deliberately not `#[non_exhaustive]`.** Adding a provider
is a breaking change *by design*: every fulfilment site and every downstream
`match` must handle it, which is the property the architecture is built on. The
cost is real — out-of-tree providers cannot add a variant and must drive
`AgentRun` + `prepare_request` directly (documented in `MIGRATING.md`, worked
example in `rig-vertexai/examples/tool_vertexai.rs`). This is the single
decision most likely to draw objection and it is being surfaced deliberately
rather than left to be discovered inside an 806-file diff.

**`extra_headers` is not redacted in `Debug`.** `ApiKeyLocation` redacts inline
keys, but every provider `Config` derives `Debug` with
`pub extra_headers: Vec<(String, String)>` printing verbatim — and
`extra_headers` is where a bearer token goes when a provider's auth does not
fit the `api_key` slot. Mitigating factor: `Agent`'s `Debug` omits `provider`
via `finish_non_exhaustive()`, so the common logging path is safe; the exposure
is a direct `{:?}` of a `Config` or `ProviderConfig`. Shipping documented; the
fix would be a hand-rolled `Debug` redacting known auth header names.

## Known-missing

- No reserved-key rejection: nothing rejects `system` / `instructions` /
  `system_instruction` arriving through `additional_params`. The one concrete
  case found (Gemini Interactions' input-side channel) was removed, and a grep
  found no other provider reading instruction-shaped keys — but the general
  guard does not exist.

## Verification

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features          # check the exit code, not a pass count
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo test --workspace --all-features --doc
cargo check --target wasm32-unknown-unknown -p rig-core -p rig-agent
git diff --stat -- tests/cassettes              # must be empty
```

Three traps this repo sets for anyone verifying it, all hit during development:

- A pass-count summary **cannot see a failure**. Read `$?` and count
  `test result: FAILED` lines.
- `cargo check --all-targets` does **not** compile doctests.
- `cargo check --examples` prints `no targets matched` and still **exits 0**.
  Sixteen companion examples are gated behind `rig-core/derive`, which
  `--all-features` on the companion crate does not enable.

Cassette suites replay with `--test-threads=1`.
