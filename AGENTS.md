# AGENTS.md

Operational instructions for AI coding agents working in Rig. Contributor
policy and PR etiquette live in `CONTRIBUTING.md`; verification, review, and
publication workflow in `DEVELOPING.md`; test commands in `tests/README.md`.

## First Principles

- Read the existing implementation before changing code.
- Keep changes scoped to the user's request.
- Prefer existing Rig traits, builders, modules, and error types over new abstractions.
- Do not add TODOs, stubs, placeholder implementations, or speculative APIs.
- Do not make commits, stage changes, push branches, or open PRs unless the user explicitly asks.
- Do not discard user changes.

## Repository Shape

- Root facade crate: `rig` (re-exports `rig-core`, exposes companion crates behind feature flags)
- Core crate: `crates/rig-core`
- Companion provider, vector-store, memory, and integration crates: `crates/rig-*`
- Derive macros: `crates/rig-derive`
- Verification planner and source-tree checks: `xtask/`
- Workspace example packages: `examples/*`; per-crate examples: `crates/<crate>/examples/`
- Root integration test targets: `tests/*.rs`
- Provider test modules: `tests/providers/<provider>/`; cassette fixtures: `tests/cassettes/<provider>/`
- External-service integration tests: `tests/integrations/`
- Unpublished vector-store test runner: `test-support/service-tests`

Check `Cargo.toml` and `src/lib.rs` before documenting or changing exposed
features, integrations, or module paths. When adding or exposing a companion
crate, update the root dependency, feature, facade re-export, examples, README,
and crate docs as applicable.

## Core Architecture

Rig is built around provider-agnostic traits:

- `CompletionModel` for text completion and chat models
- `EmbeddingModel` for embedding generation
- `VectorStoreIndex` for vector similarity search
- `Tool` for callable tools

Use these traits instead of creating parallel abstractions.

Configurable public types should follow Rig's builder style:

```rust
let agent = client
    .agent(openai::GPT_5_2)
    .preamble("System prompt")
    .tool(my_tool)
    .temperature(0.8)
    .build();
```

Provider clients use the generic client architecture in `rig_core::client`:

```rust
pub struct Client<P, H = BoxedHttpClient> {
    // base URL, default headers, transport `H`, provider value `P`
}
```

`P` is a value type implementing `Provider` (base URL, API-key type, builder
`Config`, URI assembly, per-request customisation, environment construction).
Each capability a provider offers is one more trait implementation on the same
type — `HasCompletion`, `HasEmbeddings`, `HasRerank`, `HasTranscription`,
`HasModelListing`, `HasImageGeneration` (feature `image`), `HasAudioGeneration`
(feature `audio`) — whose `Model<H>` names the concrete model. The blanket
impls in `rig_core::client` turn those into the user-facing `CompletionClient`,
`EmbeddingsClient`, … traits on `Client<P, H>`, so `client.completion_model(..)`
returns the provider's own model type. A capability a provider lacks is a trait
it does not implement. rig-core names no transport: `H` defaults to the erased
`BoxedHttpClient`, and `rig-reqwest` supplies `new(key)` / `from_env()` / a
transport-less `build()` through `DefaultTransportClient` /
`DefaultTransportBuilder` (re-exported by the `rig` facade prelude).

## WASM Compatibility

Rig supports WebAssembly targets. Use `WasmCompatSend` and `WasmCompatSync` in
trait bounds instead of raw `Send` and `Sync`, and `WasmBoxedFuture` for boxed
futures. When an error type stores boxed errors, use platform-specific bounds:

```rust
#[cfg(not(target_family = "wasm"))]
Box<dyn std::error::Error + Send + Sync + 'static>

#[cfg(target_family = "wasm")]
Box<dyn std::error::Error + 'static>
```

## Error Handling

- Do not use `String` as an error type for new fallible APIs; use explicit error enums with `thiserror`.
- Do not use `.unwrap()` or `.expect()` on fallible operations unless the condition is genuinely impossible and obvious from the code. Workspace clippy lints forbid `unwrap`, `expect`, `todo`, and `unimplemented`.
- Prefer `?` and meaningful error conversions.

## Documentation

- Add `///` docs to new public items and `//!` docs to new public modules.
- Keep examples current with actual APIs, model constants, module paths, and feature flags; mark examples `no_run` when they require external credentials or services.
- Do not document integrations, features, model constants, or crate paths without checking the code and manifests.
- Keep root README, crate READMEs, and crate-level Rust docs consistent when changing public-facing behavior.

## Release Documents Are Generated

- Never edit `CHANGELOG.md`, `crates/*/CHANGELOG.md`, `MIGRATING.md`, or `docs/migrations/`. CI rejects the PR.
- Put changelog bullets and migration notes in the PR description under `## Changelog` and `## Migration`, following the PR template. When the user asks to "update the changelog" or "add a migration note", that is what it means. The repository squash-merges, so those sections become the merge commit body.
- These files are regenerated on the release PR by `scripts/release-notes.sh` and the editorial pass in `scripts/release-notes-prompt.md`. Only touch them when the user explicitly says the branch is a release PR.

## Provider Changes

Before implementing or modifying a provider, study the closest existing provider
implementation. For OpenAI-compatible chat APIs, start with
`crates/rig-core/src/providers/openai/`.

- New OpenAI-chat-compatible providers MUST drive completions through
  `openai::completion::GenericCompletionModel<Ext>` by implementing
  `OpenAICompatibleProvider` on the provider extension (see `minimax`, `zai`,
  `groq`, or `deepseek`). Wire-dialect differences belong in the trait's hooks
  (`completion_path`, `prepare_request`, `finalize_request_body`) — not in a
  hand-rolled `CompletionModel`, request struct, or `TryFrom<message::Message>`
  conversion. Anthropic-shaped APIs use `AnthropicCompatibleProvider` the same way.

Provider implementations should include, in the order a new author writes them:

- one provider value type named for the provider (`Perplexity`, `Azure`, …),
  `Clone + Debug`, holding whatever the requests need (a query-string key, an
  endpoint); credential-bearing fields get a redacting `Debug` impl
- a `Provider` impl: `NAME`, `BASE_URL`, `VERIFY_PATH`, `type ApiKey`
  (`BearerAuth`, `Nothing`, or a provider key type implementing `ApiKey`),
  `type Config` (`()` unless the builder needs settings; otherwise a plain
  `*Config` struct with setters on `ClientBuilder<H>`), `type EnvInput`,
  `build`, `from_env` (`Client::from_env_api_key` covers the common shape),
  `from_val`, and — only when needed — `finish`, `build_uri`, `prepare`
- one `Has*` impl per supported capability, each naming the concrete model
  type in `Model<H>` and constructing it from `&Client<Self, H>`
- public `Client<H = BoxedHttpClient>` and `ClientBuilder<H = Missing>` aliases
  over `client::Client<Provider, H>` / `client::ClientBuilder<Provider, H>`,
  and re-exports of the provider type and every model type from the module root
- explicit API-key marker/auth types that insert the intended headers, with redacted debug behavior for credential-bearing values
- model constants where useful, current with the provider's real API
- request conversion from Rig request types without adding fields the provider API does not support
- response conversion into Rig response types, including token usage and tool or multimodal content where applicable
- streaming support, following existing streaming normalization patterns, when the provider supports it
- provider-response error preservation: non-2xx completion responses surface
  through the capability error's one funnel, `from_http_response(status, body)`
  (or `?` on the transport error, which routes through it), stamped with
  `with_provider_request_id` and `with_response_headers` when the call site has
  them, so retry/status logic can inspect `provider_response_status()`, the raw
  provider body, the request id and `Retry-After`. `HttpError` /
  `ErrorKind::Http` is only a transport failure that produced no provider reply
  and never carries a status; a provider reply classifies as
  `ErrorKind::ProviderResponse` on every wire
- `ProviderResponseExt`, telemetry spans, and GenAI fields following existing conventions
- tests or examples appropriate to the provider

Do not add request or response fields that do not exist in the provider's real API.

For provider bug fixes or behavior changes, add or update regression coverage in
one of these places, preferring the smallest reliable scope:

- unit tests near the implementation in `crates/rig-core/src/providers/...`;
- cassette-backed provider tests in `tests/providers/<provider>/cassette/`;
- ignored live tests only when cassette replay is unsuitable.

## Vector Store Changes

Vector stores should live in companion crates unless there is a strong reason to
place them in `rig-core`. Implement both `top_n` and `top_n_ids`, use an
appropriate backend-specific filter type, return `VectorStoreError` variants
instead of ad hoc string errors, and use `WasmCompatSend` / `WasmCompatSync` bounds.

## Agent Hook Changes

Agent hooks are per-run lifecycle observers and steerers. `AgentHook` exposes
one method per lifecycle event, and every method receives the run-scoped
`HookContext` (run id, turn, streaming flag, agent name, shared `Scratchpad`).
Each method returns an event-specific action type, so unsupported combinations
are rejected by the compiler.

Composition through `HookStack` remains event-dependent:

- **Completion calls accumulate and merge.** Every
  `CompletionCallAction::Patch(RequestPatch)` is merged in registration order;
  `Stop` short-circuits the stack.
- **Every effect crosses the dispatch boundary.** Completions, tool calls,
  memory loads and appends, and retrievals are dispatched on the agent's bus,
  and `on_dispatch` / `on_outcome` see each of them: each hook's
  `DispatchAction::Patch` is what the next hook sees, the first
  `DispatchAction::Deny` wins (a denied tool call is the skipped result the
  model sees), and each `OutcomeAction::Replace` is what the next hook sees.
  The internal families (`Memory`, `Retrieve`, `Embed`, `Rerank`, `Custom`)
  are observe-only until a hook opts in through `observes`.
- **Model turns** return `ModelTurnAction` (`Continue`, `Retry`, or `Stop`);
  a retry or stop short-circuits the remaining hooks for that event.
- **Invalid tool calls** return `InvalidToolCallAction` (`Fail`, `Retry`,
  `Repair`, `Skip`, or `Stop`).
- **Observe-only events** return `ObservationAction` (`Continue` or `Stop`).
- **`RunSettled` fires exactly once per run**, before the run's error reaches
  the consumer on any surface.

Register observe-only hooks before steering hooks because stop actions
short-circuit. Nested `HookStack`s must preserve merge and chaining semantics.
`RequestPatch` remains per-turn and non-sticky; its documented merge rules are
append `extra_context`, shallow-merge `additional_params`, intersect
`active_tools`, and last-writer-wins scalars/history with a warning.

Every hook semantic must behave identically on streaming and non-streaming
surfaces (`AgentRunner::stream` and `AgentRunner::run` share `drive_agent`).

## Style

- Use full `where` clauses for complex trait bounds.
- Comments should explain why, not restate what the code does.
- Follow local naming, module layout, and test patterns.
- Test modules are sibling files, never inline blocks: write
  `#[cfg(test)] mod tests;` and put the body in `foo/tests.rs` (for `foo.rs`)
  or `tests.rs` beside `mod.rs`/`lib.rs`. `cargo xtask check-test-layout`
  enforces this in CI.
- Avoid unrelated refactors.

## Cassette Regression Tests

Provider regressions should usually include cassette-backed tests. Read
`tests/README.md` for layout, replay/record commands, and the cassette diff
review checklist before adding, updating, or running provider tests. Replay
needs no API keys; record mode needs the provider's key and overwrites fixtures,
so keep record runs targeted to the provider and test being changed. The scrub
checks in `tests/common/cassette_safety.rs` are not a substitute for inspecting
generated fixtures before presenting changes.

## Verification

Run the smallest useful local check for the changed behavior; documentation
edits get diff and link review only. Obtain independent full-diff review, fix
confirmed P0/P1 findings, and publish promptly when authorized — CI is the
comprehensive gate. Never claim fully verified or ready to merge while required
checks are pending or failing. `DEVELOPING.md` is the complete workflow:
check selection, `cargo xtask verify` modes, review loop, publication, and
CI completion. Keep progress reports outside the repository.
