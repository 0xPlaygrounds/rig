# AGENTS.md

Operational instructions for AI coding agents working in Rig.

For contributor-facing policy, PR expectations, and accountability guidance, see
`CONTRIBUTING.md`. This file is for repository-specific engineering rules that
agents must follow while reading, editing, testing, and documenting code.

## First Principles

- Read the existing implementation before changing code.
- Keep changes scoped to the user's request.
- Prefer existing Rig traits, builders, modules, and error types over new abstractions.
- Do not add TODOs, stubs, placeholder implementations, or speculative APIs.
- Do not make commits, stage changes, push branches, or open PRs unless the user explicitly asks.
- Do not discard user changes.

## Repository Shape

- Root facade crate: `rig`
- Core crate: `crates/rig-core`
- Companion provider and vector-store crates: `crates/rig-*`
- Derive macros: `crates/rig-derive`
- Workspace example packages: `examples/*`
- Per-crate examples: `crates/<crate>/examples/`
- Root integration test targets: `tests/*.rs`
- Provider test modules: `tests/providers/<provider>/`
- Provider cassette fixtures: `tests/cassettes/<provider>/`
- External-service integration tests: `tests/integrations/`

The root `rig` crate re-exports `rig-core` and exposes companion crates behind
feature flags. Check `Cargo.toml` and `src/lib.rs` before documenting or changing
exposed features, integrations, or module paths. If adding or exposing a
companion provider/vector-store crate, update the root dependency, feature,
facade re-export, examples, README, and crate docs as applicable.

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

Provider clients use the generic client architecture:

```rust
pub struct Client<Ext = Nothing, H = reqwest::Client> {
    // ...
}
```

Providers declare capabilities explicitly with `Capable<T>` and `Nothing`.

## WASM Compatibility

Rig supports WebAssembly targets.

Use `WasmCompatSend` and `WasmCompatSync` in trait bounds instead of raw `Send`
and `Sync`.

Use `WasmBoxedFuture` for boxed futures.

When an error type stores boxed errors, use platform-specific bounds:

```rust
#[cfg(not(target_family = "wasm"))]
Box<dyn std::error::Error + Send + Sync + 'static>

#[cfg(target_family = "wasm")]
Box<dyn std::error::Error + 'static>
```

## Error Handling

- Do not use `String` as an error type for new fallible APIs.
- Use explicit error enums with `thiserror`.
- Do not use `.unwrap()` or `.expect()` on fallible operations unless the condition is genuinely impossible and obvious from the code.
- Prefer `?` and meaningful error conversions.

## Documentation

- Add `///` docs to new public items.
- Add `//!` docs to new public modules.
- Keep examples current with actual APIs, model constants, module paths, and feature flags.
- Mark examples `no_run` when they require external credentials or services.
- Do not document integrations, features, model constants, or crate paths without checking the code and manifests.
- Keep root README, crate READMEs, and crate-level Rust docs consistent when changing public-facing behavior.

## Provider Changes

Before implementing or modifying a provider, study the closest existing provider
implementation. For OpenAI-compatible chat APIs, start with:

`crates/rig-core/src/providers/openai/`

Provider implementations should include:

- Provider extension and builder types
- `Provider` implementation
- `Capabilities` declaration
- `ProviderBuilder` implementation
- `ProviderClient::{from_env, from_val}`
- public `Client` and `ClientBuilder` aliases; the `ClientBuilder` API-key generic must match `ProviderBuilder::ApiKey`
- explicit API-key marker/auth types with redacted debug behavior for credential-bearing values
- model constants where useful
- request conversion from Rig request types
- response conversion into Rig response types
- streaming support when the provider supports streaming
- provider-response error preservation through the relevant Rig error helpers
- `ProviderResponseExt` and telemetry spans following existing GenAI conventions
- tests or examples appropriate to the provider

Do not add request or response fields that do not exist in the provider's real API.

For provider bug fixes or behavior changes, add or update regression coverage in
one of these places, preferring the smallest reliable scope:

- unit tests near the implementation in `crates/rig-core/src/providers/...`;
- cassette-backed provider tests in `tests/providers/<provider>/cassette/`;
- ignored live tests only when cassette replay is unsuitable.

## Vector Store Changes

Vector stores should live in companion crates unless there is a strong reason to
place them in `rig-core`.

Implement both:

- `top_n`
- `top_n_ids`

Use an appropriate backend-specific filter type.

Return `VectorStoreError` variants instead of ad hoc string errors.

Use `WasmCompatSend` and `WasmCompatSync` bounds.

## Agent Hook Changes

Agent hooks are per-run lifecycle observers and steerers. `AgentHook<M>` exposes
event-specific lifecycle methods, each receiving a run-scoped `HookContext`
(run id, turn, streaming flag, agent name, shared `Scratchpad`) and returning an
event-specific action. Invalid event/action combinations must remain
unrepresentable by the public types.

How a `HookStack` combines actions depends on the lifecycle method — this is the
central contract:

- **Completion calls — accumulate and merge.** Every hook runs; each
  `CompletionAction::Patch(RequestPatch)` is merged in registration order into
  one effective patch. A mergeable patch does **not** short-circuit later hooks;
  `CompletionAction::Stop` does.
- **Tool calls/results — chain.** Every hook runs; a
  `ToolCallAction::Rewrite` or `ToolResultAction::Rewrite` is threaded into the
  next hook's event so rewrites compose. Skip and stop actions are terminal.
- **Every other lifecycle — first non-default action wins.** This covers
  completion responses, finished model turns, invalid tool calls, and streamed
  deltas.

Register observe-only hooks (telemetry) before steering hooks because stop
actions short-circuit the stack. A `HookStack` pushed into another stack must
compose correctly, including preserving an inner argument rewrite when a later
inner hook skips or stops.

Preserve the event-specific action vocabulary:

- `CompletionAction`: continue, patch, or stop;
- `ObserveAction`: continue or stop;
- `ToolCallAction`: run, rewrite, skip, or stop;
- `ToolResultAction`: keep, rewrite, or stop;
- `InvalidToolCallAction`: continue/fail, retry, repair, skip, or stop.

`RequestPatch` is per-turn and non-sticky (it never mutates the agent baseline);
its per-field merge rules (append `extra_context`, shallow-merge
`additional_params`, intersect `active_tools`, last-writer-wins scalars/`history`
with a warning) are documented on the type. Every new hook semantic must behave
identically on both surfaces — check streaming and non-streaming paths
(`AgentRunner::stream` and `AgentRunner::run` share one drive loop, `drive_agent`).

## Style

- Use full `where` clauses for complex trait bounds.
- Comments should explain why, not restate what the code does.
- Follow local naming, module layout, and test patterns.
- Avoid unrelated refactors.

## Cassette Regression Tests

Provider regressions should usually include cassette-backed tests. Read
`tests/README.md` before adding, updating, or running provider tests.

- Test code lives under `tests/providers/<provider>/cassette/`.
- Fixtures live under `tests/cassettes/<provider>/...`.
- Replay cassettes by default; this should not require provider API keys.
- Record mode requires the relevant provider API key and overwrites fixtures.
- Keep record runs targeted to the provider and test being changed.

Replay examples:

```bash
cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test anthropic anthropic::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test gemini gemini::cassette -- --nocapture --test-threads=1
```

Record example:

```bash
RIG_PROVIDER_TEST_MODE=record \
  cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
```

Review cassette diffs carefully. They must not contain API keys, bearer tokens,
cookies, provider account identifiers, or unrelated request/response churn. The
repo includes cassette scrub/safety checks in `tests/common/cassette_safety.rs`,
but agents are still responsible for inspecting generated fixtures before
presenting changes.

## Verification

Run the smallest useful checks first, then broaden as needed. For tests, prefer
the targeted commands in `tests/README.md` before running broad workspace checks.

Before considering code complete, run when feasible:

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```

For documentation changes, also consider:

```bash
cargo doc --workspace --no-deps
```

If a command cannot be run, say why and tell the user exactly what remains
unverified.
