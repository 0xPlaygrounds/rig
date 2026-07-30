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

Rig is data-oriented: providers are serde configuration, not trait
implementations, and the runtime is not generic over a model type.

- Completion and streaming go through each provider's `functions` module — a
  serde `Config` plus free `complete`/`open_stream` functions.
- `ProviderConfig` (`rig-agent`) is the one enum an `Agent` holds. It is
  deliberately **not** `#[non_exhaustive]`: adding a provider is a breaking
  change by design, so hosts can match exhaustively.
- `ProviderDescriptor` (`rig-core/src/providers/descriptor.rs`) is the
  capability sheet for a provider.
- Embeddings, transcription, image generation, audio generation, and rerank
  are per-provider free functions over plain configs.
- Vector stores expose concrete inherent methods over a pre-embedded
  vocabulary. There is no shared store trait, and none should be added.
- `Tool` is the portable record contract (`rig_core::tool::PortableTool`).

Prefer concrete records and enums over new trait abstractions. Do not
reintroduce a generic model, client, or store parameter.

Configurable public types should follow Rig's builder style:

```rust
let agent = client
    .agent(openai::GPT_5_2)
    .preamble("System prompt")
    .tool(my_tool)
    .temperature(0.8)
    .build();
```

A provider is a serde `Config` plus free functions, not a generic client:

```rust
// crates/rig-core/src/providers/<provider>/functions.rs
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor { /* … */ };

pub struct Config { /* model, api_key, base_url, knobs — all serde */ }

pub async fn complete(cfg: &Config, rt: &HttpRuntime, request: CompletionRequest)
    -> Result<CompletionResponse, CompletionError>;
pub async fn open_stream(/* … */) -> Result<StreamingCompletionResponse, CompletionError>;
```

Capabilities are declared as data on the `DESCRIPTOR` const, not as trait
`const`s or marker types. I/O goes through `HttpRuntime`; keep request
building and response parsing as pure functions (`build_request_body`,
`parse_response`) so they are testable without a transport.

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

- a `functions` module holding the provider's serde `Config`
- a `DESCRIPTOR: ProviderDescriptor` const declaring capabilities honestly —
  fulfilment code reads it to fail fast, so a wrong flag is a runtime bug
- `Config::new`, `Config::from_env`, and `with_*` knob setters
- redacted `Debug` for credential-bearing values; never log an API key
- pure `build_request*` / `parse_*` functions, with `complete` and
  `open_stream` composing them over `HttpRuntime`
- a `ProviderConfig` variant in `rig-agent` so the provider can drive an agent
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

## Agent Hooks

Agent hooks are attach-and-forget **records**, not trait impls. A hook is a
`HookEntry` (`rig::hooks`) wrapping a named async callback over owned
`HookEvent` values, returning a `HookDecision`. `Hooks` is the ordered list;
`AgentBuilder::add_hook` takes one `HookEntry`.

```rust
use rig::hooks::{HookDecision, HookEntry, HookEvent};

fn logger() -> HookEntry {
    HookEntry::new("logger", |event| {
        let decision = match event {
            HookEvent::BeforeModelCall { turn, .. } => {
                tracing::info!(turn, "model call");
                HookDecision::Continue
            }
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
}
```

A callback answers with the `HookDecision` variant matching the event it
received; any other variant (including `Continue`) means "no opinion". The
decision vocabulary itself — `RequestPatch`, `CompletionCallAction`,
`ToolCallAction`, `ToolResultAction`, `InvalidToolCallAction`,
`ObservationAction`, `ModelTurnAction` — is unchanged and still lives at
`rig::agent::hook`.

There is no `HookContext` and no `Scratchpad`. Run identity and shared state
are host-owned: capture them in the closure (see
`examples/request_hook`). Note the lifetime difference from the old
run-scoped scratchpad — closure state lives as long as the `HookEntry` and is
shared by every clone of it, so it spans **all** runs of an agent and
interleaves across concurrent ones. State that must be per-run has to be keyed
or reset explicitly. `turn` is a field on the events that carry it.

`HookEvent::TextDelta` and `HookEvent::ToolCallDelta` fire once per streamed
token and are opt-in: an entry receives them only if it was built with
`HookEntry::observing_deltas()`. Drivers check `Hooks::observes_deltas()` once
per run and skip building delta events entirely when no entry opted in — the
data form of the old `observes(StepEventKind)` interest hint.

Composition is event-dependent:

- **Completion calls accumulate and merge.** Every
  `CompletionCallAction::Patch(RequestPatch)` is merged in registration order;
  the first `Stop` short-circuits, and later entries are not invoked.
- **Tool calls and results chain.** `ToolCallAction::Rewrite` and
  `ToolResultAction::Rewrite` are threaded into later entries. A tool-call
  `Skip` or either event's `Stop` is terminal, preserving the rewrite
  accumulated before it. Tool-call argument rewrites chain as
  `serde_json::Value`, not JSON-encoded strings.
- **Invalid tool calls** return `InvalidToolCallAction` (`Fail`, `Retry`,
  `Repair`, `Skip`, or `Stop`). The first `Some` resolution wins; `None`
  everywhere preserves fail-fast behavior.
- **Model-turn and observe-only events.** The first non-`Continue` wins.
  Observation events return `ObservationAction`.

Register observe-only entries before steering entries because stop actions
short-circuit. The folds live in `rig::agent::hook`
(`fold_completion_actions`, `fold_observation_actions`,
`fold_invalid_resolutions`, `ToolCallResolution`, `ToolResultResolution`) and
are shared by both drivers, so every driver composes decisions identically —
reuse them rather than reimplementing a fold.

`RequestPatch` remains per-turn and non-sticky; its documented merge rules are
append `extra_context`, shallow-merge `additional_params`, intersect
`active_tools`, and last-writer-wins scalars/history with a warning.

Every hook semantic must behave identically on the blocking and streaming
session drivers (`Agent::run` and `Agent::stream_run`).

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
