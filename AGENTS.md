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
- Root examples: `examples/`
- Per-crate examples: `crates/<crate>/examples/`
- Provider-backed tests: `tests/providers/` and provider-specific integration tests

The root `rig` crate re-exports `rig-core` and exposes companion crates behind
feature flags. Check `Cargo.toml` before documenting or changing exposed
features, integrations, or module paths.

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
- client type aliases
- model constants where useful
- request conversion from Rig request types
- response conversion into Rig response types
- streaming support when the provider supports streaming
- telemetry spans following existing GenAI conventions
- tests or examples appropriate to the provider

Do not add request or response fields that do not exist in the provider's real API.

## Vector Store Changes

Vector stores should live in companion crates unless there is a strong reason to
place them in `rig-core`.

Implement both:

- `top_n`
- `top_n_ids`

Use an appropriate backend-specific filter type.

Return `VectorStoreError` variants instead of ad hoc string errors.

Use `WasmCompatSend` and `WasmCompatSync` bounds.

## Prompt Hook Changes

Prompt hooks are per-request lifecycle hooks.

When modifying hook behavior, preserve the intended control flow:

- `HookAction::Continue`
- `HookAction::Terminate`
- `ToolCallHookAction::Continue`
- `ToolCallHookAction::Skip`
- `ToolCallHookAction::Terminate`

Check both streaming and non-streaming paths.

## Style

- Use full `where` clauses for complex trait bounds.
- Comments should explain why, not restate what the code does.
- Follow local naming, module layout, and test patterns.
- Avoid unrelated refactors.

## Verification

Run the smallest useful checks first, then broaden as needed.

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
