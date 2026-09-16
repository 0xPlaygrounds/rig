# AGENTS.md

Operational instructions for AI coding agents working in Rig. Contributor
policy and PR etiquette live in `CONTRIBUTING.md`; verification, review, and
publication workflow in `DEVELOPING.md`; test commands in `tests/README.md`.
This file states rules, not APIs: read the code and manifests for names,
signatures, module paths, and feature flags.

## First Principles

- Read the existing implementation before changing code.
- Keep changes scoped to the user's request.
- Prefer existing Rig traits, builders, modules, and error types over new abstractions.
- Do not add TODOs, stubs, placeholder implementations, speculative APIs, or backwards compatibility shims.
- Do not make commits, comments, stage changes, push branches, or open PRs unless the user explicitly asks.
- Do not discard user changes.

## Repository Shape

- Root facade crate: `rig` (re-exports the core crate, exposes companion crates behind feature flags)
- Core crate: `crates/rig-core`
- Companion provider, vector-store, memory, and integration crates: `crates/rig-*`
- Derive macros: `crates/rig-derive`
- Verification planner and source-tree checks: `xtask/`
- Workspace example packages: `examples/*`; per-crate examples: `crates/<crate>/examples/`
- Root integration test targets: `tests/*.rs`
- Provider test modules: `tests/providers/<provider>/`; cassette fixtures: `tests/cassettes/<provider>/`
- External-service integration tests: `tests/integrations/`
- Unpublished vector-store test runner: `test-support/service-tests`

Check the root manifest and facade source before documenting or changing
exposed features, integrations, or module paths. When adding or exposing a
companion crate, update the root dependency, feature, facade re-export,
examples, README, and crate docs as applicable.

## WASM Compatibility

Rig supports WebAssembly targets. Use Rig's WASM-compatible `WasmSend`/`WasmSync`
alias bounds instead of raw `Send` and `Sync`, and its WASM-compatible boxed
future alias for boxed futures. When an error type stores boxed errors, gate
the `Send + Sync` bound on `target_family = "wasm"` the way existing error
types do.

## Error Handling

- Do not use `String` as an error type for new fallible APIs;
- Workspace clippy lints forbid `unwrap`, `expect`, `todo`, and `unimplemented`.
- Prefer `?` and meaningful error conversions.

## Documentation

- Keep examples current with actual APIs, model constants, module paths, and feature flags; mark examples `no_run` when they require external credentials or services.
- Do not document integrations, features, model constants, or crate paths without checking the code and manifests.
- Keep root README, crate READMEs, and crate-level Rust docs consistent when changing public-facing behavior.

## Release Documents Are Generated

- Never edit `CHANGELOG.md`, `crates/*/CHANGELOG.md`, `MIGRATING.md`, or `docs/migrations/`. CI rejects the PR.
- Put changelog bullets and migration notes in the PR description under `## Changelog` and `## Migration`, following the PR template. When the user asks to "update the changelog" or "add a migration note", that is what it means. The repository squash-merges, so those sections become the merge commit body.
- These files are regenerated on the release PR by `scripts/release-notes.sh` and the editorial pass in `scripts/release-notes-prompt.md`. Only touch them when the user explicitly says the branch is a release PR.

## Provider Changes

Before implementing or modifying a provider, study the closest existing
provider implementation and mirror it. Providers whose wire format is a
dialect of an existing one (OpenAI-chat-compatible, Anthropic-compatible) MUST
reuse the shared generic completion model for that dialect and express their
differences through its extension hooks — never a hand-rolled completion
model, request struct, or message conversion.

For provider bug fixes or behavior changes, prefer recording cassette backed provider tests.

## Vector Store Changes

Vector stores should live in companion crates. Implement both the scored-document and the
id-only search methods, use an appropriate backend-specific filter type,
return the vector-store error enum's variants instead of ad hoc string
errors, and use the WASM-compatible bounds.

## Style

- Comments should explain why, not restate what the code does.
- Follow local naming, module layout, and test patterns.
- Test modules are sibling files, never inline blocks: write
  `#[cfg(test)] mod tests;` and put the body in `foo/tests.rs` (for `foo.rs`)
  or `tests.rs` beside `mod.rs`/`lib.rs`. `cargo xtask check-test-layout`
  enforces this in CI.
- Avoid unrelated refactors.
- Prefer the cleanest implementation.
- Break things if the result is better.

## Cassette Regression Tests

Provider regressions should usually include cassette-backed tests. Read
`tests/README.md` for layout, replay/record commands, and the cassette diff
review checklist before adding, updating, or running provider tests. Replay
needs no API keys; record mode needs the provider's key and overwrites fixtures,
so keep record runs targeted to the provider and test being changed. The scrub
checks in `tests/common/cassette_safety.rs` are not a substitute for inspecting
generated fixtures before presenting changes. Make sure cassettes never leak keys before pushing. 

## Verification

Run the smallest useful local check for the changed behavior; documentation
edits get diff and link review only. Obtain independent full-diff review, fix
confirmed P0/P1 findings, and publish promptly when authorized — CI is the
comprehensive gate. Never claim fully verified or ready to merge while required
checks are pending or failing. `DEVELOPING.md` is the complete workflow:
check selection, `cargo xtask verify` modes, review loop, publication, and
CI completion. Keep progress reports outside the repository.
