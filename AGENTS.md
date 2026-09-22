# AGENTS.md

Operational instructions for AI coding agents working in Rig. Contributor
policy and PR etiquette live in [CONTRIBUTING.md](CONTRIBUTING.md); verification,
review, and publication in [DEVELOPING.md](DEVELOPING.md); test conventions and
commands in [tests/README.md](tests/README.md).
This file states rules, not APIs: read the code and manifests for names,
signatures, module paths, and feature flags.

## First Principles

- Read the existing implementation before changing code.
- Keep changes scoped to the user's request.
- Prefer existing Rig traits, builders, modules, and error types over new abstractions.
- Do not add TODOs, stubs, placeholder implementations, speculative APIs, or backwards compatibility shims.
- Do not make commits, comments, stage changes, push branches, or open PRs unless the user explicitly asks.
- Do not discard user changes.

## Efficient Investigation

- Reuse instructions and unchanged source already in context. Read the relevant
  sections of linked documents, not every document for every task.
- Use the repository map below to narrow searches to the owning crate or test
  target. Read complete relevant constructs; widen only when needed.
- Check actual signatures and existing test conventions before writing examples
  or smoke programs, rather than discovering APIs through compiler errors.
- Keep commands and edits on the intended worktree. Confirm the language server
  indexes it before trusting references; empty results outside its workspace
  are not proof of no callers. Inspect native/WASM branches explicitly.
- Request only decision-relevant output. Recover missing diagnostic context from
  truncated output rather than dumping or fetching the entire log again.

## Repository Shape

- Root facade crate: `rig` (re-exports the core crate, exposes companion crates behind feature flags)
- Core crate: `crates/rig-core`
- Companion provider, vector-store, memory, and integration crates: `crates/rig-*`
- Derive macros: `crates/rig-derive`
- Verification planner and source-tree checks: `xtask/`
- Workspace example packages: `examples/*`; per-crate examples: `crates/<crate>/examples/`
- Root integration test targets: `tests/*.rs`
- Record/replay home: `crates/rig-cassette` — the engine (`src/`), the provider cassette corpus (`fixtures/cassettes/<provider>/`), the effect-log golden corpus (`fixtures/effects/`), every cassette-backed provider target (`tests/<provider>.rs` with `tests/providers/<provider>/` and the shared drivers in `tests/common/`), the cache-prefix guard, and the effect-bus verification targets `verify` and `world_replay`. Corpora and `tests/` are excluded from the published package.
- Live-only provider test modules stay in the facade: `tests/providers/<provider>/`, driven by `tests/<provider>.rs`
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

## Vector Store Changes

Vector stores should live in companion crates. Implement both the scored-document and the
id-only search methods, use an appropriate backend-specific filter type,
return the vector-store error enum's variants instead of ad hoc string
errors, and use the WASM-compatible bounds.

## Style

- Module docs state the module's purpose in at most three paragraphs and include
  one short example when there is a public entry point. No headers, tables,
  history, or rhetorical phrasing.
- Item docs state the contract, inputs, outputs, errors, and caller invariants.
  Allow one sentence of non-obvious rationale. No issue numbers, bug narratives,
  or comparisons with other providers.
- Inline comments explain a non-obvious why in one or two sentences. Delete
  restated code, control-flow narration, and design discussions.
- Use short sentences and no em-dashes. Keep true `# Safety` and `# Panics`
  sections. Doc examples must compile with current APIs; use `no_run` for
  credentials, network access, or services.
- Move needed design rationale to the crate's CONTRACT.md or README, checking
  for and merging existing coverage rather than duplicating it.
- Follow local naming, module layout, and test patterns.
- Test modules are sibling files, never inline blocks: write
  `#[cfg(test)] mod tests;` and put the body in `foo/tests.rs` (for `foo.rs`)
  or `tests.rs` beside `mod.rs`/`lib.rs`. `cargo xtask check-test-layout`
  enforces this in CI.
- Avoid unrelated refactors.
- Prefer the cleanest implementation.
- Break things if the result is better.

## Cassette Regression Tests

Provider regressions should usually include cassette-backed tests. Follow
[tests/README.md](tests/README.md#cassette-provider-tests) for replay/record
commands and fixture safety review; never record or rewrite fixtures merely
to make verification pass.

## Verification

Follow [DEVELOPING.md](DEVELOPING.md) for check selection, independent review,
and authorized publication. Run the smallest useful local check; docs-only
edits get diff and link review, not Rust builds. Keep progress reports outside
the repository and never claim merge readiness with required checks pending.
