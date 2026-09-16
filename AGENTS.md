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
- Do not add TODOs, stubs, placeholder implementations, or speculative APIs.
- Do not make commits, stage changes, push branches, or open PRs unless the user explicitly asks.
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

## Core Architecture

Rig is built around a small set of provider-agnostic traits — one each for
completion models, embedding models, vector-store indexes, and tools. Use
them instead of creating parallel abstractions.

Configurable public types follow a builder style: construct through the
client, chain setters, finish with `build()`.

Provider clients share one generic client type parameterised over a provider
value and an HTTP transport. The provider value describes the wire (base URL,
credential type, builder configuration, URI assembly, request customisation,
environment construction). Each capability a provider supports is one more
capability-trait implementation on that same provider type, naming the
concrete model type; a capability the provider lacks is a trait it does not
implement. Blanket impls in the core client module turn those into the
user-facing client traits, so callers get the provider's own model types.
The core crate names no HTTP transport; the default transport is supplied by
a companion crate and erased behind a boxed type.

## WASM Compatibility

Rig supports WebAssembly targets. Use Rig's WASM-compatible `Send`/`Sync`
alias bounds instead of raw `Send` and `Sync`, and its WASM-compatible boxed
future alias for boxed futures. When an error type stores boxed errors, gate
the `Send + Sync` bound on `target_family = "wasm"` the way existing error
types do.

## Error Handling

- Do not use `String` as an error type for new fallible APIs; use explicit error enums with `thiserror`.
- Do not use `.unwrap()` or `.expect()` on fallible operations unless the condition is genuinely impossible and obvious from the code. Workspace clippy lints forbid `unwrap`, `expect`, `todo`, and `unimplemented`.
- Prefer `?` and meaningful error conversions.
- A provider's non-2xx reply must reach the caller through the capability error's single provider-response funnel, carrying status, body, response headers, and request id, so retry and status logic can inspect them. The transport-error variant is reserved for failures that produced no provider reply and never carries a status.

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

Before implementing or modifying a provider, study the closest existing
provider implementation and mirror it. Providers whose wire format is a
dialect of an existing one (OpenAI-chat-compatible, Anthropic-compatible) MUST
reuse the shared generic completion model for that dialect and express their
differences through its extension hooks — never a hand-rolled completion
model, request struct, or message conversion.

A provider implementation consists of:

- one provider value type named for the provider, holding whatever its
  requests need; credential-bearing fields get a redacting `Debug` impl
- the provider trait impl (name, base URL, verification path, credential type,
  builder config, environment input, constructors, and request customisation
  only when needed)
- one capability-trait impl per supported capability, naming the concrete
  model type; unsupported capabilities are simply not implemented
- public client and client-builder type aliases over the generic client, and
  re-exports of the provider type and every model type from the module root
- explicit credential marker/auth types that insert the intended headers and
  redact debug output
- model constants where useful, current with the provider's real API
- request conversion from Rig request types without adding fields the provider
  API does not support
- response conversion into Rig response types, including token usage and tool
  or multimodal content where applicable
- streaming support, following existing streaming normalization patterns, when
  the provider supports it
- provider-response error preservation through the funnel described under
  Error Handling
- provider-response metadata, telemetry spans, and GenAI attributes following
  existing conventions
- tests or examples appropriate to the provider

Do not add request or response fields that do not exist in the provider's real API.

For provider bug fixes or behavior changes, add or update regression coverage in
one of these places, preferring the smallest reliable scope:

- unit tests near the implementation;
- cassette-backed provider tests in `tests/providers/<provider>/cassette/`;
- ignored live tests only when cassette replay is unsuitable.

## Vector Store Changes

Vector stores should live in companion crates unless there is a strong reason to
place them in the core crate. Implement both the scored-document and the
id-only search methods, use an appropriate backend-specific filter type,
return the vector-store error enum's variants instead of ad hoc string
errors, and use the WASM-compatible bounds.

## Agent Hook Changes

Agent hooks are per-run lifecycle observers and steerers: one method per
lifecycle event, each receiving the run-scoped context and returning an
event-specific action type so unsupported combinations are rejected by the
compiler. Composition through the hook stack is event-dependent — some events
accumulate and merge patches in registration order, some chain each hook's
output into the next, and stop/deny actions short-circuit the remaining hooks.
Read the hook module docs for the exact rules per event before changing them.

Invariants to preserve:

- Every effect (completions, tool calls, memory operations, retrievals) crosses
  the dispatch boundary and is visible to the dispatch/outcome hooks; internal
  effect families are observe-only unless a hook opts in.
- The run-settled event fires exactly once per run, before the run's error
  reaches the consumer on any surface.
- Register observe-only hooks before steering hooks, because stop actions
  short-circuit. Nested hook stacks preserve merge and chaining semantics.
- Per-turn request patches are non-sticky; keep their documented merge rules
  (append context, shallow-merge params, intersect active tools,
  last-writer-wins scalars/history with a warning).
- Every hook semantic behaves identically on the streaming and non-streaming
  surfaces, which share one driver.

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
