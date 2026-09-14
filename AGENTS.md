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
- Unpublished vector-store test runner: `test-support/service-tests`

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

## Release Documents Are Generated

- Never edit `CHANGELOG.md`, `crates/*/CHANGELOG.md`, `MIGRATING.md`, or `docs/migrations/`. CI rejects the PR.
- Put changelog bullets and migration notes in the PR description under `## Changelog` and `## Migration`, following the PR template. When the user asks to "update the changelog" or "add a migration note", that is what it means.
- These files are regenerated on the release PR by `scripts/release-notes.sh` and the editorial pass in `scripts/release-notes-prompt.md`. Only touch them when the user explicitly says the branch is a release PR.

## Provider Changes

Before implementing or modifying a provider, study the closest existing provider
implementation. For OpenAI-compatible chat APIs, start with:

`crates/rig-core/src/providers/openai/`

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

Default to minimal relevant local checks, prompt authorized publication, and comprehensive GitHub CI. Inspect the intended diff against its actual merge base, preserve unrelated changes, and check formatting/whitespace as applicable. Run the smallest useful check for changed behavior; for documentation or instruction edits, inspect the diff and links/consistency rather than running Rust compilation, full docs, or workspace tests.

`cargo xtask verify --pr --base <intended-base-ref>` and `--full` are optional for explicit requests or deliberate debugging, not prepublication gates. `--changed` is also optional: inspect its dry-run selection when it could expand broadly. Shared inputs and unknown files can select the full plan; choose explicit small checks or CI instead. Do not automatically run workspace/all-feature, all-provider/example, docs/doctest, WASM, Docker, or dependency-floor matrices locally or install expensive CI-only prerequisites. Explain the concrete need before an expensive local check. See [DEVELOPING.md](DEVELOPING.md) for the complete workflow and command selection semantics.

Finish implementation, examples, tests, migration notes, and scope review first. Review the full intended diff, including staged, unstaged, and relevant untracked files. Obtain independent full-diff review before publication. Validate findings, fix confirmed P0/P1 and in-scope lower-severity issues, or document why lower-severity findings should remain. After fixes, run useful targeted local checks and the final independent review; repeat if new confirmed P0/P1 issues appear. Do not restart the full local suite after each fix or rerun unchanged checks without a concrete reason. Keep progress reports outside the repository.

Commit/push/open a non-draft PR promptly when authorized; do not delay publication to duplicate CI locally. After publication, inspect required CI on the current committed head and actionable review feedback. Confirm selected CI jobs cover the task's comprehensive acceptance criteria; report and address missing coverage within scope. Absent or skipped coverage is not successful verification. Preserve CI coverage, required checks, feature-isolation matrices, regression tests, and assertions. Fix in-scope failures with targeted checks, then push when authorized for CI verification; never disable jobs or skip failing checks to speed publication.

Report publication separately from verification: “PR opened; CI pending” is valid for a publication handoff. Continue monitoring when the task asks for completion through CI; an ordinary PR-opening request does not require waiting for all CI. Never claim fully verified or ready to merge while required checks are pending/failing, confirmed P0/P1 findings remain, or required work is unresolved. Report the PR link, local checks actually run, review/CI state, and concrete blockers concisely; do not broaden scope to unrelated failures. These requirements do not independently authorize commits, pushes, PR creation, merging, or comments on PRs/issues. Do not open draft PRs or comment on PRs/issues unless asked.
