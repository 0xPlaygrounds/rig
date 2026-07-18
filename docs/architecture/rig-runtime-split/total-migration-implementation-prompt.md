# Implementation prompt: complete the Rig runtime split in this PR

Use this document as the complete implementation prompt for turning the runtime
split research package into production code. It is intentionally standalone:
the implementer must still inspect the repository and linked evidence, but
should not need another planning prompt to understand the target, constraints,
sequencing, or completion gate.

## Mission

Complete the full researched runtime migration in the implementation pull
request:

- narrow `rig-core` to portable contracts, canonical values, and low-level
  provider/backend integration boundaries;
- move the complete current classic agent runtime to a new `rig-agent` crate;
- build a new, genuinely ECS-native `rig-bevy` runtime as a sibling of
  `rig-agent`;
- update the root `rig` facade, feature graph, preludes, derive macros,
  companion crates, examples, tests, and documentation;
- establish shared behavioral conformance without sharing production
  orchestration;
- leave the full workspace, required targets, documentation, and CI green.

Do all implementation work in one unique implementation pull request. Do not
defer a required runtime slice to a future PR. Use ordered commits and hard
internal phase gates inside this PR so the researched migration order remains
reviewable. PR #2186 remains the documentation-only research source; the later
user request for a unique implementation PR supersedes its original same-PR
instruction.

This is a breaking migration. Backward compatibility and semver are not
constraints. Prefer clean ownership and explicit migration errors over shims
that violate dependency direction.

## Authority and source of truth

Follow instructions in this order:

1. the current user request and repository `AGENTS.md` files;
2. current source and manifests at the actual checked-out revision;
3. this implementation prompt;
4. the research package in this directory;
5. ECS PR #6 only as experimental evidence, never as a specification.

Before editing, read the complete research package:

- [`README.md`](README.md);
- [`architecture-decision.md`](architecture-decision.md);
- [`ownership-and-migration.md`](ownership-and-migration.md);
- [`ecs-reference-analysis.md`](ecs-reference-analysis.md);
- [`runtime-import-inventory.md`](runtime-import-inventory.md);
- [`research-ledger.md`](research-ledger.md);
- [`knowledge-graph.json`](knowledge-graph.json).

Read the current implementation before moving it. Study the nearest existing
provider, tool, WASM, facade, derive, and test patterns. Do not implement from
line-number references alone; the links identify evidence at the research
revision and must be refreshed if `main` has moved.

## Exact research baseline

The research package was produced against:

| Item | Revision |
| --- | --- |
| Rig source, intended base, and merge base | `87f3f5b77a3caeffa10d60225c41e386753bf05e` |
| Merged typed-hook/response-retry PR #2182 base | `d6d2dfa089868223c19040956bbcce62f6311173` |
| PR #2182 head | `f5737b34dc889e146c3be1c126f84d2d326ad36a` |
| PR #2182 merge | `87f3f5b77a3caeffa10d60225c41e386753bf05e` |
| ECS reference PR #6 base | `6f3df71c6c14698e38bd0706d9b142ec3d0d9187` |
| ECS reference PR #6 head | `8f1d72dd50bfb723f088c91eeef4202451a08f09` |
| ECS PR #6/current Rig merge base | `1f1ee4d9bb58b24f9e85572c783f93f173e65dc4` |

At research time, the workspace contained 83 packages: 21 library/proc-macro
packages and 62 example packages. `rig-core` exposed 25 top-level public
modules and 25 built-in provider modules. The root facade declared 41 features,
20 of which selected 18 companion crates. The public-reference inventory found
326 external Rust files coupled to current runtime surfaces: 70 examples and
256 test/fixture files.

Re-establish these facts before implementation. Record the current branch,
HEAD, base, merge base, staged/unstaged/untracked files, and remote PR state.
Preserve all existing user changes. The five untracked Mach-O files recorded by
the research package (`rtn`, `rtn2`, `rtnalias`, `rtnalias2`, and `unstable`)
were pre-existing and must not be edited, staged, deleted, or committed unless
the user separately changes their scope.

## Research conclusions that are implementation requirements

### Selected topology

Implement this production dependency graph:

```text
                         rig
                  facade / namespaces
                  /        |        \
          rig-agent     rig-bevy    integrations/providers
                  \        |        /
                   \       |       /
                        rig-core
```

The graph rules are non-negotiable:

1. `rig-core` does not depend on `rig-agent`, `rig-bevy`, `bevy_ecs`, or a
   runtime-specific public type.
2. `rig-agent` and `rig-bevy` are siblings. Each depends on `rig-core`; neither
   depends on the other.
3. Root `rig` may compose and re-export core, either runtime, and optional
   integrations.
4. Provider, vector-store, memory, and companion crates depend on the narrowest
   portable layer and do not acquire a runtime dependency for backend work.
5. Hooks remain entirely classic-runtime concepts. ECS policy remains entirely
   Bevy-runtime state and systems.
6. Shared production code is limited to stable contracts and canonical values.
   Shared behavior lives in test-only conformance scenarios.
7. Do not introduce a public `AgentRuntime` trait or hidden shared agent state
   machine.
8. Do not make Bevy mandatory or feature-gated inside `rig-core`.
9. Do not solve path compatibility by making `rig-core` re-export
   `rig-agent`; that necessarily creates the forbidden reverse dependency.
10. Built-in provider packaging is orthogonal. Provider implementations may
    remain temporarily in `rig-core`, but all provider source must be runtime
    neutral when this migration is complete.

Validate the actual package graph with `cargo metadata` and `cargo tree` under
all relevant feature combinations. A diagram or intended manifest is not proof.

### Rejected architecture shortcuts

Do not reopen these rejected approaches during implementation:

- keeping the classic runtime in `rig-core` and merely adding `rig-bevy` beside
  it leaves providers and stores coupled to classic behavior;
- putting Bevy behind a `rig-core` feature still permits feature unification,
  conditional public APIs, target/MSRV leakage, and private reverse imports;
- extracting only hooks leaves the state machine, tools, prompt facades,
  extraction, memory timing, telemetry, and constructors coupled;
- driving both runtimes with `AgentRun` or another shared production engine
  prevents ECS from owning topology and progression;
- making Bevy mandatory in core imposes its dependencies and release cadence on
  every provider, backend, and low-level completion consumer;
- copying ECS PR #6 wholesale replaces mature classic behavior and mixes
  unrelated provider, cassette, tool, WASM, and MSRV changes.

The selected design combines the three-crate topology with independent runtime
implementations and shared test-only conformance. Crate symmetry is not a reason
to invent a common public runtime trait.

### What the ECS experiment proved—and did not prove

ECS PR #6 changed 674 files, adding 48,572 lines and deleting 52,033. Its
`rig-core` portion changed 103 files (+33,205/-30,699). It replaced the classic
runtime, made Bevy unconditional in core, raised Rust requirements, weakened
WASM portability, propagated raw `Send + Sync`, rewrote tools and macros,
changed 36 provider source files, changed 152 provider test files, and churned
315 cassettes.

Use PR #6 only for these reusable ECS invariants:

- authoritative ECS components and relationships;
- deterministic, explicitly ordered schedules/system sets;
- stable IDs, correlation IDs, generations, and immutable turn snapshots;
- owned asynchronous effect requests, deltas, and completions;
- cancellation plus stale, late, duplicate, superseded, and foreign-result
  rejection;
- policies represented as data interpreted by ordered systems;
- local and hosted drivers running the same schedules;
- explicit domain snapshots with stable IDs and implementation rebinding;
- separate retirement, cancellation, observation, and cleanup transitions;
- deterministic batch commit independent of query or completion order;
- debugging/explanation views derived from ECS facts.

Reject these PR #6 choices:

- mandatory Bevy or public Bevy re-exports in core;
- wholesale deletion/replacement of the classic runtime;
- weakening `WasmCompatSend`, `WasmCompatSync`, or `WasmBoxedFuture` contracts;
- raising core/agent MSRV to satisfy Bevy;
- raw `World` serialization or persistence of runtime handles/tasks/entities;
- wrapping `HookStack` as ECS policy;
- changing provider wire behavior or rerecording cassettes for runtime work;
- copying its public facade, migration document, or exact implementation APIs;
- regressing Copilot or any other WASM-capable provider because the ECS runtime
  needs a different effect boundary.

### Ownership summary

Keep these responsibilities in `rig-core`:

- canonical messages, content blocks, documents/media, tool calls/results,
  identifiers, usage, requests, responses, and provider errors;
- `CompletionModel`, completion request/building, typed raw completion response,
  raw streaming choices/deltas/finals, provider stream accumulation, and narrow
  provider capability facts such as
  `CompletionModel::composes_native_output_with_tools`;
- low-level client/provider/capability traits and model construction;
- embedding, rerank, transcription, audio/image, model-listing, HTTP, SSE, and
  loader contracts that do not orchestrate an agent;
- portable vector-store and memory backend contracts;
- portable tool authoring metadata/call contract plus canonical tool output and
  error values;
- provider-response preservation, provider telemetry conventions, and WASM
  compatibility helpers;
- core-only test models, request validators, and provider mapping fixtures;
- built-in provider implementations only as transitional integration code,
  with no imports or return types from either runtime.

Move these responsibilities intact to `rig-agent` before refactoring them:

- `Agent`, `AgentBuilder`, `AgentRunner`, `AgentRun`, `AgentRunStep`, turn state,
  pending calls, manual stepping, serialization, and `drive_agent`;
- `AgentHook`, `HookStack`, every hook event/action/context, `RunId`,
  `Scratchpad`, `RequestPatch`, invalid-tool actions, and response retry;
- `Prompt`, `Chat`, `TypedPrompt`, `StreamingPrompt`, `StreamingChat`, their
  request/response types, and runtime errors;
- `OutputMode` selection and structured-output recovery;
- `ToolContext`, `ToolSet`, dynamic/erased registry and dispatch, tool server,
  snapshots, execution sequencing, and agent-as-tool behavior;
- memory load-before-run and append-after-commit orchestration;
- extractor, CLI/Discord agent integrations, dynamic context, retrieval-tool
  registration, and classic runtime telemetry;
- all classic-specific unit, serialization, parity, concurrency, and hook
  tests.

Build these responsibilities in `rig-bevy`:

- agent/model/run/operation/capability/grant/store entities and relationships;
- stable identity, generation, tenancy, ownership, and retirement state;
- deterministic schedules, system sets, progress, quiescence, and livelock
  detection;
- model/tool/store effect dispatch and validated ingress;
- ECS-native policies, capability/grant decisions, output recovery, and retry;
- stream subscriptions and provisional versus committed output;
- local and hosted handles, cancellation, terminal observation, and cleanup;
- memory/store adapters, persistence/restoration/rebinding, redaction, and
  debug/explanation views;
- runtime-specific tests for ordering, effects, stale results, concurrency,
  isolation, persistence, and scheduling.

Split these mixed modules rather than assigning them wholesale:

- `completion`: core model/value contracts versus classic prompt facades and
  runtime errors;
- `streaming`: core provider primitives versus classic multi-turn requests and
  events;
- `client`: core model constructors versus runtime extension traits;
- `tool`: portable authoring/output versus classic context/registry/server and
  Bevy capability/effect state;
- `embeddings` and `vector_store`: portable values/index contracts versus
  runtime registration as executable retrieval tools;
- `memory`: portable backend/policy contract versus runtime commit timing;
- `telemetry`: provider conventions versus runtime lifecycle spans/events;
- `prelude`: a portable core prelude plus distinct classic and Bevy preludes;
- `test_utils`: provider/core fixtures versus shared conformance and
  runtime-specific harnesses.

## Design decisions for this implementation

The research deliberately left several choices for implementation review. To
make one-PR execution finite, use the defaults below. Deviate only when a
source-backed prototype proves the default cannot satisfy the stated
invariants; document the evidence and replacement decision in the same PR.

### Tool authoring boundary

Use two explicit authoring layers:

1. `rig_core::tool::Tool` is portable and does not accept classic mutable
   `ToolContext`, Bevy `World`, `Entity`, grants, or runtime IDs. It receives
   owned/borrowed canonical arguments and returns canonical typed output/error
   through WASM-compatible bounds.
2. `rig_agent::tool::ContextualTool` (exact name may follow local naming) owns
   the current `&mut ToolContext` behavior. The classic runtime adapts both
   portable and contextual tools into its registry.

`rig-bevy` adapts only portable tools through owned effect inputs. ECS-native
context comes from explicit capability/grant entities and policy facts, not an
arbitrary mutable type map passed into an async future.

Update `#[rig_tool]` so a function without a `ToolContext` parameter implements
the portable core trait, while a function with an explicitly typed classic
context implements the classic contextual trait. Do not infer context from an
unrelated user type with the same identifier. Emit precise migration diagnostics
and add trybuild coverage for renamed `rig`, `rig-core`, and `rig-agent`
dependencies.

If a portable invocation context is truly required by existing non-classic
consumers, it must be a narrow owned/read-only core contract with no registry,
type map, ECS, or lifecycle state. Do not move the current `ToolContext` into
core under a different name.

### Runtime construction and method collision avoidance

The core client trait retains model construction only. Implement blanket
classic extensions in `rig-agent`:

```rust,ignore
pub trait AgentClientExt: rig_core::client::CompletionClient {
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel>;
    fn extractor<T>(
        &self,
        model: impl Into<String>,
    ) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: schemars::JsonSchema
            + for<'de> serde::Deserialize<'de>
            + serde::Serialize
            + rig_core::wasm_compat::WasmCompatSend
            + rig_core::wasm_compat::WasmCompatSync
            + 'static;
}

pub trait AgentModelExt: rig_core::completion::CompletionModel + Sized {
    fn into_agent_builder(self) -> AgentBuilder<Self>;
}
```

Remove `CompletionClient::agent()`/`extractor()` from core. Remove OpenAI's
inherent `GenericCompletionModel::into_agent_builder()` from provider code and
provide the same spelling through `AgentModelExt`. Verify every provider client
and model receives classic construction without provider-specific runtime code.

Give Bevy an intentionally distinct extension such as
`BevyCompletionClientExt::bevy_agent()`, or require explicit runtime/spec
construction. Importing both runtime preludes must never produce two `agent()`
candidates.

### Facade and feature policy

- Add root `agent` and `bevy` features backed by optional runtime dependencies.
- Keep classic behavior in the root default feature set.
- Keep Bevy opt-in, namespaced under `rig::bevy`, and absent from the default
  prelude.
- `rig::prelude` combines the portable core prelude with the classic prelude.
- `rig::bevy::prelude` exposes Bevy construction and common ECS runtime types.
- Prefer deliberate re-exports over the current undifferentiated core glob.
- Root `rig` may temporarily preserve useful classic paths. `rig-core` may not
  re-export moved classic symbols.
- Core-only users should depend directly on `rig-core`; disabling root defaults
  must not silently select either runtime.

Preserve the intent of all 41 current root features. Runtime-free transport,
provider, vector-store, memory, loader, derive, and WASM features must not
select Bevy or classic runtime dependencies unless their public API truly
constructs that runtime.

### WASM and MSRV policy

Preserve current `rig-core` and classic-runtime WASM behavior and MSRV unless a
separate, evidence-backed compatibility decision is included in this PR.

Treat `rig-bevy` as native-only and experimental initially. Select a
`bevy_ecs` version deliberately. If it requires a newer compiler, set and test
that requirement on `rig-bevy` only and add an appropriate isolated CI lane;
do not raise the workspace, `rig-core`, `rig-agent`, provider, or companion MSRV
solely for Bevy. Do not leak raw native `Send + Sync` bounds into portable APIs;
apply stricter executor bounds in Bevy adapters.

### Provider-final policy

Retain `CompletionResponse<T>` and `StreamingCompletionResponse<R>` typed raw
finals at direct core boundaries. Runtime progression consumes canonical
choice, usage, message ID, finish metadata, and errors. The acceptance contract
for each surface is explicit:

- direct core blocking and streaming calls expose their existing typed raw
  final;
- classic streaming exposes the typed provider-final event;
- classic blocking does not need a new raw-final API to complete the split; an
  opt-in typed callback/collector is permitted, and must be type-tested if
  implemented, but `AgentRun` never serializes it;
- Bevy local blocking handles expose a typed provider final at the effect/run
  boundary;
- Bevy local streaming subscriptions expose a typed provider-final event after
  provisional deltas;
- hosted/erased Bevy paths expose a documented non-persisted diagnostics
  envelope and explicitly do not promise the provider's concrete type.

- Never persist arbitrary provider responses in ECS snapshots or coerce them to
  `serde_json::Value` as canonical state.

### Conformance package and support status

Create an unpublished workspace test-support crate named
`rig-runtime-conformance` (or the closest repository-consistent name). It
depends on `rig-core` only in production terms and is consumed as a dev
dependency by both runtimes. Any harness trait or fixture that must be named by
those crates is `pub` only as part of this unpublished test-support API. Do not
re-export it from `rig-core`, either runtime's production API, root `rig`, or a
published prelude, and do not turn it into a production runtime abstraction.

Complete the functional runtime migration in this PR, including the required
Bevy vertical slices and provider acceptance. Mark `rig-bevy` experimental.
Do not claim supported/default status because the researched readiness gate
requires operational history beyond one PR. Classic remains the default. The
absence of a support declaration is not permission to leave required code,
tests, persistence, or conformance as a stub.

Built-in provider crate decomposition and a Bevy extractor are separate
follow-ups. They are not required for the runtime split if provider source is
runtime-neutral and Bevy's documented native agent surface is complete.

## Classic-runtime semantic preservation

Move the current classic implementation before changing it. Preserve public
behavior, serialized run state, hook semantics, retry accounting, streaming
parity, and diagnostics. Do not opportunistically redesign the classic runtime
while extracting it.

The typed hook and response-retry behavior merged in PR #2182 is part of the
baseline:

- every hook method receives the run-scoped `HookContext`;
- event-specific action types reject unsupported combinations at compile time;
- completion-call patches accumulate in registration order and `Stop`
  short-circuits;
- tool-call and tool-result rewrites chain through later hooks;
- tool-call `Skip` and event `Stop` actions are terminal;
- invalid calls retain `Fail`, `Retry`, `Repair`, `Skip`, and `Stop` behavior;
- observe-only events retain `Continue`/`Stop` behavior;
- nested `HookStack`s preserve merge/chaining order;
- `RequestPatch` stays per-turn and non-sticky, with append, shallow-merge,
  intersection, and last-writer rules unchanged;
- streaming and non-streaming surfaces share the same driver semantics;
- response retry is tool-free, rolls back rejected content, adds corrective
  feedback, reruns request preparation/hooks, consumes the total model-call
  budget, and preserves usage without recording rejected content as accepted
  telemetry;
- retry never creates an empty rejected history turn.

Retain `AgentRun` and `drive_agent` together in `rig-agent`. They are a good
classic sans-I/O boundary, but must not become a shared engine driven by Bevy.

## Bevy runtime invariants

The Bevy runtime is complete only if ECS owns authoritative progression rather
than mirroring classic state.

### State and topology

- Model agents, runs, turns/operations, tool calls, capabilities, grants,
  stores, policies, and terminal state as focused components and relationships.
- Use stable domain IDs separate from Bevy `Entity` values.
- Represent ownership/tenant boundaries explicitly and validate them at every
  capability/effect boundary.
- Avoid central mega-components and opaque serialized classic state.
- Define explicit schedule labels/system sets and required `ApplyDeferred`
  boundaries.
- Make progress, quiescence, terminal observation, livelock, and cleanup
  distinct states.
- Results must not depend on entity insertion order, query iteration order, or
  asynchronous completion order.

### Owned effects

- No `World`, query borrow, component reference, or runtime lock guard may enter
  an async model/tool/store future.
- Systems snapshot owned request inputs and dispatch effects.
- Effect completions carry stable run/operation IDs, generation, correlation,
  and tenant/capability identity.
- Only ingress systems validate and commit completions to authoritative state.
- Duplicate, stale, late, superseded, canceled, and foreign-world completions
  are rejected or recorded diagnostically without mutating newer state.
- Queue/backpressure and concurrency limits are explicit and bounded.
- Completion commit and accounting are idempotent.

### Tools, policy, and concurrency

- Advertised tool definitions come from an immutable per-turn capability
  snapshot.
- A tool invocation resolves the exact snapshotted implementation/revision;
  replacement or retirement cannot swap an in-flight call to another tool.
- Grants and tenant ownership are checked before dispatch.
- Invalid-tool recovery, response retry, output modes, suppression, stop, and
  cancellation are ordered ECS policy/system decisions, not hook callbacks.
- Parallel tool bodies may complete in any order, but canonical transcript and
  events commit in model-call order.
- Define the terminal winner when stop/cancel/error/output finalization race.
- Suppressed tools never execute and ideally never create an executable effect.

### Streaming and terminal behavior

- Stream deltas are provisional observations, not committed transcript state.
- Subscriptions distinguish provisional deltas, accepted final content,
  rollback/rejection, cancellation, provider failure, and terminal completion.
- Blocking and streaming handles drive the same schedules/state; blocking may
  ignore deltas but must reach the same canonical terminal outcome.
- Terminal state is externally observable before cleanup removes supporting
  entities or retained results.
- A provider error after a final-looking delta/final must not expose false
  success.

### Memory, persistence, and debugging

- Load memory before the first model request and append only newly committed
  canonical messages.
- Store effects use the same correlation, generation, cancellation, and
  idempotency discipline as model/tool effects.
- Persist explicit versioned domain records with stable IDs, not a raw `World`.
- Never persist `Entity`, tasks, channels, executor handles, client objects,
  trait-object implementation pointers, raw provider finals, credentials, or
  arbitrary secrets.
- Restoration validates schema version and integrity, reconstructs topology,
  and explicitly rebinds model/tool/store implementations.
- Missing or mismatched implementations produce typed errors, never panics or
  silent substitution.
- Debug/explanation views are derived from authoritative facts and redact
  sensitive inputs by default.
- Cleanup cannot race terminal observers or erase diagnostics before configured
  retention expires.

## Shared behavioral conformance

Build scripted core models, tools, stores, controllable futures/effects,
canonical transcript/event expectations, and scenario inputs. Run the same
observable scenarios against private adapters for both runtimes without
requiring identical internal APIs.

Implement all of these scenarios:

1. **Model-call budgets:** zero rejects the initial call; `N` permits exactly
   `N` total calls including retries and continuations.
2. **Canonical transcript validity:** valid role order, no empty synthetic turn,
   and rejected/provisional content excluded from committed history.
3. **Tool-call/result pairing:** every committed call has exactly one matching
   result; parallel arrival still commits in call order.
4. **Usage and completion accounting:** exactly one call record per billed
   completed model operation; retry/rejection usage rules match classic
   behavior; duplicates do not double count.
5. **Invalid-tool recovery:** `Fail`, `Retry`, `Repair`, `Skip`, and `Stop`, with
   suppressed calls never executed.
6. **Response retry and rollback:** tool-free constraint, corrective feedback,
   fresh preparation/policy, total budget, and accepted-only content.
7. **Stop and cancellation:** terminal behavior prevents later dispatch/commit,
   preserves diagnostics, and distinguishes cancellation from cleanup.
8. **Structured output:** Native/Tool/Prompted/Auto selection, provider
   composition capability, collision-safe synthetic tool, bounded recovery,
   and best-effort exhaustion.
9. **Memory:** load timing, committed-only append, failure mapping, idempotency,
   and no append on failed/stopped runs.
10. **Blocking/streaming parity:** identical committed history, usage, final
    content, errors, and terminal reason.
11. **Provider-final exposure:** typed final available where promised and no
    false success after a later provider error.
12. **Provisional streaming:** early deltas observable but never silently
    promoted after retry, rejection, stop, or cancellation.
13. **Tool suppression:** invalid peer, policy skip, output finalization, stop,
    and cancellation prevent execution.
14. **Concurrency:** bounded execution, deterministic commit/events, and
    defined sibling drain/cancel behavior.
15. **Stale-result handling:** duplicate, stale, superseded, canceled,
    wrong-generation, wrong-tenant, and foreign-world results cannot mutate
    authoritative state.

Keep provider cassette tests at the core/provider mapping layer. Run a small
provider-backed acceptance matrix for both runtimes covering at least:

| Provider surface | Blocking | Streaming | Tools | Structured output | Raw final |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI Responses | yes | yes | yes | native + tool | per surface matrix below |
| Anthropic Messages | yes | yes | yes | tool/prompted | per surface matrix below |
| Gemini Generate Content or Interactions | yes | yes | yes | native + recovery | per surface matrix below |

For each provider row, apply this raw-final acceptance matrix:

| Surface | Required acceptance |
| --- | --- |
| Direct core blocking | concrete typed `CompletionResponse<T>::raw_response` |
| Direct core streaming | concrete typed final from `StreamingCompletionResponse<R>` |
| Classic blocking | canonical final required; optional typed collector tested if implemented |
| Classic streaming | concrete typed provider-final runtime event |
| Bevy local blocking | concrete typed provider final returned/exposed by the local handle |
| Bevy local streaming | concrete typed provider-final subscription event after deltas |
| Bevy hosted/erased | documented non-persisted diagnostics envelope; no concrete-type claim |

Use replay/scripted coverage by default. Live credentials are not required to
declare the PR complete unless repository policy explicitly makes a live test
required. Do not rerecord provider cassettes merely because paths or runtimes
moved.

## One-PR implementation sequence

Do not flatten this work into one unreviewable edit. Use the following ordered
phases as commits or clearly separated commit groups in the implementation PR.
Keep each phase buildable and reviewable where feasible. Run the phase gate
before advancing; if a temporary bridge is unavoidable, keep it same-crate or
in root `rig`, document it, and remove it before the final gate.

### Phase 0 — baseline, decisions, and guardrails

1. Fetch the intended base and inspect the full existing PR diff.
2. Record current workspace packages, features, MSRVs, target gates, public
   modules, provider modules, runtime imports, and cassette hashes.
3. Read current classic runtime and all relevant tests, including PR #2182
   behavior.
4. Add dependency-direction assertions and compile fixtures before moving code.
5. Record the selected tool, Bevy version/MSRV, facade feature, raw-final, and
   conformance-package decisions in the architecture docs.

Gate: baseline tests and CI commands pass; no user files are lost; the intended
target graph and forbidden edges are mechanically testable.

### Phase 1 — conformance fixtures and portable tool boundary

1. Add the unpublished conformance crate and scenario ledger with scripted core
   effects but no runtime implementation.
2. Split portable tool definitions/output/errors/authoring from classic
   context, registry, server, snapshots, dispatch, and concurrency.
3. Update dynamic tools, embedded/retrieved tools, RMCP adapters, agent-as-tool,
   derive expansion, and tests through deliberate adapters.
4. Preserve current classic behavior behind the new boundary.

Gate: portable tool modules import no classic execution type; classic tool and
hook suites remain behavior-identical; derive trybuild and WASM checks pass.

### Phase 2 — invert dependencies and extract `rig-agent`

1. Split low-level completion/streaming/client/memory/telemetry from classic
   facade and orchestration code.
2. Create `crates/rig-agent` and move the complete classic runtime intact.
3. Move `AgentRun` and `drive_agent` together; move hooks, prompt/streaming
   requests, output recovery, extractor, classic tools, integrations, memory
   orchestration, and telemetry.
4. Publish `AgentClientExt` and `AgentModelExt`; remove core/provider runtime
   constructors, including OpenAI's inherent builder method.
5. Move classic tests and add the conformance adapter.
6. In the same commit group, add the minimal root manifest dependencies,
   feature gates, and facade re-exports needed for the moved crate to compile.
   Phase 3 completes and documents the deliberate public facade; do not leave
   the workspace broken between phase commits.

Gate: `rig-agent -> rig-core` only; core/provider source cannot name classic
types; serialized run fixtures and all classic semantics remain unchanged; the
workspace and default root facade compile at the phase boundary.

### Phase 3 — facade, features, macros, companions, and call sites

1. Replace indiscriminate root glob behavior where it obscures ownership.
2. Add classic-default and Bevy-opt-in feature/namespace wiring.
3. Split core/classic/Bevy preludes without extension-method collisions.
4. Update `rig-derive`, all companion crates, all 70 runtime-bearing examples,
   all 256 inventoried external test/fixture files, READMEs, and rustdoc.
5. Keep provider cassette fixtures byte-for-byte unchanged unless a separately
   proven provider contract defect requires a targeted update.

Gate: every generated path and import names its actual owner; the root feature
powerset compiles; core-only graphs select no runtime; examples and docs build.

### Phase 4 — minimal `rig-bevy` boundary and topology

1. Add `crates/rig-bevy` with Bevy isolated to that crate.
2. Implement focused components, relationships, stable IDs, bundles only where
   useful, schedule labels/system sets, progress/quiescence, and local driver.
3. Add runtime/spec/handle construction without pretending to be classic
   `Agent`.
4. Add deterministic ordering, foreign/stale handle, tenancy skeleton,
   deferred-command, quiescence, and livelock tests.

Gate: core/agent dependency trees contain no Bevy; neither runtime depends on
the other; ECS state is authoritative and deterministic under shuffled entity
insertion.

### Phase 5 — model effects, streaming, and terminal commit

1. Adapt `CompletionModel` through owned effect inputs and completion ingress.
2. Implement call budgets, canonical history, usage/call accounting, provider
   errors, stream subscriptions, raw-final side channels, terminal state,
   cancellation, and local/hosted handles.
3. Reject duplicate/late/stale/wrong-generation/wrong-tenant effects.
4. Run shared model, streaming, terminal, cancellation, and stale-result
   scenarios.

Gate: no ECS borrow enters a future; every completion commits at most once;
blocking/streaming parity holds; terminal state is observable before cleanup.

### Phase 6 — tools, grants, policy, recovery, and concurrency

1. Add capability/grant/revision entities and immutable advertised snapshots.
2. Dispatch portable tools through owned effects with exact snapshot identity.
3. Implement ECS-native invalid-call policy, output mode/recovery, response
   retry, suppression, stop/cancel ordering, parallel batches, replacement, and
   retirement.
4. Add tenant/grant isolation, deterministic commit, racing-terminal, and
   in-flight replacement tests.

Gate: all shared tool/retry/output/concurrency scenarios pass; policy is ECS
data/systems rather than callbacks or a hidden registry/state machine.

### Phase 7 — stores, memory, persistence, debugging, and cleanup

1. Add vector/memory/store capability adapters and correlated effects.
2. Implement memory conformance and committed-only append.
3. Add versioned stable domain snapshots, integrity validation, restoration,
   explicit implementation rebinding, and migration errors.
4. Add redacted explanation/debug views, configurable result retention,
   retirement, and safe cleanup.
5. Test crash/restart, missing/mismatched implementations, deterministic
   snapshots, cancellation/late effects, and secret/handle exclusion.

Gate: restored state reproduces canonical domain state after rebinding; no raw
entities, tasks, provider finals, clients, or secrets are persisted.

### Phase 8 — full conformance, acceptance, hardening, and documentation

1. Run every shared scenario against both runtimes.
2. Run runtime-specific hook and ECS topology/effect/persistence suites.
3. Run the provider acceptance matrix using replay/scripted tests and targeted
   live tests only when authorized and necessary.
4. Add classic and Bevy examples, migration guidance, feature documentation,
   crate READMEs, root README updates, and support-status documentation.
5. Audit error paths, panic/unwrap usage, cancellation, backpressure, tenant
   isolation, redaction, telemetry, target bounds, and public docs.
6. Remove every temporary bridge, compatibility alias, TODO, placeholder,
   unimplemented path, and dead migration feature.

Gate: no shared conformance divergence is hidden by weakened assertions. Any
intentional runtime-specific difference is named, tested, and documented.

### Phase 9 — final PR completion gate

1. Inspect the full diff against the current merge base, including staged,
   unstaged, and untracked files.
2. Recompute the package/feature/dependency graph and prove every forbidden
   edge absent.
3. Run the complete verification matrix below.
4. Have a fresh independent reviewer inspect the entire diff for correctness,
   security, regressions, hidden shared orchestration, unsafe edge cases, and
   missing tests.
5. Validate each finding against current code; fix all confirmed P0/P1 issues
   and all in-scope lower-severity issues.
6. Rerun affected checks and repeat independent review until no confirmed
   P0/P1 remains.
7. Push to the implementation PR, wait for every required CI check, and inspect
   all unresolved actionable review threads.

Gate: all required checks are successful, the PR is mergeable, no actionable
review thread is unresolved, and the completion audit is satisfied.

## Verification matrix

Start with targeted crate/phase checks, then run the full matrix. Use exact
repository CI commands where they differ from these examples.

### Formatting, lint, tests, and docs

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --all-features
```

Also reproduce CI's root-test compilation, targeted classic agent nextest
suite, and broad nextest suite when `cargo-nextest` is available.

### Target and feature isolation

```bash
cargo check -p rig-core --no-default-features
cargo check -p rig-core --features wasm --target wasm32-unknown-unknown
cargo check -p rig-agent --all-features
cargo check -p rig-agent --target wasm32-unknown-unknown
cargo check -p rig-bevy --all-features
cargo check -p rig --no-default-features
cargo check -p rig --no-default-features --features agent
cargo check -p rig --no-default-features --features bevy
cargo check -p rig --no-default-features --features agent,bevy
cargo check -p rig --all-features
```

If exact feature names differ, document the final names and test the equivalent
core-only, classic-only, Bevy-only, combined, default, and all-features graph.
Run appropriate core/agent MSRV lanes and an isolated Bevy MSRV lane. A
native-only Bevy policy must be explicit and target-gated; it must not cause the
portable lanes to fail.

### Dependency direction

Use `cargo metadata`, `cargo tree`, and a small automated assertion to prove:

```text
rig-core !-> rig-agent
rig-core !-> rig-bevy
rig-core !-> bevy_ecs
rig-agent !-> rig-bevy
rig-bevy !-> rig-agent
provider/store/memory integration production dependencies !-> either runtime
rig-agent -> rig-core
rig-bevy -> rig-core
rig -> selected runtimes + core + selected integrations
```

Check normal, build, dev, target-specific, default, no-default, and all-feature
edges separately. Test-only conformance edges must not enter production graphs.

### Runtime behavior

- all 15 shared conformance scenarios against both runtimes;
- complete classic hook, response-retry, run serialization/stepping,
  blocking/streaming parity, memory, tool, and concurrency suites;
- Bevy topology/order/quiescence, effect ingress, stale/late/duplicate,
  cancellation, tenant/grant, output recovery, concurrency, persistence,
  restoration, redaction, cleanup, and local/hosted parity suites;
- provider-final and provider-error ordering tests;
- fuzz/property/stress tests where ordering, generation, snapshots, or
  concurrent terminal actions have combinatorial state.

### Providers, cassettes, macros, examples, and docs

At minimum replay the OpenAI, Anthropic, and Gemini cassette suites with one
test thread, plus every provider suite touched by a real mapping change. Run the
full root all-features test target when feasible. Review cassette hashes and
diffs; runtime-only movement should leave fixtures unchanged.

Run every `rig-derive` trybuild case, including dependency rename/path
resolution and portable versus contextual tool signatures. Compile all
workspace examples and doc examples under their declared features. Ensure every
runtime-bearing example explicitly selects classic or Bevy APIs.

Validate all local links, source anchors, JSON/YAML artifacts, Mermaid diagrams,
crate READMEs, root README, facade docs, feature docs, and rustdoc. Document
intentional experimental limitations without using them to excuse missing
implementation.

## Safety and quality requirements

- Preserve unrelated user changes and never use destructive Git commands.
- Use Rig's existing builders, traits, error types, WASM helpers, provider
  response helpers, and telemetry conventions.
- Add `///` docs to every new public item and `//!` docs to every public module.
- Use explicit `thiserror` enums for new fallible APIs; no new `String` errors.
- Avoid `.unwrap()`/`.expect()` except for genuinely impossible, obvious
  invariants. Never panic on malformed persisted/effect/provider input.
- Use `WasmCompatSend`, `WasmCompatSync`, and `WasmBoxedFuture` in portable
  contracts; apply native executor bounds only at Bevy adapters.
- Do not add TODOs, stubs, placeholder implementations, speculative public
  abstractions, or tests that merely assert construction.
- Do not weaken existing tests, conformance expectations, lint settings, or
  target support to make the migration pass.
- Do not alter provider requests/responses, cassettes, credentials, or account
  identifiers unless a separately proven provider defect is in scope.
- Redact credentials, tenant data, tool arguments/results, provider raw
  responses, and memory contents from debug/telemetry/persistence by default.
- Bound queues, concurrency, retries, recovery loops, and schedule progress.
- Define cancellation ownership and drop behavior; do not rely on cleanup as
  cancellation.
- Treat tool grants, tenant ownership, snapshot integrity, rebinding, and effect
  ingress as security boundaries.
- Comments should explain why; avoid unrelated cleanup and refactors.

## Completion definition

Do not declare the migration complete merely because the workspace compiles.
All of the following must be true:

1. Every public `rig-core` module and principal symbol has its documented final
   owner or an explicitly approved out-of-scope provider-packaging defer.
2. `rig-core` contains no classic or Bevy runtime implementation, constructor,
   error progression, prelude export, or dependency.
3. `rig-agent` contains the full classic runtime with PR #2182 hook/retry
   behavior and existing WASM semantics preserved.
4. `rig-bevy` contains complete ECS-native model, streaming, tool/policy,
   memory/store, persistence, debugging, cancellation, and cleanup vertical
   slices—no classic engine wrapper and no placeholder surface.
5. The root facade and all 41 feature intents compose core/classic/Bevy and
   integrations without feature leakage or method collisions.
6. Derive macros, companion crates, all inventoried runtime-reference sites,
   examples, tests, READMEs, and rustdoc use the correct owner paths.
7. Provider source is runtime-neutral and provider cassettes show no unrelated
   churn.
8. Both runtimes pass every shared conformance scenario; runtime-specific
   behavior is separately covered.
9. Dependency and target checks prove the graph acyclic and core free of Bevy.
10. Persistence/effect/concurrency/tenant/redaction security audits are clean.
11. Formatting, lint, tests, docs, examples, target checks, and required CI all
    pass.
12. A final independent full-diff review finds no unresolved confirmed P0/P1
    issue or hidden shared orchestration engine.
13. The unique implementation PR contains the entire implementation, is
    mergeable, has no pending or failing required checks, and has no unresolved
    actionable review thread.
14. `rig-bevy` is honestly documented as experimental; classic remains default
    until separate operational evidence justifies another status.

## Final handoff

The final PR report must include:

- exact base, merge base, head, branch, and commit sequence;
- final crate/feature/dependency graph and proof of prohibited-edge absence;
- complete ownership/move summary and public API examples;
- tool/macro, MSRV/WASM, raw-final, persistence, and support-status decisions;
- conformance scenario and provider acceptance results for both runtimes;
- verification commands with outcomes and any unrelated baseline limitations;
- cassette/security/redaction review results;
- independent-review findings fixed or rejected with rationale;
- CI and unresolved-thread state;
- any remaining risk that truly requires a maintainer decision.

Do not open another PR, declare a partial migration complete, or stop after
creating crate skeletons. Continue within PR #2186 until the completion
definition is satisfied or report the exact external blocker and remaining
work.
