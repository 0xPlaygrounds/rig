# Architecture decision: narrow core with independent classic and Bevy runtimes

- Status: proposed
- Date: 2026-07-18
- Source revision: `87f3f5b77a3caeffa10d60225c41e386753bf05e`
- Decision owners: Rig maintainers
- Scope: crate and public-boundary design; no production migration in this PR

## Problem

`rig-core` currently owns both provider-neutral contracts and one specific agent
runtime. Adding a Bevy-native runtime inside that crate would make Bevy, its
MSRV, and its concurrency model transitive requirements of provider clients,
vector stores, memory integrations, and low-level model users. Replacing the
classic runtime would discard a mature typed hook system and its newly merged
response-retry semantics. Keeping two runtimes in the same crate would leave
portable contracts coupled to classic agent constructors and facade traits.

The architecture must allow two genuinely different orchestration models:

- the current classic runtime, with a serializable sans-I/O `AgentRun`, a shared
  blocking/streaming driver, ordered typed hooks, mutable tool context, and
  request builders;
- a Bevy ECS runtime in which world state, schedules, systems, capability
  relationships, policies, and effect correlation are authoritative.

Both must speak the same provider contracts and canonical values and must meet
the same observable behavioral specifications. They need not share internal
state shapes, extension mechanisms, or progression code.

## Evidence

### Current boundary defects

- `CompletionClient` imports `AgentBuilder` and `ExtractorBuilder` and exposes
  them directly through `agent()` and `extractor()`
  ([`client/completion.rs:1-60`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/client/completion.rs#L1)).
- OpenAI's inherent `GenericCompletionModel::into_agent_builder()` returns the
  same classic `AgentBuilder` directly from provider code
  ([`openai/completion/mod.rs:1898-1901`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/providers/openai/completion/mod.rs#L1898)).
- `PromptError` is defined beside provider `CompletionError`, but contains
  memory, max-turn, cancellation, and unknown-tool states
  ([`completion/request.rs:140-184`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L140)).
- `Prompt`, `Chat`, and `TypedPrompt` promise tool execution and transcript
  mutation; these are runtime behaviors
  ([`completion/request.rs:357-446`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/completion/request.rs#L357)).
- `StreamingPrompt` and `StreamingChat` return the classic runtime's
  `StreamingPromptRequest`, while the same file also defines provider-facing raw
  streaming choices
  ([`streaming.rs:67-261`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L67),
  [`streaming.rs:565-626`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/streaming.rs#L565)).
- `Extractor` contains an `Agent` and builds classic output-tool behavior
  ([`extractor.rs:37-79`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L37),
  [`extractor.rs:199-242`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/extractor.rs#L199)).
- The classic `AgentRun` owns budgets, rollback state, history, usage, completion
  calls, output recovery, and tool-call resolution
  ([`agent/run/mod.rs:277-317`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/run/mod.rs#L277)).
  These are not provider contracts.
- `HookStack` composition is intentionally event-specific; it is not a generic
  observer container
  ([`agent/hook.rs:1251-1377`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L1251)).
- `ToolSet` and `ToolServerHandle` own registration, snapshots, mutation, and
  dispatch beyond the portable `Tool` authoring contract
  ([`tool/mod.rs:511`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/mod.rs#L511),
  [`tool/server.rs:126-230`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/tool/server.rs#L126)).
- All 18 companion libraries depend on `rig-core`; the facade additionally
  re-exports them behind features ([root facade](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/src/lib.rs#L30),
  [root features](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/Cargo.toml#L254)).

### Classic behavior that must survive extraction

PR #2182 merged at the exact source revision. It added a model-turn lifecycle
action with `Continue`, retry, and stop behavior; retries are restricted to
tool-free turns and consume the total model-call budget
([`agent/hook.rs:427-478`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L427),
[`agent/hook.rs:940-949`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/hook.rs#L940)).
`drive_agent` is shared by blocking and streaming entry points, preserving
parity while delegating genuinely medium-specific work
([`prompt_request/streaming.rs:390-449`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/prompt_request/streaming.rs#L390),
[`prompt_request/streaming.rs:471-489`](https://github.com/0xPlaygrounds/rig/blob/87f3f5b77a3caeffa10d60225c41e386753bf05e/crates/rig-core/src/agent/prompt_request/streaming.rs#L471)).

The extraction must therefore move the current runtime without semantic change
before Bevy work begins. PR #6 cannot be the parity oracle because its base
diverged before PR #2182.

### ECS experiment evidence

Reference PR #6 changes 674 files, with 48,572 additions and 52,033 deletions.
Within `rig-core` it changes 103 files (33,205 additions, 30,699 deletions),
deletes ten classic agent implementation files, adds twelve `runtime/*` source
files plus ECS support files, and makes `bevy_ecs` mandatory in `rig-core`.

The experiment demonstrates that ECS-native orchestration benefits from:

- components and relationships as authoritative topology;
- deterministic schedule stages and explicit ordering;
- owned asynchronous effects that re-enter through correlated completions;
- immutable per-turn capability snapshots;
- policies represented as ECS data/systems rather than a callback stack;
- stable IDs, explicit snapshots, retirement, and stale-result rejection;
- local and hosted handles over the same schedules;
- stream chunks as values while call/run lifecycle remains ECS state.

It also demonstrates the cost of a wrong crate boundary: the experiment deletes
`wasm_compat`, replaces portable bounds with raw `Send + Sync` throughout
provider code, raises workspace Rust to 1.95 for Bevy, disables a Copilot WASM
flow, rewrites tools and derive macros, and touches 315 cassettes. Those changes
are not inherent in provider behavior.

## Decision

Adopt this production dependency topology:

```text
                         rig
                  facade / namespaces
                  /        |        \
          rig-agent     rig-bevy    integrations/providers
                  \        |        /
                   \       |       /
                        rig-core
```

Apply the following dependency rules:

1. `rig-core` may not depend on `rig-agent`, `rig-bevy`, `bevy_ecs`, or any
   runtime-specific extension type.
2. `rig-agent` and `rig-bevy` are siblings that depend on `rig-core` and never
   on each other.
3. `rig` may depend on and re-export both runtimes. The classic runtime remains
   default; Bevy is feature-gated and namespaced.
4. Companion providers, stores, and memory crates depend on the narrowest
   contract layer. They must not depend on a runtime merely to implement a
   provider or backend.
5. `AgentHook`, `HookStack`, hook events/actions, `RequestPatch`, and the entire
   current agent state machine move to `rig-agent`.
6. Bevy components, bundles, relationships, schedules, systems, policies,
   effects, handles, snapshots, and debug/explanation state live in `rig-bevy`.
7. Shared production code is limited to canonical values and stable contracts.
   Shared runtime behavior is specified by test fixtures and conformance
   scenarios, not by a production orchestration engine.
8. No public `AgentRuntime` trait is introduced during extraction. A future
   trait requires a demonstrated consumer that needs a stable cross-runtime
   invocation contract.
9. Provider raw response types remain available at direct completion boundaries
   and through runtime-specific typed side channels. Canonical run progression
   does not serialize or persist arbitrary provider-native responses.
10. Built-in provider extraction is a separate follow-up. It is desirable for
    an ideal provider-neutral `rig-core`, but combining it with runtime movement
    would make review and rollback unsafe.

## Public-surface consequences

The core client trait retains only `completion_model()`. Classic conveniences
move to a `rig_agent::client::AgentClientExt` trait, re-exported by the default
`rig::prelude`, so the ordinary API remains:

```rust,ignore
use rig::prelude::*;

let agent = client.agent("model").preamble("...").build();
```

The Bevy extension is distinct and namespaced:

```rust,ignore
use rig::bevy::prelude::*;

let runtime = Runtime::builder().build()?;
let agent = runtime.spawn_agent(client.bevy_agent("model").build())?;
let run = agent.prompt("hello").await?;
```

These names illustrate ownership and collision avoidance; they are not frozen
API. What is frozen by this decision is that importing both runtime preludes
must not produce two `agent()` methods on the same client type.

The root default prelude exports core contracts plus non-colliding classic
runtime ergonomics. Portable names such as `Tool` retain their core identity;
contextual tools are explicit under `rig::agent::tool`. It does not glob-export
Bevy components or schedules. `rig::bevy::prelude`
exports the Bevy-specific common path. Direct `rig-bevy` consumption remains
supported for advanced ECS users.

## Architectural option evaluation

Scores are 1 (poor) through 5 (strong). “Drift” scores resistance to unintended
behavioral divergence; “maintenance” scores lower ongoing cost. Scores are
comparative recommendations, not measured facts.

| Option | Isolation | Compile/deps | WASM | MSRV | ECS-native | Classic hooks | Provider portability | Facade ergonomics | Testing cost | Drift | Migration | Maintenance | Total / 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Mandatory Bevy inside `rig-core` | 1 | 1 | 1 | 1 | 5 | 2 | 1 | 3 | 2 | 3 | 1 | 2 | 23 |
| 2. Classic runtime stays in `rig-core`; add `rig-bevy` | 2 | 2 | 3 | 3 | 5 | 5 | 2 | 3 | 3 | 2 | 4 | 3 | 37 |
| 3. `rig-core` + `rig-agent` + `rig-bevy` | 5 | 5 | 4 | 4 | 5 | 5 | 5 | 4 | 3 | 3 | 3 | 3 | 49 |
| 4. Shared state machine with hook/ECS adapters | 4 | 4 | 4 | 4 | 2 | 3 | 5 | 4 | 4 | 5 | 2 | 3 | 44 |
| 5. Narrow contracts + independent runtimes + shared conformance | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 3 | 3 | 3 | 52 |

Option 3 describes the selected crate topology. Option 5 describes how those
crates must be implemented and governed. They are combined, not competing final
choices.

### Option 1: mandatory Bevy in `rig-core`

Rejected. It gives ECS full control but imposes Bevy and its MSRV on every
consumer. PR #6 provides direct evidence: `bevy_ecs = 0.19` becomes an
unconditional `rig-core` dependency, the workspace Rust version rises to 1.95,
and portable/WASM bounds are rewritten. Provider integrations should not pay for
an agent runtime they do not use.

### Option 2: classic runtime remains in `rig-core`

Rejected as the target, although it could look migration-friendly. It leaves
`CompletionClient`, prompt traits, errors, extraction, tools, memory, and
telemetry coupled to one runtime. A provider or vector-store crate still pulls
the classic runtime even when used exclusively with Bevy. It also makes
`rig-core` a misleading name and preserves the original cycle pressure.

### Option 3: three crates

Accepted topology. It isolates dependencies and lets each runtime own its
extension model. On its own it does not prevent a hidden shared engine or
behavioral drift, which is why option 5's constraints are part of the decision.

### Option 4: shared state machine with adapters

Rejected. The current `AgentRun` is an effective classic sans-I/O machine and
should move intact, but it is shaped around steps (`CallModel`, `CallTools`,
`Done`), hook callbacks, a single progression owner, and serialized pending
state. Driving it from Bevy would make ECS components mirrors of opaque state or
reduce systems to I/O adapters. ECS could not independently model call entities,
parallel effects, grants, policies, generations, retirement, or schedule stages
without duplicating authority. A shared engine improves drift scores by
preventing difference, but it prevents the deliberate runtime difference that
motivates `rig-bevy`.

### Option 5: narrow contracts and conformance

Accepted implementation discipline. Each runtime owns progression. Shared
scripted models, canonical event expectations, and scenario definitions expose
behavioral drift without forcing identical internals. A test-only harness may
use a private adapter trait because it does not constrain public API or runtime
architecture.

## Why a feature-only split is insufficient

Putting Bevy behind a `rig-core` feature avoids some default compile cost but
does not repair ownership:

- feature unification can enable Bevy transitively anywhere in a dependency
  graph;
- `rig-core` public types still need conditional APIs or common representations;
- provider and store crates still depend on a crate whose runtime features
  affect MSRV, target support, and semver;
- docs and preludes expose mutually exclusive or colliding runtime concepts;
- runtime-specific code can continue importing private core internals because
  the crate boundary does not enforce direction.

A crate boundary gives Cargo and Rust a mechanically checkable dependency rule.

## Why extracting only hooks is insufficient

Hooks are one visible coupling but not the progression owner. The classic
runtime also includes:

- `Agent`, builder, runner, and serializable run state;
- prompt/chat/typed/streaming facade traits and request builders;
- max-turn, invalid-call, output-retry, response-retry, and cancellation state;
- tool registries, live servers, immutable turn snapshots, contexts, dispatch,
  and execution concurrency;
- memory load/append timing and transcript commit rules;
- dynamic context and vector-store-to-tool conveniences;
- extractor retry/output-tool behavior;
- classic run and tool telemetry;
- `CompletionClient::agent()`/`extractor()` convenience construction;
- OpenAI's inherent `GenericCompletionModel::into_agent_builder()`.

Leaving these in `rig-core` while moving hooks would create a reverse dependency
or force hooks to remain generic callbacks in core. Neither produces a narrow
contract layer.

## Why no common public runtime abstraction

The only clearly stable shared consumer contracts today are model requests,
canonical messages/content, usage, tool authoring/output, storage backend calls,
and observable outcomes. “Run an agent” is not yet narrow: classic returns
request builders and exposes manual stepping; Bevy should expose handles,
subscriptions, policies, entity state, and hosted/local scheduling.

Defining `AgentRuntime`, `AgentHandle`, or a universal lifecycle event enum now
would either erase useful runtime features or encode one runtime's assumptions
as universal. The facade can offer namespaced conveniences without a common
trait. Introduce one only when a real downstream consumer can state the minimal
operations it needs and both runtimes can implement them without adapters that
hide semantics.

## Costs of two runtimes

This decision intentionally accepts ongoing cost:

- two progression implementations and two runtime-specific test suites;
- separate docs, examples, debugging, telemetry, cancellation, and persistence
  surfaces;
- a conformance council/process to classify intentional differences;
- provider acceptance runs across a small runtime/provider matrix;
- duplicated fixes when a defect is behavioral rather than architectural;
- more complex facade documentation and support triage;
- a risk that one runtime lags features or provider support.

Mitigations are explicit ownership, a shared scenario ledger, recorded runtime
capability/support levels, required conformance before release, and refusal to
make one runtime silently emulate the other.

## Conformance and governance

Create a test-only `rig-runtime-conformance` package (or equivalent workspace
test support) that depends only on `rig-core`. It supplies scripted completion
models/effects, canonical transcripts, expected events/outcomes, and scenario
functions. Each runtime implements a private dev-only harness adapter and runs
the same scenarios in its own tests.

Govern scenario changes as observable contract changes:

1. A behavior-changing PR updates the scenario expectation first.
2. Both runtimes must pass, or the divergence must be explicitly classified as
   runtime-specific and documented.
3. Provider cassette tests remain low-level and run once against core provider
   mappings; runtime acceptance uses a small representative provider matrix.
4. Release notes state each runtime's support level and known intentional
   divergences.
5. `rig-bevy` cannot become supported/default based only on API completeness;
   it must pass the readiness gate in the migration plan.

## Consequences

### Positive

- Bevy dependency, MSRV, and target policy are isolated.
- Classic hooks and response-retry semantics remain first-class.
- ECS extensions can use native components/systems/schedules without callback
  emulation.
- Providers, stores, and low-level completion users depend on portable contracts.
- The root facade can expose both runtimes without extension-trait collisions.
- The dependency graph is acyclic and mechanically enforceable.

### Negative

- Initial migration is larger than adding a feature.
- Tool authoring and derive paths require a deliberate split.
- Some currently convenient `rig-core` APIs become `rig-agent` APIs.
- Behavioral parity relies on test quality and governance rather than shared
  production code.
- Provider decomposition remains unfinished technical debt after the runtime
  split.

## Unresolved decisions

Compatibility ownership is resolved: moved classic symbols may be re-exported
temporarily by root `rig`, but not by `rig-core`. A `rig-core` compatibility
shim would require the prohibited `rig-core -> rig-agent` dependency.

1. Exact portable versus contextual `Tool` APIs and `rig_tool` macro syntax.
2. Whether `rig-bevy` initially supports WASM/single-threaded execution or is
   explicitly native-only.
3. The typed raw-provider-final subscription and retention model for hosted
   Bevy runs.
4. Whether `rig-agent` keeps the package feature name `agent` in the root facade
   or is always enabled by default without a public feature toggle.
5. The eventual number and naming of built-in provider crates.
6. Whether test conformance support is a publishable crate or an unpublished
   workspace-only package.

These questions do not invalidate the dependency direction. They must be
resolved in the early migration PRs before public API movement or ECS runtime
implementation.
