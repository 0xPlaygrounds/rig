# ECS reference PR #6 analysis

> **Historical evidence.** This analysis is fixed to the revisions named below
> and informed the migration implemented in
> `940483b4fb30aa77d83dab52a1599652cb9e0c2a`. Baseline source links are
> commit-pinned where applicable.

Reference: `gold-silver-copper/rig-ecs` PR #6, “Implement ECS-native runtime
migration.” The PR is an open architectural experiment and will not be merged.
This analysis uses its exact base
`6f3df71c6c14698e38bd0706d9b142ec3d0d9187` and head
`8f1d72dd50bfb723f088c91eeef4202451a08f09`.

The experiment's base is not Rig's researched source revision. Its merge base
with current Rig is `1f1ee4d9bb58b24f9e85572c783f93f173e65dc4`;
PR #6 base is 1 commit ahead on its branch and current Rig is 12 commits ahead.
The experiment therefore predates current PR #2182 response-retry semantics.

## Quantitative inventory

The complete diff contains 674 changed files, 48,572 additions, and 52,033
deletions. Status counts are 20 added, 13 deleted, and 641 modified files.

This responsibility classification is exhaustive: every changed path belongs
to exactly one row. Categories describe architectural responsibility, not the
directory name alone.

| Responsibility category | Files | Additions | Deletions | Classification rule/meaning |
| --- | ---: | ---: | ---: | --- |
| Provider cassette fixtures | 315 | 8,688 | 9,745 | `tests/cassettes/**`; rerecorded provider exchanges |
| Classic agent replacement/facade | 11 | 422 | 24,523 | `rig-core/src/agent/**`; ten implementation files deleted and `agent/mod.rs` rewritten as an ECS facade |
| Companion crates | 21 | 63 | 81 | non-derive `crates/rig-*` propagation |
| Derive macro and tests | 20 | 124 | 435 | `crates/rig-derive/**`; tool context/signature migration |
| ECS runtime implementation | 14 | 31,435 | 0 | twelve `runtime/*.rs` files, `time.rs`, and ECS benchmark |
| Examples | 45 | 2,363 | 3,015 | root examples adapted to replacement runtime/tool API |
| Other `rig-core` propagation | 28 | 546 | 562 | manifest/README, core contracts/utilities, ECS examples and WASM test not in other rows |
| Other tests | 8 | 264 | 39 | non-provider, non-cassette tests |
| Portable/mixed contracts | 7 | 179 | 277 | completion client/request, streaming, extractor, memory, and vector modules |
| Provider test source | 152 | 2,359 | 7,043 | `tests/providers/**`; agent/hook/tool call-site migration and deletions |
| Provider implementations | 36 | 92 | 205 | `rig-core/src/providers/**`; mostly bound/import propagation |
| Tool boundary | 7 | 531 | 5,132 | `rig-core/src/tool/**`; context/server/registry removal and context-free authoring rewrite |
| Workspace/docs/CI | 10 | 1,506 | 976 | root manifests/toolchain, migration doc, README, workflows, and policy docs |
| **Total** | **674** | **48,572** | **52,033** | complete diff |

Within `crates/rig-core` specifically, 103 files change with 33,205 additions
and 30,699 deletions. Those 103 files are exhaustively covered by the six
`rig-core` categories above: classic replacement (11), ECS implementation (14),
other propagation (28), portable/mixed contracts (7), providers (36), and tools
(7).

Line counts demonstrate blast radius, not correctness. The strongest evidence
comes from which contracts and dependency directions change.

## What the PR replaces

The head deletes these classic implementation files:

- `agent/builder.rs`
- `agent/completion.rs`
- `agent/hook.rs`
- `agent/prompt_request/mod.rs`
- `agent/prompt_request/streaming.rs`
- `agent/run/mod.rs`
- `agent/run/output_mode.rs`
- `agent/run/streamed.rs`
- `agent/runner.rs`
- `agent/tool.rs`

It rewrites `agent/mod.rs` into compatibility handles/facades around an
authoritative ECS `World`. It also deletes `tool/extensions.rs`,
`tool/server.rs`, and `wasm_compat.rs`.

The new runtime is implemented in:

- `runtime/adapters.rs`
- `runtime/config.rs`
- `runtime/control.rs`
- `runtime/debug.rs`
- `runtime/effects.rs`
- `runtime/identity.rs`
- `runtime/mod.rs`
- `runtime/policy.rs`
- `runtime/snapshot.rs`
- `runtime/state.rs`
- `runtime/tests.rs`
- `runtime/topology.rs`

The public crate adds `pub mod runtime` and publicly re-exports the exact
`bevy_ecs` version. `bevy_ecs` is unconditional in `rig-core` with `std` and
`multi_threaded` features. This is a runtime replacement, not a sibling-runtime
experiment.

## Responsibility-level classification

### Inherently ECS-specific changes

These concepts belong in a future `rig-bevy` and can be ported in design, often
with code adaptation:

- ECS components and relationships for agents, models, runs, operations,
  capabilities, grants, policies, and stores;
- an authoritative `World` plus explicitly ordered schedules/system sets;
- change detection and immutable per-turn capability snapshots;
- owned asynchronous effect requests/completions/deltas with stable identity,
  generation checks, cancellation, and late/stale-result handling;
- policy facts and decisions as components interpreted by registered systems;
- local and hosted runtime handles that drive the same schedules;
- stream deltas as values/subscriptions while call/run lifecycle remains ECS
  state;
- explicit domain snapshots using stable IDs rather than raw `World`
  serialization;
- restoration that rebinds runtime-only model/tool/store implementations;
- debugging/explanation views derived from ECS facts;
- deterministic batch commit independent of query or completion order;
- retirement and cleanup distinct from cancellation.

The experiment's `EcsEffect`, `EffectRequest`, `EffectCompletion`,
`EffectDelta`, `RigSchedule`, `RigSet`, policy components, topology
relationships, stable IDs, `RuntimeHandle`, and snapshot records are direct
evidence that these responsibilities form a coherent runtime crate rather than
portable provider contracts.

### Genuine current crate-boundary defects exposed by the experiment

The experiment had to edit or recreate these APIs because they currently live
in `rig-core` but construct/assume the classic runtime:

- `CompletionClient::agent()` and `extractor()`;
- `Prompt`, `Chat`, `TypedPrompt`, high-level streaming traits, and their error
  progression;
- agent builder/facade/request/response types;
- extractor implementation;
- `ToolSet`, server, dispatch, and context;
- vector-store automatic tool/dynamic-context conveniences;
- memory orchestration;
- classic telemetry span reuse;
- CLI and Discord adapters;
- prelude exports and root glob facade;
- derive macro output that assumes classic `ToolContext`.

These edits are evidence for extracting `rig-agent`, not evidence that the ECS
versions belong in `rig-core`.

### Portable contract changes that should not be dictated by ECS

PR #6 changes portable APIs to satisfy its chosen native multi-threaded executor:

- replaces `WasmCompatSend`/`WasmCompatSync` with raw `Send + Sync` throughout
  completion, providers, HTTP, tools, memory, embeddings, vector stores, and
  tests;
- deletes `WasmBoxedFuture`/target-selective boxed stream types with the whole
  `wasm_compat` module;
- removes mutable `ToolContext` from `Tool::call`, changes dynamic callbacks to
  owned `'static` boxed futures, and deletes tool servers/registries;
- raises the workspace `rust-version` and toolchain from 1.94 to 1.95 for
  Bevy 0.19;
- makes `jsonschema` and `sha2` core dependencies for runtime concerns;
- changes portable/public values to support ECS serialization/persistence;
- changes extractor errors and implementation around an ECS-local agent facade.

Some of these may be independently good redesigns, but none is proven universal
by the ECS runtime. The target split applies stricter native `Send + Sync`
bounds at `rig-bevy` adapters while preserving core's target-appropriate
contracts.

### Tool API: required versus independent redesign

Required by an ECS-native runtime:

- never pass `&mut World` or ECS borrows into async tool futures;
- snapshot owned tool identity, arguments, grants, and policy before dispatch;
- represent tool calls/results/generations as ECS state/effects;
- commit parallel results in model-call order;
- keep raw outcome, model-visible presentation, policy decision, and telemetry
  facts distinct;
- make capability replacement/retirement safe for in-flight snapshots.

Not required by ECS itself:

- removing every form of typed per-call tool context from portable authoring;
- globally requiring raw `Send + Sync` instead of applying it at the Bevy
  adapter;
- deleting the classic registry/server for all consumers;
- changing the `rig_tool` macro for the classic runtime;
- removing WASM-compatible typed tools;
- changing ordinary provider tests and examples to the ECS API.

Recommendation: core exposes the smallest authoring/canonical tool contract.
`rig-agent` owns its mutable type-map context, registry, server, and dispatch.
`rig-bevy` adapts portable tools through owned effects and offers ECS-native
capability installation. Whether core `Tool` is context-free or accompanied by
a narrow portable invocation context remains a maintainer decision; PR #6's
choice should not be copied without a downstream tool migration study.

### Bevy and MSRV leakage

PR #6 adds:

```toml
bevy_ecs = { version = "0.19", default-features = false,
             features = ["std", "multi_threaded"] }
rust-version = "1.95"
```

to the workspace/core path and publicly re-exports `bevy_ecs`. That couples:

- every provider/vector/memory integration's compile graph to Bevy;
- core MSRV to Bevy's release cadence;
- public component derives and entity types to one Bevy version;
- low-level users to a multi-threaded runtime they did not select.

This is precisely what `rig-bevy -> rig-core` prevents.

### WASM leakage

The experiment removes target-selective marker and boxed-future behavior and
uses raw `Send + Sync`. It also replaces the Copilot WASM access-token exchange
with an error stating that exchange requires an external ECS effect. The latter
is an observable provider regression caused by runtime architecture, not a
Copilot protocol change.

The future design may choose an initially native-only `rig-bevy`. That choice
must be localized to `rig-bevy`; `rig-core` and `rig-agent` keep their existing
WASM contracts unless a separate decision changes them.

### Persistence and effect boundaries

The experiment correctly treats persistence and async execution as runtime
concerns:

- futures receive owned input and never borrow the world;
- completions carry stable correlation/generation state;
- explicit domain records use stable IDs;
- runtime-only clients, tasks, schedules, channels, and raw entities are
  reconstructed rather than persisted;
- cancellation and cleanup are separate transitions.

These concepts should move to `rig-bevy` intact as invariants. Exact record
schemas, public handles, queue types, and installer APIs should be redesigned in
reviewable vertical slices.

## Provider and cassette analysis

### Provider source

Thirty-six provider source files change, totaling only 92 additions and 205
deletions. Inspection shows the dominant transformation is:

```text
WasmCompatSend + WasmCompatSync  ->  Send + Sync
```

Other changes are small lint/style propagation, streaming type-erasure changes,
and the Copilot WASM regression described above. There is no architectural need
for an ECS runtime to change provider wire conversion. Providers should continue
to map core canonical requests/responses and know nothing about schedules,
entities, hooks, or policies.

### Provider tests

The 152 provider test source files lose classic hook/agent/tool-context tests or
are rewritten to the replacement facade. These changes are caused by replacing
the runtime, not provider behavior. In the target architecture:

- provider conversion/cassette tests remain core/provider tests;
- classic hook and agent behavior moves to `rig-agent` tests;
- shared runtime behavior is exercised through conformance;
- a small provider acceptance subset runs for both runtimes.

### Cassettes

All 315 cassette changes total 8,688 additions and 9,745 deletions. Of those,
254 have equal addition/deletion counts and 187 change at most two lines in each
direction. Representative completion-smoke fixtures have identical requests
but different nondeterministic model response text and usage, proving they were
rerecorded without a corresponding request-contract change.

Large streaming cassette diffs appear alongside global streaming test/API
rewrites. They are not evidence that providers require ECS. Rerecording 315
fixtures obscures the architecture diff, creates review/security risk, and
prevents isolating actual request changes. A future port must leave low-level
cassettes unchanged unless a separately justified provider mapping change
requires targeted recording.

## Facade/adaptor code created by current ownership

PR #6 keeps `agent` and the classic prelude names by building facades around ECS
handles. This is evidence that existing convenience APIs are valuable, but the
facades exist inside `rig-core` only because the original APIs also live there.

With sibling crates:

- classic `Agent`/builder/prompt APIs move directly to `rig-agent`;
- Bevy handles/specs live directly in `rig-bevy`;
- `rig` re-exports/namespaces either surface;
- client extension traits construct the selected runtime without provider
  changes.

The future `rig-bevy` should not spend thousands of lines pretending to be the
classic runtime before its native surface is available. Thin conveniences are
appropriate only after the authoritative ECS API is coherent.

## Concepts to port, redesign, reject, or defer

| Disposition | Concepts |
| --- | --- |
| Port as invariants | authoritative ECS topology; typed components/relationships; deterministic schedules; owned effect boundary; stable IDs/generations; immutable turn snapshots; stale/late/duplicate rejection; explicit policy ordering; value stream deltas; local/hosted shared schedules; explicit snapshots/rebinding; retirement/cancellation/cleanup separation; debugging from facts |
| Redesign in `rig-bevy` | public handles/builders; installer/plugin boundary; queue/backpressure; typed raw provider-final side channel; persistence schemas; exact policy component API; tool/store/model erasure; tenant/secrets boundary; output-mode state; response-retry policy |
| Keep in `rig-agent` | current `AgentRun`; `drive_agent`; `AgentHook`/`HookStack`; `RequestPatch`; classic prompt requests; `ToolContext`/registry/server; classic extractor; classic agent telemetry |
| Keep in `rig-core` | canonical messages/content/request/response/usage; completion/embedding/vector/memory contracts; raw streaming primitives; tool output/errors/portable authoring; provider response helpers; WASM compatibility |
| Reject | mandatory Bevy in core; public Bevy re-export from core; wholesale classic replacement; raw Send/Sync leakage; core MSRV forced by Bevy; shared HookStack/ECS callback adapter; raw World serialization; provider/cassette churn; Copilot WASM regression |
| Defer | built-in provider crate decomposition; universal runtime trait; making Bevy default; full WASM Bevy support; exact contextual tool migration; broad provider acceptance beyond the initial matrix |

## Behavioral gaps relative to current Rig

Because PR #6 diverged before PR #2182, its head cannot establish current
response-retry parity. A future port must explicitly cover:

- model-turn rejection only after the provider call is accounted;
- retry only for tool-free turns;
- rejected turn rollback plus corrective feedback;
- fresh request preparation and policy/hook-equivalent evaluation on retry;
- total model-call budget consumption;
- no empty rejected history turn;
- accepted-only content telemetry while retaining correct usage accounting;
- blocking/streaming parity.

Other gaps that require direct conformance proof include current HookStack merge
rules, manual stepping/serialization diagnostics, provider-final ordering,
tool execution suppression, and the mature concurrent terminal-selection tests.

## Conclusion

PR #6 validates that a real ECS-native runtime is meaningfully different from
the classic runtime and that it deserves its own crate. It also demonstrates
what happens when that runtime is installed at the wrong layer: portable bounds,
providers, tools, macros, examples, tests, MSRV, WASM, and hundreds of cassettes
move together.

Use the experiment as a source of ECS invariants and failure cases. Do not use
its crate placement, replacement strategy, portable API changes, facade names,
or migration document as requirements.
