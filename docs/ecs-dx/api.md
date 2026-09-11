# ECS API disposition and review evidence

Parent: `ad8e9b69f857992f7078d66555bce5c4db751df5`.
The tables record source-reviewed dispositions by API family; the linked member
and export inventories enumerate the individual public entries. The before/after
review guide is [review.md](review.md). Completed verification and downstream exact-commit results belong in the PR
description; progress.md retains the chronological checks and corrections.

| Surface | Disposition | Consumer and evidence |
| --- | --- | --- |
| `commands::{Agent, Prompt, RigCommands}` | Added as the primary construction/submission path | All agent examples, `tests/commands.rs`; temporary owned values initialize the authoritative graph. |
| `commands::{install, OperationError, CommandFailures}` | Added explicit installation and error handling | Immediate examples use one Result; asset example drains deferred failures. Reserved IDs are observable before application. |
| `Handlers::{descriptor, keys, descriptors}` | Retained applied-binding inspection, documented deferred visibility | commands test proves pending IDs work for construction while inspection changes only after deferred application. |
| `Handlers::register_in` | Added immediate registration | Examples no longer need nested SystemState callbacks/results. |
| `Handlers::register_typed` | Infer the concrete handler's associated Family | `bus_scene::a_typed_key_dispatches_across_ticks` needs no type annotation or turbofish. |
| `Handlers::register_erased_typed` | Retain a distinct checked dynamic path | Metadata mismatch and single-descriptor-snapshot regressions in `bus_scene`. Custom payload compatibility remains a handler contract. |
| Agent `Streamed` | Renamed to `RunStreaming` | Distinguishes run configuration from bus stream data; persisted `streamed` keys remain unchanged. |
| `lifecycle::{grant_tool, revoke_tool, retry_turn, patch_turn}` | Added checked direct and deferred operations | Captured tool advertisement/revocation, pending registration, retrieval-only links, retry feedback requests, phase/conflict/removed-target tests; direct graph oracles retained. |
| `lifecycle::{cancel, fork}` | Checked public operations | Real run tests cover resting cuts, issued workers, independent branches, observer removal and cancellation endings. |
| `inspect::{RunView, RunInfo, EffectInfo}` | Derived borrowed read access | Examples scope outcomes to their run/cohort; effect inspection walks only the selected subtree and exposes approval snapshots. |
| `stream::StreamText` | Per-effect append cursor | Interleaved Unicode streams and explicit reset tests; does not claim detection of arbitrary same-length replacement. |
| `approval` | Small native request/decision workflow | Twelve dispatch/identity tests, three input bridge tests, executed CLI paths; no input executor in the library. |
| `RigSet`, `BusSet`, `RigSchedule`, `Progress`, direct components/relationships | Retained advanced ECS extension points | `bus_api`, steering/provider/persistence tests and examples. Scheduler access and flush boundaries remain explicit. |
| `SubjectWalk` | Retained deliberate advanced access | `bus_api::a_mutating_gate_can_name_its_subject_without_conflicting_query_access` mutates PendingEffect while reading identity; Subjects would conflict with that mutable query. |
| `Subjects`, `Witnessing`, `AdapterOperation` | Retained observation boundary | Existing host/adapter witness consumers; no context added to CompletionRequest. |
| `Settings`, system query aliases and runtime systems | Internalized | No external function consumers; public scheduling sets remain. Independent component queries preserve partial run overrides and explicit clears. |
| Dispatch/collection query tuples | Replaced by named private query data | DispatchCandidate, CollectingStream, LandedEffect; stream budgets, order and fairness tests retained. |
| Assembly/awaiting query tuples | Replaced by named private query data | AssemblingRun, FreshTurn, AwaitingRun; request/output and graph suites verify behavior. |
| HandlerTable, Served, WorldServe, registry observers | Internalized within bus | Public Serve/world-handler consumer tests remain; registration owns erasure and thread affinity. |
| CollectionBudget, CollectedOutcome, Intake, WorldOutcomeCounter | Internalized within bus | Runtime bookkeeping, not host configuration; ServingPolicy remains public. |
| Record observer helpers, DeliveryBatch, ReplacedBy, WorldObserver | Internalized within bus | Recording and observed consumer outcomes remain public. |
| Refused, SeenOutcome, DispatchWitness, fingerprint, preflight aliases | Internalized within bus | Host policies use SubjectWalk/Subjects and Witnessing instead of driver markers. |
| Observed/ObservedState | Internalized within bus | Full original-answer cancellation/replay scenario moved unchanged into `bus/delivery/visibility_tests.rs`; private cleanup assertion moved to `bus/record/tests.rs`. |
| Despawning | Private to witness module | Public cancellation trace test remains; typed private-resource cleanup assertion replaces external Debug-string inspection. |
| `ExecutionStatus`, `execution_status`, `BusSet::Begin` | Added supported host inspection/ordering | Actual rigcoder producer coordinates its fixture groups using worker readiness and before-delivery input. Public tests distinguish live workers from uncollected answers; task storage stays private, delivery ordering and per-tick budgets remain unchanged. |
| Executions, Streaming::spawn, drop_execution | Internalized within bus | Native worker tests retained; WASM worker replacement/removal/despawn/shutdown test moved to `bus/effect/wasm_tests.rs` with explicit executed `wasm-rig-ecs-lib` verification and CI lane. |
| Fresh/Materialised, Folded, BatchHeld | Retained advanced graph markers | Policy/steering, reflection and batch/persistence consumers inspect real scheduling boundaries and restored state. |
| spawn_run, spawn_utterance, next_order | Retained explicitly advanced graph primitives | Independent graph/provider/WASM oracles use them; primary applications use Agent/Prompt. Raw spawn_run documents observer interruption, while checked initialization removes its rejected destination. `next_order_in` remains crate-only for assets. |

The removed blanket restricted-visibility architecture rule is replaced by actual
public consumer coverage; other bus independence/ownership guards remain.
The member/export inventories below cover fields, traits and feature modules;
[examples.json](examples.json) also records rustdoc, README, root-facade discovery
and representative provider/consumer test dispositions. Final verification must
check these claims against executed results; an inventory is not test evidence.


## Feature and member inventory

[public-members.json](public-members.json) enumerates 224 local items from the
all-features rustdoc build, including public fields/variants, inherent methods
and required trait methods. [public-exports.json](public-exports.json) records
42 module/re-export declarations, including external contracts and feature-gated
modules. These mechanical lists delimit coverage; they do not independently
prove source correctness. Inherited/derived trait methods retain their defining
Bevy/Rust contracts. The family tables cover these members; final reviewer and verification outcomes
are recorded separately.

| Family and its exposed fields/methods | Disposition | Consumer/rationale |
| --- | --- | --- |
| agent components and relationship targets | Retain native graph data and intentional direct access | run_graph, policy fold, scene/reflect, direct corpus interpreter; generated relationship collections remain controlled by Bevy. Ordinary construction no longer requires manual tuples. |
| WorldScene, RunScene, Loaded, SceneEntity, SceneKind, Target | Improve primary save/load naming; retain persisted fields and remapped-index results | Combined WorldScene now has save/load methods matching its constituents. Removed free save_world/load_world and private RunScene::take; original scene/memory/batch/reflection assertions retained. |
| SceneExtensions::register_component | Retain explicit typed, versioned application-state registration | run_scene_extensions; generic type is required because there is no component value to infer it from. Payload references are not remapped; host resources/observers stay host-owned. |
| Replay, EffectLogResource, program identity helpers | Improve immediate registration; retain recorded identity and validation APIs | Replay::register_in removes nested host result/callback in run_scene and three-golden bus_scene tests; register remains the SystemParam path. stamp_header is explicitly legacy corpus identity; check_replayable validates exact run scope. |
| reflect exports, ReflectedEntity/ReflectedScene and remote wrappers | Retain feature-gated inspector integration | reflect_roundtrip and reflect_scene; exposed reflected payloads are for inspection, not a claim that tasks or approval authorization persist. |
| assets types/loaders/handles, Applied, AssetsSet, AssetsPlugin | Retain feature-gated Bevy asset integration | prompt_from_assets explicitly installs AssetPlugin/AssetsPlugin and orders submission; Applied guards one-time application. Prompt asset and command Prompt use module-qualified imports. |
| bus effect components, WorldEffect, Answer/Asked/Typed, WorldHandler | Retain supported handler and world-system contracts | bus_api, bus_world, bus_scene, WASM tests; runtime-owned workers remain private. Handler erasure stays behind registration. |
| bus Scene/SceneEffect and validate_parent_indices | Retain advanced bus-only persistence and composed-graph validation | bus_scene and combined agent scene; persisted public fields represent schema, not live task handles. Validation/reconstruction order unchanged. |
| HoldOwners, HoldTransition, acquire_hold/release_hold | Retain low-level ownership mutation and observation | batch/witness tests and approval workflow. Boolean results report whether an ownership transition occurred; approval supplies the richer liveness/proposal/hold-loss diagnostics its consumer needs. No universal per-branch enum added. |
| policy RequestGraph/fold_request, output/text derivations | Retain deliberate pure-policy extension surface | One fold remains authoritative; unit goldens and direct graph tests pin wire behavior. RequestGraph documents owned vectors/documents accurately. |
| bus collection constants, replay collector/idle diagnosis | Internalize | Only runtime/collector tests use these; host policy remains ServingPolicy and public schedule sets. |
| systems::witness::install/ending_of | Internalize | Installation belongs to install_agent; ending conversion is shared only by its observers. Emitter identity remains public for witness consumers. |
| shared Serve/Reply/adapters and optional AdapterOperation | Retain existing serving boundary | Provider construction and provider cassette helper use CompletionAdapter/ToolAdapter. No new adapter context on CompletionRequest, no timing/semantic-trace framework. |

Construction failure cleanup and staged relationship visibility are documented in
commands and exercised by immediate/deferred observer-removal tests. These are
intentional corrections to checked construction, not a promise to roll back
arbitrary application observer actions.
