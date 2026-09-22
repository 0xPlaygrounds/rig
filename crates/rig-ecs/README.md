# rig-ecs

Rig's effect bus and agent runtime in a Bevy `World`. Effects and handlers are
entities; systems dispatch work and publish outcomes; relationships carry causal
and conversation structure. `App::new().add_plugins(rig_ecs::RigPlugin::default())`
is a host; `app.update()` is one pass of `RigSchedule`, and the default runner
updates when a task raises `bus::Wake`. Handler tasks and application systems
retain their own execution costs; bounded collection does not make arbitrary
user work nonblocking.

The bus uses `bevy_ecs` and `bevy_tasks`. Agent components, request assembly,
steering and cassette-owned replay build on that bus. See [CONTRACT.md](CONTRACT.md) for detailed
behavior and [the provider comparison guide](../../tests/ecs_parity/README.md)
for the scope of cross-runtime regression tests.

Concrete logs and replay adapters live in `rig-cassette` with its `ecs` feature;
this runtime never depends on that crate. The adapter does not require
`rig-agent` or native HTTP. Attach any `rig_core::serve::Recorder` with
`bus::Recording::install`, or retain an `EffectLogRecorder` through
`rig_cassette::ecs::EffectLogResource::install`.
For replay, add `rig_cassette::ecs::ReplayPlugin` **after** `RigPlugin` or
`BusPlugin`, then register with `rig_cassette::ecs::Replay::register(&mut World, &EffectLog)`.
The plugin installs the recorded-delivery collector and idle-refusal diagnosis;
the runtime retains generic scheduling, delivery and observation mechanisms.

## The run as a graph

Each utterance owns ordered `ChildOf`/`Children` content entities, each carrying
one reflected `agent::content::parts::ContentPart` enum. Query `&ContentPart`
(or `&mut ContentPart`) and match variants such as `ContentPart::Text(text)`;
the former separate payload components are gone. Media metadata structs are
variant fields, not components. `ContentPart::ToolResult` owns ordered text,
image or JSON child entities; `ToolResultStatus` remains a separate component.
`read_message` reconstructs role-checked DTOs; `write_message` validates sources
before replacing the graph. Binary payloads remain shared through `BinaryAssets`.

Variant changes are not component additions/removals. For lifecycle-observed
changes, insert the replacement `ContentPart` and match variants in
`On<Insert, ContentPart>` / `On<Discard, ContentPart>` observers; these do not
observe in-place edits. Previously disjoint mutable payload queries now access
the same component: use one matching query, or a `ParamSet`.

Checkpoint format 2 stores these enum components; format 1 checkpoints are
refused, with no legacy component adapter.

`rig_ecs::checkpoint::save_world` takes the world as reflected data: every
entity with a registered reflected component, components by type path, an
`Entity` in a component as that entity's checkpoint index, parents before
children in `Children` order. A host's own components travel with the entities
they sit on once the host registers the type with `#[reflect(Component)]` in
the world's `AppTypeRegistry`; an unregistered type path is refused on load.
In-flight state (`Serving`, `Streaming`, `Handler`, cache views) is not
reflected and never saved; relationship targets are rebuilt by their sources'
hooks. Resources, system-local state and live tasks remain host-owned. Bind
handlers before loading (a checkpoint handler bound to a key the world serves
merges into the world's, which wins); install application insertion observers
afterward to avoid reacting to partially restored state.

`rig_cassette::ecs::identity::stamp_run` captures supported effective
configuration, including run-over-agent overrides. That module's
`check_replayable` checks the requested run's exact
`Scope`; it does not search for another matching policy hash. Applications
must declare a nonempty `agent::PolicyVersion` for their custom systems,
ordering and otherwise-unhashed configuration. A missing declaration is
reported as unverified. This declaration is not an automatic code fingerprint
or a check of ambient credentials and external state. A rig-agent golden's
builder header is that runtime's own and is not an effective compatibility check.

`check_replayable` can run in a fresh world bound to the log's replayers.
Recorded model identity and capabilities stay authoritative, including native
output composition with tools. Reapply the same middleware before checking:
a replayer supplies the inner handler's recorded exchanges, and does not
execute a descriptor's named layers by itself. Changed semantic models,
capabilities, effective settings and application versions still change the
identity. Replayer registration includes every scoped required row, so an
advertised but unused tool remains available for scene reconstruction; an
unexpected call to it fails with divergence.

If a selected model is removed before dispatch or between turns, assembly
terminates the run with `Failed(Failure::Provider(report))`. A missing
relationship or descriptor reports `HandlerUnavailable`; a non-completion
binding reports its key and wrong family. An outstanding tool can finish
before the next assembly detects the missing model. This differs from
cancelling an in-flight operation.

The request the model sees is derived, never authored: a run entity, utterances `ChildOf` it in sibling (`Children`) order, documents as their own entities attached to a turn by link entities, tools as the handler entities the bus already has (granted by link entities), the model as a relationship, every setting a component — and one function, `policy::fold_request`, that `fold_turn` calls at `RigSet::Assemble` over what `gather_turn` walked, writing the wire `CompletionRequest` into the turn's `PendingEffect`. `CONTRACT.md` names the walk field by field with the golden that pins each; the world interpreter (`crates/rig-cassette/tests/corpus/world.rs`) exercises request assembly through the maintained corpus, including tools, memory and steering.

| Entity | Components |
|---|---|
| Agent | `Owner`, `Preamble`, `Temperature`, `MaxTokens`, `AdditionalParams`, `ToolChoiceSpec`, `Output { mode, schema }`, `OutputToolConfig`, `MaxTurns`, `DefaultMaxTurns`, `InvalidCalls`; `UsesModel` → the model's handler entity; `Grant` link entities → tool handler entities; `Context` link entities → documents |
| Document | `DocumentId`, `DocumentText`, `DocumentProps`; attached to a turn by an `Attachment` link |
| Utterance | `Utterance`, `Role`, `MessageId` for assistants; `ChildOf` the run, with ordered content children |
| Content part | `ContentPart`; `ChildOf` an utterance or a `ContentPart::ToolResult` entity; optionally `ToolResultStatus` on a tool result |
| Run | `Run`, `RunOf` → agent, `RunSeq`, `StreamRequested`, `Cursor`, a `RunPhase` (`LoadingMemory`, `Assembling`, `AwaitingModel`, `ResolvingTools`) or an ending (`Settled`, `Failed(Failure)`), `RunResult`, `Usage`, `OutputRetries`, `OutputToolName`, the run's own overrides of the agent's settings, the bus's `Scope`, a `Name` |
| Turn | `Turn`, `ChildOf` the run; `Advert` links → the tools it advertised; `Attachment` links → its documents; `Outputs` (per tick for a stream); `Reprompt`; `Batch` while its tool calls are out; `systems::{Fresh, Folded, Materialised}` |
| Effect | the bus module's, `ChildOf` the turn: the completion, then one per call to a granted tool (`ToolCallSlot` says which call; the bus's `ToolInputs` carries the run's `ToolContextSpec`) — the batch is the turn's children, `ToolPolicy { concurrency }` on the run or the agent says how many fly at once |
| Invalid call | `InvalidCall` + `Resolution`, `ChildOf` the turn |
| the checkpoint | `rig_ecs::checkpoint::{save_world, load_world}`: the whole world, graph and effects alike |

`OutputToolConfig` optionally sets the output tool's reserved name, description,
and preamble augmentation. A run's component overrides the agent's. With a
schema, an explicit name commits tool output; a conflicting granted tool fails
before dispatch. Once minted, the name remains fixed for the run. Checkpoints
and replay identity retain this configuration.

The agent's sets, around the bus's:

| set | true before | written during |
|---|---|---|
| `RigSet::Advance` | a run in `Assembling` has no fresh turn | a turn with its adverts and attachments, or `Failed(MaxTurns)` |
| `RigSet::Select` | a run may lack a model of its own | the agent's `UsesModel`, copied — a routing system before it gives the run another |
| `RigSet::Assemble` | a fresh turn's graph is complete | `gather_turn` leaves the graph on the turn as `AssemblyInputs`; `fold_turn` folds it and spawns the effect; the run is `AwaitingModel` |
| `RigSet::Patch` | the folded effect is a `PendingEffect` | the second steering slot: a user system rewrites the folded request |
| `RigSet::Release` | a turn's tool batch is out | `release_batch` releases its `rig-ecs/batch` owner up to the concurrency, in call order. Policies use `bus::acquire_hold` and `bus::release_hold` with distinct stable emitter names; `Held` remains until every owner releases. A pre-existing bare `Held` is retained as an independent unknown owner. |
| *`BusSet::Gate` … `BusSet::Judge`* | | |
| `RigSet::Fold` | the effect may have streamed or landed | `Outputs` on the turn |
| `RigSet::Judge` | the turn's outputs are complete | a user system may rewrite them, or a tool child's `EffectOutcome` |
| `RigSet::Materialise` | a complete turn is unread, or its batch has landed | `land_batch`: one user utterance of the results in call order (CONTRACT §8), or a failure; then the chain `record_usage`, `judge_invalid_calls`, `read_turn`, `materialise_assistant`, `materialise_batch`, `materialise_reprompt`, `materialise_answer`: the assistant utterance, the answer, a reprompt, an invalid call, the tool batch, or a failure (CONTRACT §4) |
| `RigSet::Checkpoint` | committed graph writes are visible | the host's slot to inspect or save before the next advance |
| `RigSet::Settle` | a run settled or failed | `append_memory` (a remembering run's append); `diagnostics::measure` when a `DiagnosticsStore` exists; observers on `Settled` / `Failed` |

`systems::diagnostics` publishes `bevy_diagnostic` paths — `EFFECTS_IN_FLIGHT`, `EFFECTS_PENDING`, `RUNS_LIVE` — one measurement per pass, when the app has a `DiagnosticsStore` (`RigPlugin` adds `DiagnosticsPlugin` and `TimePlugin` if absent).

`tests/tool_batch.rs` pins the batch: two calls are two children dispatched in call order and one utterance of results; `ToolPolicy` sets how many fly at once; a `Judge` system's replacement reaches history while the record keeps the answer; a `Gate` denial is a skipped result and no record; a despawned child fails the run `Cancelled`; a system's `Resolution::{Repair, Retry}` renames or retries an invalid call (`Skip` and `Ignore` likewise). The corpus's nesting cases run through a key the world serves (`Handlers::register_open`), the `lookup` tool answered by a system that spawns its child `ChildOf` the call.

A run leaves its graph — turns, utterances, adverts, attachments, the settled effects and a stream's fold — in the world until the host takes it out: `RunCommands::despawn_run` (on `World`, or queued on `Commands`) despawns an ended run whole and refuses one still running (`tests/run_lifetime.rs`). A host that runs for long despawns the runs it is done reading; the world keeps nothing of a run by itself. The bus takes at least one effect per tick whatever `command_capacity` says.

The first steering slot is any system before `Assemble`: it edits the graph. `tests/run_graph.rs` pins the wins: an utterance despawned leaves the next request; one document entity feeds two runs; a grant link advertises a tool and its removal un-advertises it; a model swapped on the run changes the next key; a `Patch` system's rewrite reaches the handler and the record; a system before `Assemble` rewrites an utterance. `tests/run_scene.rs` pins that the graph is the state: a run checkpointed mid-turn resumes in a fresh world to the same second request.

## Steering with systems

Application systems change components at explicit schedule boundaries. See
[CONTRACT §9](CONTRACT.md#9-steering-every-hook-is-a-system) for the detailed
mapping to agent hook actions and the corresponding regression tests.

| Intent | Boundary and operation |
| --- | --- |
| Start or stop a run | Observe `On<Add, Run>`; insert `Cancelled` on an unfinished run to stop it. Issued effects may continue draining. |
| Select a model | Set the run's `UsesModel` after `Advance`, before `Select`. |
| Shape a request | Merge `RequestPatch` on the fresh turn before `Assemble`, or edit the folded effect in `Patch`. |
| Patch, deny or hold dispatch | Change pending work in `BusSet::Gate`; use distinct owner names with `acquire_hold`/`release_hold`. |
| Replace an answer | Change `EffectOutcome` in `BusSet::Judge` or the completed turn's `Outputs` in `RigSet::Judge`; the record retains the handler's answer. |
| Retry or resolve an invalid call | Write `Retry` or `Resolution` before `Materialise`, within the supported turn/stream boundaries. |
| Observe completion | Observe `Settled`/`Failed`; await memory acknowledgements separately when the application requires them. |

Order systems that compose patches explicitly and apply deferred writes between
them. A policy declaration names the application's composition; it does not
fingerprint system implementation. Memory and retrieval are graph relationships:
`Remembers`/`Conversation` select persistence, and `Retrieves`/`Retrieval` attach
retrieved documents or advertise retrieved tools.

## Handler replies and host polling

`Serve::serve` returns `Reply::Outcome(Result<Outcome, ErrorReport>)` or an
owned `Reply::Stream(StreamEvents)`. `Dispatch` carries the effect ID, requested
delivery mode and scopes. Adapters retain their domain traits. `Reply::written`
provides a writer whose future and bounded private receiver are polled together;
`deferred()` provides an external resolver and the future a handler awaits.

The initial task prepares the reply and folds unary requests on the executor.
For streaming requests, one owned worker polls the stream into a bounded private
queue. Native parsing, writer work and streamed verdicts run on pool threads;
browser `!Send` work uses Bevy’s web executor and cannot preempt synchronous work
on the browser thread. Keep ticking to collect ready delivery and settle effects.

Collect drains at most 64 items per effect per pass, sharing
`STREAM_WORK_PER_TICK` (4,096) queue checks across the pass. It rotates through
dispatch order when that allowance runs out, resuming after the last served
`Seq` next pass. These limits bound streaming delivery work, not CPU time,
payload bytes, or the cost of user systems. Every pass gets a fresh allowance.

`stream_capacity` supplies the queue's shared slots, clamped to at least one;
the single sender has one additional reserved slot. The worker awaits each send
before polling again. The writer's own bridge is separate. The task lives in
the non-send `Executions` table under the effect's entity, beside its `Serving`
marker or `Streaming { events, .. }` component. Leaving `InFlight` — removing
it, despawning, dropping the world — cancels the task, including a worker
parked on a full queue. An active native
poll can finish before cancellation drops its future; closed recordings reject
its late observations. Handler replacement leaves already-owned work intact.
Streaming serial slots last through EOF and layer work after `Final`.

Recording keeps the original handler answer and events through layer verdicts.
A recorded answer can survive cancellation while a verdict waits; recording an
original item does not establish consumer delivery. Both task and stream results
still publish tool output before reaching `EffectOutcome` and shared settlement.

## Vocabulary

| Concept | API |
|---|---|
| a dispatch | `commands.spawn(PendingEffect { key, kind })`; `PendingEffect::{new, typed, custom}` |
| dispatch order | `Seq`, stamped on add from `SeqCounter` (global, reserved) |
| the effect's id | `Issued` after `Dispatch`; `Reserved` before it, for a checkpoint's or a log's id |
| taken, in flight | `InFlight { key }` plus `Serving` (initial task) or `Streaming { events, fold, delivered }`, the task a row in the non-send `Executions` (`Tasks` is the `SystemParam` over it); `ServedBy(Entity)` → the handler entity (its `Serves` the inverse) |
| a handler that is a system was asked | `Asked<E>`; the system answers with `Answer<E>` — or, for a key bound open (`Handlers::register_open`, any family), the effect entity itself, answered by submitting `WorldOutcome` |
| the answer | `EffectOutcome(Result<Outcome, ErrorReport>)`; a stream's per-tick fold in `Streamed { events, errors, text, outcome }`, with every error and its item position retained independently of recording |
| the record closed | `Landed { entity, id }`, an entity event `settle` triggers on the effect; it bubbles up `ChildOf` (effect → turn → run → agent), `original_event_target()` the effect |
| held by a decision | `Held` |
| a program's scope | `Scope(String)` on an ancestor; read into the record |
| a tool call's context (beside the effect) | `ToolInputs(ToolContext)` on the effect entity, attached to the handler's `Dispatch` context; what the tool published lands as `ToolOutputs(ToolContext)` when the outcome does (`Publishing` holds the slot in flight) |
| a handler | an entity with `Bound { key, descriptor }` (immutable: every change is an insert) and a `Name`; the erased handler in the `NonSend` `HandlerTable`, marked by `Handler` on the same entity; `HandlerIndex` (key → entity), kept exact by `Bound`'s hooks |
| the registry | `Handlers` (a `SystemParam`): `register`, `register_erased`, `register_typed`, `register_world`, `register_open`, `deregister`, `descriptor`, `keys`, `descriptors`; `Handlers::with(world, ..)` outside a system |
| host model assembly | provider-specific constructors or optional rig-core registry configuration → `CompletionModel` → `CompletionAdapter` → `Handlers::register_erased`; no provider recipe or credential resolver in ECS |
| restoration requirements | `Checkpoint::requirements()` exposes saved descriptors; `Checkpoint::validate(&World)` validates saved data before host construction |
| a typed view | `Typed<F>(Key<F>)`, wherever a system wants it |
| the driver | `dispatch` in `BusSet::Dispatch`; `collect_tasks`, `collect_streams`, `settle` in `BusSet::Collect` |
| the record | `Recording` (any `rig_core::serve::Recorder`); `rig_cassette::ecs::EffectLogResource` (an `EffectLogRecorder` installed as both); for every task-served handler, `Dispatch` installs a recording observer (`WorldObserver`, its slots in `Observed`) so a layer's `discard` and `patch` reach the record and the record keeps the innermost handler's answer |
| a checkpoint | `rig_ecs::checkpoint::{save_world, load_world}` |
| replay | `rig_cassette::ecs::Replay::{register, load}`, by id; requires its `ReplayPlugin` |
| the policy | `Policy(ServingPolicy)`: intake per tick and serial keys; `stream_capacity` bounds driver delivery queues |

## The schedule

`bus::BusPlugin` adds `RigSchedule` with four ordered sets after `Update`
(`MainScheduleOrder::insert_after`), then the generic `RigEnd` schedule.
It sets `woken_runner(idle)` as the app's runner: the schedule runs
**once per `app.update()`**, and the runner updates when `Wake` is raised or
every `idle` at most. Tasks raise `Wake` as they finish or deliver; a host
system that needs another pass raises `wake.signal()`.
Cassette's `ReplayPlugin` installs idle replay diagnosis in `RigEnd`, running
when the pass raised nothing (`Wake::generation` unchanged).
The base bus uses `bevy_ecs`, `bevy_tasks` and a private `futures` delivery queue.
`BusPlugin::install(&mut World)` installs the runtime in a bare world driven by
`world.run_schedule(RigSchedule)`. User systems belong in `RigSchedule`,
ordered against its sets, never beside the runner.

| set | true before | written during |
|---|---|---|
| `Gate` | pending effects are as spawned | a user system patches a `PendingEffect`, denies one (`EffectOutcome(Err(..))`), or holds one (`Held`) |
| `Dispatch` | every un-held, un-answered `PendingEffect` is a candidate | the plugin takes them in `Seq` order up to the pass's intake: `Issued`, `InFlight`, `ServedBy`, `Serving`/`Streaming`/`Asked`; a record opens |
| `Collect` | handlers may have finished or streamed | the plugin writes `Streamed`, `EffectOutcome`; the record closes (`settle`); `InFlight` goes |
| `Judge` | this pass's outcomes have landed and are recorded | a user system may rewrite an `EffectOutcome` before anything after `Judge` reads it |

Decisions are program, never record: the record is what the handler answered, taken in `Collect`, between the two slots. A `Gate` denial never had `InFlight`, so it is no record; a `Judge` rewrite is `Changed`, not `Added`, so it is not re-recorded. Despawning an effect cancels it (its task drops, the record says `Cancelled`) and Bevy despawns its `ChildOf` descendants with it.

## Serial serving and re-entrancy

Under `ServingPolicy::serial_per_handler`, `Dispatch` takes a key only when nothing is `InFlight` on it — a query over the handler's `Serves`. An effect whose ancestor (up `ChildOf`) is in flight on its own key could only wait for itself: it is refused before any dispatch with a `Request` report and no record.

## Handlers that are systems

`Handlers::register_world::<E>(key)` binds a `WorldEffect` (a `CustomEffect` whose payload and answer are `Send + Sync`). A dispatch to the key lands as `Asked<E>` on the effect entity; a user system with any `World` access inserts `Answer<E>`; the plugin queues it for publication as `EffectOutcome` in `Collect`. `Handlers::register_open(key, family)` binds a key of any family to the world itself: the dispatch is taken and left on its entity, `InFlight`, for a system to answer by submitting `WorldOutcome::new(outcome)` — a tool a system serves, nesting what it needs as effects `ChildOf` the call (`bus_world::a_system_serves_an_open_tool_key_and_nests_a_completion_under_it`). The collector publishes answers in submission order; submissions after `Collect` become visible in the next pass. No task is required. Unary only: a system answers once. A handler that must reach the world is one of these; a handler served as a task cannot.

## Regression tests and supported boundaries

The tests under [tests](tests) exercise dispatch, cancellation, bounded intake,
streaming, registry replacement, world-served handlers, checkpoints, replay and WASM.
`bus_scale` covers large pending sets; `bus_world` covers Gate/Judge and nesting;
`bus_scene` covers save/load; `run_identity` covers scoped replay checks.
The producer-owned corpus in `rig-cassette` separately exercises the interpreters.
Passing those tests establishes their specific assertions, not arbitrary
application scheduling or whole-program equivalence.

World-served handlers answer unary effects; custom world streaming is unsupported.
Checkpoints hold every registered reflected component; hosts save resources
and other state explicitly. Nondeterministic operations belong in host handlers.

## The witness: decisions beside the record

The effect log is the replay oracle: what a handler served and answered.
A `Witnessing` resource (`rig_ecs::bus::Witnessing::install(world, sink)`,
any `rig_core::observe::Witness`; `rig_core::observe::ObservationLog` is the
bounded in-memory one) records what happened *around* those exchanges, as
typed `Observation`s with a `Subject` (scope, dispatch order, effect id,
parent, key), a `Stage` (`Gate`, `Dispatch`, `Handler`, `Collect`, `Judge`,
`Runtime`, `Host`), an `Emitter` and an `Action`:

| site | fact |
|---|---|
| a `Gate` system writes `Held` / removes it | `Held` / `Released` (emitter unknown unless the policy emits) |
| a `Gate` system answers an intent before dispatch | `Denied { reason }` |
| `Dispatch` | `Issued`; `Refused { handler_unavailable \| reentrant \| ids_exhausted }` |
| a layer (`Intercept`) denies | `Denied { layer_discarded }` at `Handler`, the emitter named after the layer |
| `Collect` | `StreamTruncated { delivered, tail, errors }`, `Landed { outcome }`, `Replaced` when a layer's verdict differed from the record (emitter named after the layer that said it replaced; unknown for a difference no layer claimed), `Cancelled` for a despawn in flight |
| a `Judge` system overwrites a settled outcome (by insert; an in-place `Mut` rewrite must emit for itself) | `Replaced { recorded, consumed }`, whenever the whole value changed |
| the agent runtime | `Ended { settled \| max_turns \| provider \| cancelled \| … }` |
| Gemini unary and SSE drivers | `Adapter` send/status, usage, provider verdict/error envelope, EOF and closure facts at `Handler`, with execution-local operation and send ordinal; explicit invocation context takes precedence over the bus context |
| a host system | `Witnessing::emit(subject, Stage::Host, Emitter::versioned(..), HostAction::action(..))` |

Attach `bus::AdapterOperation` to a pending completion before dispatch when
the host owns retry correlation. The bus binds its current effect, scope and
parent while retaining the supplied operation's witness and send counter.
The component's nonzero host attempt ordinal stays separate from the HTTP
send ordinal. Without it, the bus creates an operation for the individual
effect. The component is runtime-only and is not scene/effect-log state.

The observation API does not promise semantic execution comparison. Tests use
`rig_core::test_utils::observations` for deterministic fixture comparisons;
applications own their comparison and aggregation rules. Optional host-clock
stamps are diagnostic data. No run, handler or provider interval is calculated.
`crates/rig-ecs/tests/bus_witness.rs` and `run_witness.rs` cover the emitted facts.

Checkpoints preserve named hold owners and the batch scheduling marker together.
A restored batch releases only its own hold; independent policy holds remain in
force. Bare `Held` barriers remain unknown holds and require host reevaluation.
Observation enablement does not affect hold ownership or checkpoint correctness.

## Replay delivery and streaming

Records remain in dispatch order. When `EffectLogResource::install` installs
an ECS recorder, the log also records `header.deliveries`: an ordered trace
of outcome insertions and stream item counts grouped by schedule pass.
Handler completion and collector delivery are separate boundaries. Replay
buffers ready handler data and exposes each recorded batch together, allowing
policy systems to run between batches. Concurrent live serving stays
concurrent. Coincident agent turns materialise and land tool batches in
`RunSeq` order, independent of irrelevant entity archetypes.

Use `Replay::policy_visible()` for policies that choose the first visible
answer or inspect partial `Streamed` state. Install an
`EffectLogRecorder::keeping_stream_events()` when recording streams for this
mode. It refuses missing delivery metadata or omitted stream bytes and
reports an inconsistent trace as `ReplayFailure`. The same policy must
reproduce recorded cancellations; cancelled losers are not given invented
outcome insertions. A replay that does not reproduce a required cancellation
fails explicitly.

Supported policy observation points are `On<Add, EffectOutcome>` and systems
ordered after the **entire** `BusSet::Collect`, with the same relevant ordering
live and on replay. Systems interleaved between individual collectors or
reading handler readiness/inboxes are outside this guarantee. Submit world
answers with `WorldOutcome` or typed `Answer<E>`; direct insertion of an
in-flight `EffectOutcome` records a `header.delivery_limitations` diagnostic
and causes policy replay to refuse the log. Gate denials and Judge replacements
retain their existing roles. A world answer submitted after Collect is
published in the following pass. `WorldOutcome` is transient like a ready
task; collect it before saving a checkpoint to retain the answer.

`Replay::default()` supports exchange consumers: it honors available delivery
batches, but a folded stream supplies only its final answer and recorded
cancellations are returned as errors. Logs without delivery metadata
replay exchanges without a policy-order guarantee. Keeping event bytes
preserves the event sequence; keeping delivery batches additionally preserves
which events partial-state policy sees together. `header.stream_errors`
retains error items at their original positions, including errors before or
after `Final`; a folded outcome cannot recover those positions. One event per pass is not
an exact replacement for a live multi-event batch. Kept events and the trace
increase log size in proportion to recorded events and delivery batches.

These guarantees require the same declared program and relevant schedule
ordering. They do not reproduce wall-clock gaps, arbitrary resources,
system-local state, ambient inputs or external writes. Saved IDs permit
subset replay; a program that creates new effects must still reproduce its
causal dispatches. Generic custom and world-served streaming remain
unsupported: `StreamWriter` does not change which effect families stream.

## Host assembly and restoration

The host owns model selection, credentials, endpoint/transport policy, SDK
preparation and runtime lifetime. ECS executes the already-built handler.
Store launch settings separately from execution checkpoints; a descriptor is
not proof of equal credentials, endpoint or transport policy.

`checkpoint::load_world(checkpoint, world, mode, handlers)` takes an explicit
`RestoreMode` and already-built `(HandlerKey, ErasedHandler)` pairs. `Strict`
checks the original saved descriptors before aliasing; `Replace` explicitly
accepts descriptor changes within the same effect family. Supplied handlers
are installed even when their descriptors match (credential rotation is not
a no-op). Omitted keys must already be served in the destination. The entire
saved binding set and graph are validated before state or handlers are installed.
Use `Strict` and `[]` to retain matching preinstalled handlers, including replayers.
Every unfinished effect must have an original saved handler contract; a destination
binding does not invent one. Saving captures data for inspection, not proof of
resumability. Inspect `requirements()` and call `validate(world)` before assembly;
actual implementation compatibility is checked at `load_world`.

[`host_resume`](examples/host_resume.rs) demonstrates host construction, strict
reconstruction and independent effect replay with an explicitly offline transport.

Select replay before credential lookup, diagnostic-secret collection or SDK
initialization. Old checkpoints containing removed provider-binding component
paths are refused; move their launch settings into the host explicitly rather
than silently dropping them. See [the contract](CONTRACT.md#121-host-assembly-runtime-execution).

## Stream and custom-answer snapshots

A completed stream's `Streamed` (events, errors, text, terminal fold) is
checkpoint data like any other component: it is restored before its answer is
inserted and its handler is never executed again. `load_world` returns a
result: it validates the whole checkpoint in a scratch world first and rejects
unfinished streams with already-delivered progress, leaving the destination
untouched. There is no generic provider cursor: save before progress or after
completion. Safe unanswered intents restart under their saved IDs (a taken but
unanswered effect loads as `Reserved(id)`).

Custom outcomes store the user's JSON value in `Outcome::Custom { payload }`.
Typed strings, scalars, arrays and objects round-trip without changing the
answer type, including objects with an `outcome` field of their own.

## Memory finalization across snapshots

`Settled` means the model run produced its answer, not that external memory
has committed it. `MemoryAppendScheduled` persists the fact that finalization
created an append; its child effect carries the request, dispatch id and
outcome. Loading a snapshot before finalization schedules that append;
loading a queued, in-flight or completed append does not create another
operation. See `tests/memory_resume.rs` for live-handler tests at these cuts.

An unanswered effect is retried on load under its saved id. The external
write may already have happened: a process can stop between the write and
collecting its answer. This is not exactly-once execution. A host needing
deduplication must durably associate an external idempotency key with the
operation (including a session/log namespace, since `EffectId` alone is not
globally unique), use an idempotent handler, or reconcile ambiguous writes
before resuming. Despawning or abandoning a world does not roll back writes.

Save only at a schedule boundary after deferred commands have been applied,
and persist the checkpoint together with its matching log. Inserting
components while loading invokes Bevy insertion observers and change
detection; application observers must not interpret rehydration as a new
business event. Install such observers after loading or explicitly guard
them during restoration.

## The prelude and the features

`rig_ecs::prelude` names what a user's systems need and nothing else: the sets (`RigSet`, `BusSet`), the components a user writes (`Cancelled`, `RequestPatch`, `Retry`, `Resolution`, `Held`, `UsesModel`, `Grant`, `Context`, `Remembers`, `Retrieves`) and the components a user reads (`Streamed`, `Outputs`, `EffectOutcome`, `RunResult`, `Settled`, `Failed`, `Usage`).

Every component of the bus and the graph derives `Reflect`; the rig-core values they hold reflect through opaque remote wrappers (`bus::reflect`, `agent::reflect` — serialized as their wire form, so an inspector shows an effect entity's payload as the log would); `checkpoint::register_types` (`reflect::install_reflect`) registers them all, and `RigPlugin` calls it. The checkpoint is canonical (entities in `Children` order, an `Entity` in a component as its index in the checkpoint), so a world and the world its checkpoint loads into export the same JSON (`tests/reflect_scene.rs`); every component round-trips through `ReflectSerializer` / `ReflectDeserializer` / `FromReflect` by value (`tests/reflect_roundtrip.rs`). The runtime-only components (`Serving`, `Streaming`, `Handler`, `Publishing`, `Observed`, `Asked`, `Answer`, `Typed`, and the asset handles) reflect nothing. rig-core takes no Bevy dependency.

`assets` (off by default): `assets::PromptAsset` (a `.md` / `.txt` file) and `assets::ToolDefinitions` (a `.json` array of `{ name, description, parameters }`) are `bevy_asset` assets with loaders; `PromptHandle` / `ToolsHandle` on an agent become its `Preamble` and its `Grant`s — one per definition, in file order, to the bound handler whose descriptor is the tool of that name; a definition nothing serves is not granted — the tick the asset loads, once (`Applied<A>`). `assets::AssetsPlugin` after `bevy_asset::AssetPlugin`; its systems run in `Update` in `assets::AssetsSet`, before `RigSchedule`. `tests/assets_prompt.rs`, `examples/prompt_from_assets.rs` (an in-memory source; a directory with the default one).

## The examples, side by side

Runnable programs over scripted mocks in `examples/support`: `agent_with_tools` (`Grant` links and a run entity for `dynamic_tools` and `prompt`), `human_in_the_loop` (a system in `BusSet::Gate` reading stdin for `AgentHook::on_dispatch`: approve, deny with an `EffectOutcome`, abort with `Cancelled`), `best_of_n` (`agent::fork` n − 1 times, a judging system over the settled runs, for a parallel fan-out), `streaming_ui` (a streamed run and a system after `RigSet::Fold` on `Changed<Streamed>` for a polled stream), `prompt_from_assets` (the `assets` feature). `cargo run -p rig-ecs --example <name>` — none needs a key.

## On wasm

Everything a system holds is `Send + Sync` on every target. The erased handler and the task are `!Send` on wasm, so on every target they live in the `NonSend` `HandlerTable` and `Executions` behind the `Handler`, `Serving` and `Streaming` component names — one storage, one code path — and a system that registers or dispatches runs on the main thread. `tests/bus_wasm.rs` drives the schedule by hand: `bevy_app`'s runner on the web is frame-scheduled by the browser.

Cancellation after an original handler answer has been observed records a
`DeliveryKind::Cancelled` boundary instead of an outcome delivery. This preserves
that original answer (including one awaiting a layer verdict) without claiming
that the consumer received it. Policy replay must reproduce the cancellation;
exchange replay returns cancellation and leaves undelivered terminal items hidden.
Ordinary cancellation without an observed answer retains its existing encoding.
