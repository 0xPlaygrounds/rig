# rig-ecs

rig inside a Bevy `World`. Two layers: `rig_ecs::bus`, the effect bus as a plugin in the native shape — effects are entities, handlers are entities, the driver is a system, an outcome is a component, causality is `ChildOf`, a scene is a checkpoint — and the agent runtime over it (`agent`, `policy`, `systems`, `replay`): the run as a graph, the request as its fold. Nothing awaits, nothing blocks, nothing is probed; rig-agent is not in the graph; nothing is copied from rig-agent.

## The run as a graph

`agent::scene::WorldScene` saves the library's supported graph/effect state,
not arbitrary world state. Install `agent::scene::SceneExtensions` in both
worlds to register application components under stable, versioned names.
Registered serde components on saved graph entities round-trip with those
entities; a saved but unregistered name is rejected on load. Component payloads
must be entity-independent and their serde implementations pure and
deterministic. Embedded entity IDs are not remapped. Resources, custom effect
components, extra entities, system-local state and live tasks remain host-owned.
Bind handlers and install registrations before loading; install application
insertion observers afterward to avoid reacting to partially restored state.

`replay::stamp_run` captures supported effective configuration, including
run-over-agent overrides. `check_replayable` checks the requested run's exact
`Scope`; it does not search for another matching policy hash. Applications
must declare a nonempty `agent::PolicyVersion` for their custom systems,
ordering and otherwise-unhashed configuration. A missing declaration is
reported as unverified. This declaration is not an automatic code fingerprint
or a check of ambient credentials and external state. The builder-only
`stamp_header` remains a corpus header, not an effective compatibility check.

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

The request the model sees is derived, never authored: a run entity, utterances `ChildOf` it in `Order`, documents as their own entities attached to a turn by link entities, tools as the handler entities the bus already has (granted by link entities), the model as a relationship, every setting a component — and one function, `policy::fold_request`, that walks the graph at `RigSet::Assemble` and writes the wire `CompletionRequest` into the turn's `PendingEffect`. `CONTRACT.md` names the walk field by field with the golden that pins each; the corpus's third interpreter (`crates/rig-verify/tests/corpus/world.rs`) replays every completion-only golden through it.

| design (§3.1) | here |
|---|---|
| Agent | `Owner`, `Preamble`, `Temperature`, `MaxTokens`, `AdditionalParams`, `ToolChoiceSpec`, `Output { mode, schema }`, `MaxTurns`, `DefaultMaxTurns`, `InvalidCalls`; `UsesModel` → the model's handler entity; `Grant` link entities → tool handler entities; `Context` link entities → documents |
| Document | `DocumentId`, `DocumentText`, `DocumentProps`; attached to a turn by an `Attachment` link |
| Utterance | `Utterance`, `Role`, `Parts` (the message's parts, verbatim), `Order`; `ChildOf` the run |
| Run | `Run`, `RunOf` → agent, `RunSeq`, `Streamed`, `Cursor`, a phase (`Assembling`, `AwaitingModel`, `Settled`, `Failed(Failure)`), `RunResult`, `Usage`, `OutputRetries`, `OutputToolName`, the run's own overrides of the agent's settings, the bus's `Scope` |
| Turn | `Turn`, `ChildOf` the run, `Order`; `Advert` links → the tools it advertised; `Attachment` links → its documents; `Outputs` (per tick for a stream); `Reprompt`; `Batch` while its tool calls are out; `systems::{Fresh, Folded, Materialised}` |
| Effect | the bus module's, `ChildOf` the turn: the completion, then one per call to a granted tool (`ToolCallSlot` says which call; the bus's `ToolInputs` carries the run's `ToolContextSpec`) — the batch is the turn's children, `ToolPolicy { concurrency }` on the run or the agent says how many fly at once |
| Invalid call | `InvalidCall` + `Resolution`, `ChildOf` the turn |
| the scene | `agent::scene::RunScene` beside the bus's `Scene` |

The agent's sets, around the bus's:

| set | true before | written during |
|---|---|---|
| `RigSet::Advance` | a run in `Assembling` has no fresh turn | a turn with its adverts and attachments, or `Failed(MaxTurns)` |
| `RigSet::Select` | a run may lack a model of its own | the agent's `UsesModel`, copied — a routing system before it gives the run another |
| `RigSet::Assemble` | a fresh turn's graph is complete | the fold spawns the effect; the run is `AwaitingModel` |
| `RigSet::Patch` | the folded effect is a `PendingEffect` | the second steering slot: a user system rewrites the folded request |
| `RigSet::Release` | a turn's tool batch is out | `release_batch` un-holds the next calls up to the concurrency, in call order |
| *`BusSet::Gate` … `BusSet::Judge`* | | |
| `RigSet::Fold` | the effect may have streamed or landed | `Outputs` on the turn |
| `RigSet::Judge` | the turn's outputs are complete | a user system may rewrite them, or a tool child's `EffectOutcome` |
| `RigSet::Materialise` | a complete turn is unread, or its batch has landed | `land_batch`: one user utterance of the results in call order (CONTRACT §8), or a failure; `materialise`: the assistant utterance, the answer, a reprompt, an invalid call, the tool batch, or a failure |
| `RigSet::Settle` | a run settled or failed | observers on `Settled` / `Failed` |

`tests/tool_batch.rs` pins the batch: two calls are two children dispatched in call order and one utterance of results; `ToolPolicy` sets how many fly at once; a `Judge` system's replacement reaches history while the record keeps the answer; a `Gate` denial is a skipped result and no record; a despawned child fails the run `Cancelled`; a system's `Resolution::{Repair, Retry}` renames or retries an invalid call (`Skip` and `Ignore` likewise). The corpus's nesting cells (matrix Q) run through a key the world serves (`Handlers::register_open`), the `lookup` tool answered by a system that spawns its child `ChildOf` the call.

The first steering slot is any system before `Assemble`: it edits the graph. `tests/run_graph.rs` pins the wins: an utterance despawned leaves the next request; one document entity feeds two runs; a grant link advertises a tool and its removal un-advertises it; a model swapped on the run changes the next key; a `Patch` system's rewrite reaches the handler and the record; a system before `Assemble` rewrites an utterance. `tests/run_scene.rs` pins that the graph is the state: a run saved mid-turn resumes in a fresh world to the same second request.

## Every rig-agent hook action, as a system

No hook trait: a user system writes a component at a set boundary, a library system reads it later (CONTRACT §9). One row per rig-agent hook method and action, each naming the corpus cell that pins it (`crates/rig-verify/tests/corpus/world_hooks.rs` writes every one of them for the corpus; `tests/steer_hooks.rs` pins the library's side).

| rig-agent | here | pinned by |
|---|---|---|
| `on_run_start` — observe, dispatch | an observer `On<Add, Run>` (a dispatch: spawn a `PendingEffect` `ChildOf` the run — its `Seq` precedes the first completion's) | `anthropic_host_custom_at_start`, `anthropic_hooks_lookup_before_run`, `openai_host_embed_prompt`, `mock_oracle_rerank` |
| `on_run_start` — `stop(reason)` | insert `Cancelled(reason)` on the run | `mock_endings_stop_at_start` |
| `on_run_start` — `rewrite(prompt)` | rewrite the prompt `Utterance`'s `Parts` before `Assemble` | `run_graph::a_system_before_assemble_rewrites_an_utterance` |
| `on_model_select` — `select(label)` | a system after `RigSet::Advance`, before `RigSet::Select`, inserting `UsesModel(route)` on the run (`Route` links on the agent put the route in the required row) | `anthropic_serving_model_route`, `anthropic_shaping_route_on_first_turn`, `anthropic_shaping_late_route` |
| `on_model_select` — `stop` | `Cancelled` on the run, same moment | `mock_endings_stop_at_model_select` |
| `on_completion_call` — `patch(RequestPatch)` | `RequestPatch` on the fresh turn (a system on `Added<Fresh>` before `Assemble`); several merge in schedule order | `anthropic_hooks_preamble_override`, `anthropic_shaping_*` (every field), `anthropic_shaping_merged_three` |
| `on_completion_call` — dispatch | spawn the effect on `Added<Fresh>` before `Assemble`, so it precedes the completion | `anthropic_host_custom_at_completion_call` |
| `on_completion_call` — `stop` | `Cancelled` on the run, before or in `Patch` (the folded effect is despawned unissued: no record) | `mock_endings_stop_at_completion_call` |
| `on_dispatch` — `patch(kind)` on a tool | rewrite the tool child's `PendingEffect` in the bus's `Gate` | `anthropic_hooks_patch_tool_args{,_streamed}` |
| `on_dispatch` — `deny(reason)` / `skip` | `EffectOutcome(Err(Denied))` on the tool child in `Gate`: the skipped result the model sees, no record | `anthropic_hooks_deny_tool{,_streamed}` |
| `on_dispatch` — `stop` | `Cancelled` on the run from `Gate`: the child is despawned unissued | `anthropic_endings_tool_dispatch_cancelled{,_streamed}` |
| `on_outcome` — `replace` a tool result | rewrite the tool child's `EffectOutcome` in the bus's `Judge`: history holds the replacement, the record the answer | `anthropic_hooks_replace_tool_result`, `anthropic_hooks_two_hooks` |
| `on_outcome` — `replace` a completion | rewrite the turn's `Outputs.content` in `RigSet::Judge` | `anthropic_hooks_replace_answer` |
| `on_outcome` — `stop` after a tool | `Cancelled` from `On<Add, EffectOutcome>` on the tool child (the record holds the real answer, nothing is committed) | `anthropic_endings_tool_outcome_cancelled{,_streamed}` |
| `on_outcome` — `stop` on an answer | `Cancelled` in `RigSet::Judge` | `anthropic_endings_answer_outcome_cancelled` |
| `on_outcome` — dispatch | spawn from `On<Add, EffectOutcome>` on the tool child | `anthropic_host_custom_at_outcome{,_streamed}`, `anthropic_oracle_concurrent_notes` |
| `on_model_turn_finished` — `retry_with_feedback` / `repeat` | `Retry { feedback }` on the turn in `RigSet::Judge`; `materialise` makes the turn and the feedback history and asks again | `anthropic_hooks_demand_done` |
| `on_model_turn_finished` — `stop` | `Cancelled` in `RigSet::Judge` (a stateful stop reads `Cursor.turn`) | `anthropic_endings_turn_finished_stop{,_streamed}`, `anthropic_endings_answer_turn_stop`, `anthropic_oracle_stop_after_turn_two` |
| `on_invalid_tool_call` — `Retry { feedback }` / `Repair { tool_name }` / `Skip { reason }` / `Fail` / (unhandled: `Ignore`) | `Resolution::{Retry { feedback }, Repair { to }, Skip { reason }, Fail, Ignore}` on the `InvalidCall` entity, before `RigSet::Materialise` (CONTRACT §8.2) | `mock_invalid_*`, `mock_delta_{retry,repair,skip}`, `mock_hooks_retry_twice` |
| `on_text_delta` / `on_tool_call_delta` / `on_reasoning_delta` — observe, `stop` | a system after `RigSet::Fold` on `Changed<Outputs>` (text) or the effect's `Changed<Streamed>` (tool-call, reasoning deltas); a stop inserts `Cancelled`, the stream is the handler's to end | `anthropic_endings_text_delta_stop`, `anthropic_endings_tool_call_delta_stop`, `mock_delta_stop_on_{name,arguments}` |
| `on_run_settled` — observe, dispatch | an observer `On<Add, Settled>` / `On<Add, Failed>` | `anthropic_host_custom_at_settled`, `anthropic_host_custom_start_and_settled` |
| `observes(kind)` | a query; nothing to declare | `anthropic_hooks_observe_everything` |
| `tool_concurrency` | `ToolPolicy { concurrency }` on the run or the agent | `anthropic_serving_concurrent_concurrency_two` |
| `HookContext::bind` refused (a key nothing serves) | the system finds no `Bound` for the key and dispatches nothing | `anthropic_host_custom_unserved` |
| a hook's effect with no wire form | `PendingEffect::custom` refuses it; nothing is spawned | `mock_leftovers_unserializable_from_hook` |
| the chaining rules (`HookStack`) | schedule order: two systems patching one turn run in order; the first `Cancelled` ends the run; a `Retry` short-circuits because `materialise` reads it before committing | `anthropic_shaping_merged_three` |

Every hook cell of the corpus is written against the public sets and components above with no library change beyond the sets — the claim of `how-the-ecs-dissolves-rig-agent.md` §12, tested; the one set stage 4 added is `RigSet::Release` (stage 3's, between `Patch` and the bus's `Gate`). Memory and retrieval are components on the agent, not hooks: `Remembers(memory)` + `Conversation(id)` make a run load before its first turn and append at its settle (a `ClearAtStart` hook is an observer on the load's outcome, a `ClearAtSettled` one a system after `RigSet::Settle` on the append); `Retrieves(index)` + `Retrieval { samples, what }` links make `Advance` mark the turn `Retrieving` and `Assemble`'s first pass spawn one `Retrieve` effect per link before every fold, and `attach_retrieved` turn the results into attachments and adverts (a `Retrievable` grant is advertised only when retrieved). CONTRACT §11–§12.

## Vocabulary

| design (`rig-bevy-three-layer-design.md` §2) | here |
|---|---|
| a dispatch | `commands.spawn(PendingEffect { key, kind })`; `PendingEffect::{new, typed, custom}` |
| dispatch order | `Seq`, stamped on add from `SeqCounter` (global, reserved) |
| the effect's id | `Issued` after `Dispatch`; `Reserved` before it, for a scene's or a log's id |
| taken, in flight | `InFlight { key }` plus `Serving(Task)` (unary) or `Streaming { task, events, fold }` (stream) |
| a handler that is a system was asked | `Asked<E>`; the system answers with `Answer<E>` — or, for a key bound open (`Handlers::register_open`, any family), the effect entity itself, answered by submitting `WorldOutcome` |
| the answer | `EffectOutcome(Result<Outcome, ErrorReport>)`; a stream's per-tick fold in `Streamed { events, text, outcome }` |
| held by a decision | `Held` |
| a program's scope | `Scope(String)` on an ancestor; read into the record |
| a tool call's context (format 5: beside the effect, never in it) | `ToolInputs(ToolContext)` on the effect entity, attached to the handler's sink by `Dispatch`; what the tool published lands as `ToolOutputs(ToolContext)` when the outcome does (`Publishing` holds the slot in flight) |
| a handler | an entity with `Bound { key, descriptor }`; the erased handler in the `NonSend` `HandlerTable` |
| the registry | `Handlers` (a `SystemParam`): `register`, `register_erased`, `register_typed`, `register_world`, `register_open`, `deregister`, `descriptor`, `keys`, `descriptors`; `Handlers::with(world, ..)` outside a system |
| a typed view | `Typed<F>(Key<F>)`, wherever a system wants it |
| the driver | `dispatch` in `BusSet::Dispatch`; `collect_tasks`, `collect_streams`, `settle` in `BusSet::Collect` |
| interception | user systems in `BusSet::Gate` (patch, deny, hold) and `BusSet::Judge` (replace) |
| the record | `Recording` (any `rig_core::serve::Recorder`); `EffectLogResource` under `replay`; for a handler whose descriptor names layers, `Dispatch` installs a sink observer (`WorldObserver`, its slots in `Observed`) so a layer's `discard` and `patch` reach the record and the record keeps the innermost handler's answer |
| a scene | `Scene::{save, load, first_gap}` |
| replay | `Replay::{register, load}`, by id |
| the policy | `Policy(ServingPolicy)`: intake per tick, stream buffer, serial keys |

## The schedule

`BusPlugin` adds `RigSchedule` with four sets in order and runs it to quiescence from one exclusive system in `Update` (`run_to_quiescence`: while a plugin system marks `Progress`, at most `QUIESCENCE_CAP` passes). Users add their systems to `RigSchedule`, ordered against the sets, never to `Update`.

| set | true before | written during |
|---|---|---|
| `Gate` | pending effects are as spawned | a user system patches a `PendingEffect`, denies one (`EffectOutcome(Err(..))`), or holds one (`Held`) |
| `Dispatch` | every un-held, un-answered `PendingEffect` is a candidate | the plugin takes them in `Seq` order up to the tick's intake: `Issued`, `InFlight`, `Serving`/`Streaming`/`Asked`; a record opens |
| `Collect` | handlers may have finished or streamed | the plugin writes `Streamed`, `EffectOutcome`; the record closes (`settle`); `InFlight` goes |
| `Judge` | this pass's outcomes have landed and are recorded | a user system may rewrite an `EffectOutcome` before anything after `Judge` reads it |

Decisions are program, never record: the record is what the handler answered, taken in `Collect`, between the two slots. A `Gate` denial never had `InFlight`, so it is no record; a `Judge` rewrite is `Changed`, not `Added`, so it is not re-recorded. Despawning an effect cancels it (its task drops, the record says `Cancelled`) and Bevy despawns its `ChildOf` descendants with it.

## Serial serving and re-entrancy

Under `ServingPolicy::serial_per_handler`, `Dispatch` takes a key only when nothing is `InFlight` on it. An effect whose ancestor (up `ChildOf`) is in flight on its own key could only wait for itself: it is refused before any dispatch with a `Request` report and no record.

## Handlers that are systems

`Handlers::register_world::<E>(key)` binds a `WorldEffect` (a `CustomEffect` whose payload and answer are `Send + Sync`). A dispatch to the key lands as `Asked<E>` on the effect entity; a user system with any `World` access inserts `Answer<E>`; the plugin queues it for publication as `EffectOutcome` in `Collect`. `Handlers::register_open(key, family)` binds a key of any family to the world itself: the dispatch is taken and left on its entity, `InFlight`, for a system to answer by submitting `WorldOutcome::new(outcome)` — a tool a system serves, nesting what it needs as effects `ChildOf` the call (`bus_world::a_system_serves_an_open_tool_key_and_nests_a_completion_under_it`). The collector publishes answers in submission order; submissions after `Collect` become visible in the next pass. No task is required. Unary only: a system answers once. A handler that must reach the world is one of these; a handler served as a task cannot.

## The proofs

The Bevy host fixture's fourteen proofs and the eight unproven behaviours of `rig-bevy-requirements.md` §4.8, each a named test:

| proof / behaviour | test |
|---|---|
| 1 the bounds hold as components | `bus_effects::every_component_a_system_holds_is_send_sync` |
| 2 the handler's future is held in the entity | `bus_effects::a_pending_effect_is_taken_served_and_answered` |
| 3 answered across ticks, no waker per frame | `bus_effects::a_pending_effect_is_taken_served_and_answered` |
| 4 despawn in flight cancels; pre-dispatch despawn never serves | `bus_effects::despawning_an_effect_in_flight_cancels_its_handler`, `bus_effects::an_effect_despawned_before_dispatch_is_never_served` |
| 5 the intake bound blocks nobody | `bus_effects::the_intake_bound_leaves_the_rest_pending_and_blocks_nobody` |
| 6 a stream lands per tick; drop mid-stream cancels | `bus_effects::a_stream_accumulates_per_tick`, `bus_effects::despawning_mid_stream_cancels_the_handler` |
| 7 register, dispatch, deregister from systems | `bus_effects::handlers_are_registered_and_removed_from_systems` |
| 8 a system answers; the serial key waits for it | `bus_world::a_system_answers_an_asked_effect_and_the_key_waits_for_it` |
| 9 a handler survives (nothing dies, nothing is re-registered) | `bus_scale::handlers_outlive_every_effect_and_serve_again` |
| 10 scene round-trip | `bus_scene::a_scene_saves_intent_and_a_loaded_world_reissues_what_was_unanswered` |
| 11 a decision suspended, made from a system next tick; a denial is no record | `bus_world::a_held_effect_is_denied_or_approved_from_a_system_next_tick` |
| 12 nested dispatch: `ChildOf`, `parent` in the record, same-key nesting refused, despawn ⇒ `Cancelled` | `bus_world::a_child_effect_records_its_parent_and_a_reentrant_one_is_refused`, `bus_world::despawning_a_parent_cancels_its_children_in_flight_and_never_serves_the_queued` |
| 13 checkpointed scene resumed in a fresh world | `bus_scene::a_checkpoint_and_the_logs_tail_resume_in_a_fresh_world` |
| 14 ten thousand in flight cost one bounded tick | `bus_scale::ten_thousand_effects_in_flight_cost_one_bounded_tick` |
| §4.8 streaming per tick as a component | `bus_effects::a_stream_accumulates_per_tick` |
| §4.8 a typed key in a component across ticks | `bus_scene::a_typed_key_dispatches_across_ticks` |
| §4.8 a thousand pendings from four parallel spawners, `Seq` order in the log | `bus_scale::a_thousand_effects_from_four_parallel_spawners_resolve_in_seq_order` |
| §4.8 register over a live key; the family-change refusal | `bus_scene::a_live_key_is_reserved_and_never_changes_family` |
| §4.8 whole-tree cancel by recursive despawn | `bus_world::despawning_a_parent_cancels_its_children_in_flight_and_never_serves_the_queued` |
| §4.8 scene round-trip | `bus_scene::a_scene_saves_intent_and_a_loaded_world_reissues_what_was_unanswered` |
| §4.8 `EffectLog` as a resource with a replayer in the world | `bus_scene::three_goldens_replay_through_a_world_by_id` |
| §4.8 a streamed golden through a world | `bus_scene::three_goldens_replay_through_a_world_by_id`; every golden in `rig-verify/tests/world_replay.rs` |
| patch in `Gate`, replace in `Judge`, the record keeps the answer | `bus_world::gate_patches_and_judge_replaces_but_the_record_keeps_the_answer` |
| the quiescence cap ends the tick | `bus_scale::the_quiescence_cap_ends_the_tick` |
| the browser target, executed | `bus_wasm` (three tests under `wasm-bindgen-test-runner`) |
| rig-bus's deleted tests, under their old names | `bus_successors` |

## What it deliberately does not have

No hook trait, no history vector, no step enum, no run struct copied from anywhere, no batch machine (the batch is the turn's children and a query): steering is a system between sets. Program identity is data: `replay::stamp_run` writes the run's scope into `LogHeader::programs` and `replay::check_replayable` refuses a foreign log by policy or by row (`tests/run_identity.rs`). Memory is the graph and retrieval attaches (`tests/memory_graph.rs`); two runs on one agent are two `spawn_run`s; resume is a scene load (`agent::scene::{save_world, load_world}`, every resume and checkpoint row of the corpus as a world cell, CONTRACT §13). The `bus` module still has no agent-shaped item and its suite is agent-free (the guard checks). No streaming answers from a system yet (a later PR). `Scene` is the crate's own serde form and stores what this module owns; a host's other components are its own to save. No `Now`, no `Random`: nondeterminism is an effect a host registers, and the guard refuses a clock or a random draw in this crate.

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
task; collect it before saving a scene to retain the answer.

`Replay::default()` supports exchange consumers: it honors available delivery
batches, but a folded stream supplies only its final answer and recorded
cancellations are returned as errors. Older logs without delivery metadata
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
Rig already had a sans-IO serializable state machine before this ECS runtime;
these tests establish the specified ECS behavior, not an architectural verdict.

## Stream and custom-answer snapshots

`SceneEffect.streamed` preserves completed events, text and terminal fold.
A completed stream is restored before its answer is inserted and its handler
is never executed again. The nullable field is required on the wire:
`null` establishes no saved stream state; omission could conceal a lost prefix.
`Scene::load` and `load_with` return a result, as does the paired `WorldScene`
loader. They reject unfinished streams with already-delivered progress before
spawning entities. There is no generic provider cursor: save before progress
or after completion. Safe unanswered intents restart under their saved IDs.

Custom outcomes store the user's JSON value in `Outcome::Custom { payload }`.
Typed strings, scalars, arrays and objects round-trip without changing the
answer type, including objects with an `outcome` field of their own. The wire
shape differs from the previous flattened custom variant.

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
and persist the scene together with its matching log/checkpoint. Inserting
components while loading invokes Bevy insertion observers and change
detection; application observers must not interpret rehydration as a new
business event. Install such observers after loading or explicitly guard
them during restoration.

## The prelude and the features

`rig_ecs::prelude` names what a user's systems need and nothing else: the sets (`RigSet`, `BusSet`), the components a user writes (`Cancelled`, `RequestPatch`, `Retry`, `Resolution`, `Held`, `UsesModel`, `Grant`, `Context`, `Remembers`, `Retrieves`) and the components a user reads (`Streamed`, `Outputs`, `EffectOutcome`, `RunResult`, `Settled`, `Failed`, `Usage`).

`reflect` (off by default): every component of the bus and the graph derives `Reflect`, the rig-core values they hold reflect through opaque remote wrappers (`bus::reflect`, `agent::reflect` — serialized as their wire form, so an inspector shows an effect entity's payload as the log would), `reflect::ReflectPlugin` registers them all, and `reflect::ReflectedScene` is the world as reflected data beside the serde scene: canonical (entities ordered by content, an `Entity` in a component as its index in the scene, a relationship target's indexes sorted), so a world and the world its `WorldScene` loads into export the same JSON (`tests/reflect_scene.rs`); every component round-trips through `ReflectSerializer` / `ReflectDeserializer` / `FromReflect` by value (`tests/reflect_roundtrip.rs`). The runtime-only components (`Serving`, `Streaming`, `Publishing`, `Observed`, `Asked`, `Answer`, `Typed`, and the asset handles) reflect nothing. rig-core takes no Bevy dependency.

`assets` (off by default, implies `reflect`): `assets::Prompt` (a `.md` / `.txt` file) and `assets::ToolDefinitions` (a `.json` array of `{ name, description, parameters }`) are `bevy_asset` assets with loaders; `PromptHandle` / `ToolsHandle` on an agent become its `Preamble` and its `Grant`s — one per definition, in file order, to the bound handler whose descriptor is the tool of that name; a definition nothing serves is not granted — the tick the asset loads, once (`Applied<A>`). `assets::AssetsPlugin` after `bevy_asset::AssetPlugin`. `tests/assets_prompt.rs`, `examples/prompt_from_assets.rs` (an in-memory source; a directory with the default one).

## The examples, side by side

The same programs as rig's root examples, each a page of user code over a scripted mock (`examples/support`) — 49 to 102 lines each with their comments, not the thirty the design hoped for — so the translation is shown: `agent_with_tools` (`Grant` links and a run entity for `dynamic_tools` and `prompt`), `human_in_the_loop` (a system in `BusSet::Gate` reading stdin for `AgentHook::on_dispatch`: approve, deny with an `EffectOutcome`, abort with `Cancelled`), `best_of_n` (`agent::fork` n − 1 times, a judging system over the settled runs, for a parallel fan-out), `streaming_ui` (a streamed run and a system after `RigSet::Fold` on `Changed<Streamed>` for a polled stream), `prompt_from_assets` (the `assets` feature). `cargo run -p rig-ecs --example <name>` — none needs a key.

## On wasm

Everything a system holds is `Send + Sync` on every target. The erased handler lives in a `NonSend` resource on every target, one spelling, so a system that registers or dispatches runs on the main thread. `tests/bus_wasm.rs` drives the schedule by hand: `bevy_app`'s runner on the web is frame-scheduled by the browser.
