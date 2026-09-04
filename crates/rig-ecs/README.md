# rig-ecs

rig inside a Bevy `World`. One module today, `rig_ecs::bus`: the effect bus as a plugin, in the native shape — effects are entities, handlers are entities, the driver is a system, an outcome is a component, causality is `ChildOf`, a scene is a checkpoint. Nothing awaits, nothing blocks, nothing is probed; rig-bus is not in the graph.

The `bus` module is written as if it were already its own crate (every item `pub` or private to its file, no import from a sibling module, no agent-shaped identifier, its tests in `tests/bus_*.rs`, a root guard enforcing all four) and becomes `rig-bevy` by a `git mv` when a second consumer exists. The agent runtime the later modules add consumes it through its public items only.

```rust,ignore
App::new()
    .add_plugins((ScheduleRunnerPlugin::default(), BusPlugin::default()))
    .add_systems(Startup, (register_the_model, ask).chain())
    .add_observer(print_the_answer)
    .run();

fn register_the_model(mut handlers: Handlers) {
    handlers.register("model", CompletionAdapter::new("gpt", client)).ok();
}

fn ask(mut commands: Commands) {
    commands.spawn(PendingEffect::new("model", EffectKind::Completion { request, stream: false }));
}

fn print_the_answer(answered: On<Add, EffectOutcome>, outcomes: Query<&EffectOutcome>) {
    println!("{:?}", outcomes.get(answered.event().entity));
}
```

`examples/hello_model.rs` is that program over a scripted mock.

## Vocabulary

| design (`rig-bevy-three-layer-design.md` §2) | here |
|---|---|
| a dispatch | `commands.spawn(PendingEffect { key, kind })`; `PendingEffect::{new, typed, custom}` |
| dispatch order | `Seq`, stamped on add from `SeqCounter` (global, reserved) |
| the effect's id | `Issued` after `Dispatch`; `Reserved` before it, for a scene's or a log's id |
| taken, in flight | `InFlight { key }` plus `Serving(Task)` (unary) or `Streaming { task, events, fold }` (stream) |
| a handler that is a system was asked | `Asked<E>`; the system answers with `Answer<E>` |
| the answer | `EffectOutcome(Result<Outcome, ErrorReport>)`; a stream's per-tick fold in `Streamed { events, text, outcome }` |
| held by a decision | `Held` |
| a program's scope | `Scope(String)` on an ancestor; read into the record |
| a handler | an entity with `Bound { key, descriptor }`; the erased handler in the `NonSend` `HandlerTable` |
| the registry | `Handlers` (a `SystemParam`): `register`, `register_erased`, `register_typed`, `register_world`, `deregister`, `descriptor`, `keys`, `descriptors`; `Handlers::with(world, ..)` outside a system |
| a typed view | `Typed<F>(Key<F>)`, wherever a system wants it |
| the driver | `dispatch` in `BusSet::Dispatch`; `collect_tasks`, `collect_streams`, `settle` in `BusSet::Collect` |
| interception | user systems in `BusSet::Gate` (patch, deny, hold) and `BusSet::Judge` (replace) |
| the record | `Recording` (any `rig_core::serve::Recorder`); `EffectLogResource` under `replay` |
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

`Handlers::register_world::<E>(key)` binds a `WorldEffect` (a `CustomEffect` whose payload and answer are `Send + Sync`). A dispatch to the key lands as `Asked<E>` on the effect entity; a user system with any `World` access inserts `Answer<E>`; the plugin turns it into the `EffectOutcome`. No sink, no mailbox, no task. Unary only: a system answers once. A handler that must reach the world is one of these; a handler served as a task cannot.

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

No agent, no loop, no memory semantics, no hook trait, no policy vocabulary: those are the crate's later modules, and nothing here may anticipate them. No streaming answers from a system yet (a later PR). No `reflect` yet: `Scene` is the crate's own serde form and stores what this module owns; a host's other components are its own to save until the `reflect` PR extends it. No `Now`, no `Random`: nondeterminism is an effect a host registers, and the guard refuses a clock or a random draw in this crate.

## On wasm

Everything a system holds is `Send + Sync` on every target. The erased handler lives in a `NonSend` resource on every target, one spelling, so a system that registers or dispatches runs on the main thread. `tests/bus_wasm.rs` drives the schedule by hand: `bevy_app`'s runner on the web is frame-scheduled by the browser.
