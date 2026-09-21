//! Checkpoints and replay: the fixture's proofs 10 and 13 in the native
//! shape, the goldens replayed through a world by id (the streamed one
//! included), the log as a resource, typed keys across ticks, and
//! re-registration.
//!
//! | proof / behaviour | test |
//! |---|---|
//! | 10 checkpoint round-trip: intent, ids, outcomes, causality | `a_checkpoint_saves_intent_and_a_loaded_world_reissues_what_was_unanswered` |
//! | 13 checkpointed log: the log's tail replayed over a fresh world | `a_checkpoint_and_the_logs_tail_resume_in_a_fresh_world` |
//! | §4.8 three goldens through a world, the streamed one asserting its events | `three_goldens_replay_through_a_world_by_id` |
//! | §4.8 `EffectLog` as a resource with a replayer inside the world | `three_goldens_replay_through_a_world_by_id` |
//! | §4.8 a typed key in a component across ticks | `a_typed_key_dispatches_across_ticks` |
//! | §4.8 register over a live key; the family-change refusal | `a_live_key_is_reserved_and_never_changes_family` |

use crate::bus_support;

use rig_core::serve::Dispatch;
use std::sync::{Arc, atomic::Ordering};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::Replay;
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder};
use rig_core::{
    completion::CompletionRequest,
    effect::{EffectFamily, EffectKind, HandlerKey, Key, Outcome, family},
    serve::Serve,
};
use rig_ecs::{
    bus::{EffectOutcome, Handlers, InFlight, Issued, PendingEffect, Reserved, Streamed, Typed},
    checkpoint::{Checkpoint, Counters as SavedCounters, RestoreMode, load_world},
};

/// A golden log from the corpus.
fn golden(name: &str) -> EffectLog {
    let path = format!(
        "{}/../rig-cassette/fixtures/effects/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden is committed");
    serde_json::from_str(&text).expect("the golden loads")
}

#[test]
fn completed_stream_survives_json_checkpoint_without_serving_again() {
    let (mut live, _, _) = served();
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();
    tick_until(&mut live, "stream complete", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let before = serde_json::to_value(live.world().get::<Streamed>(effect).unwrap()).unwrap();
    assert_eq!(before["events"].as_array().unwrap().len(), STREAM_CAP + 3);
    let saved = checkpoint(&mut live);
    let (mut restored, _, counters) = served();
    let loaded = load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let loaded = loaded.with::<PendingEffect>(restored.world())[0];
    tick(&mut restored, 3);
    let after = serde_json::to_value(
        restored
            .world()
            .get::<Streamed>(loaded)
            .expect("completed stream restored"),
    )
    .unwrap();
    assert_eq!(before, after);
    assert_eq!(counters.stream_sends.load(Ordering::SeqCst), 0);
    assert_eq!(
        serde_json::to_value(&restored.world().get::<EffectOutcome>(loaded).unwrap().0).unwrap(),
        before["outcome"]
    );
}

#[test]
fn unfinished_stream_with_observed_progress_is_refused_before_spawning() {
    let counters = Arc::new(Counters::default());
    let mut live = app();
    register(&mut live, "model", MockModel::endless(&counters));
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();
    tick_until(&mut live, "partial stream", |world| {
        world
            .get::<Streamed>(effect)
            .is_some_and(|streamed| !streamed.text.is_empty())
    });
    assert!(live.world().get::<EffectOutcome>(effect).is_none());
    let saved = checkpoint(&mut live);
    // The destination serves the saved key, so the refusal is about the
    // stream's missing cursor and nothing else.
    let mut restored = app();
    register(&mut restored, "model", MockModel::new(&counters));
    let before = restored.world().entities().len();
    let error = load_world(&saved, restored.world_mut(), RestoreMode::Strict, [])
        .expect_err("no cursor to prevent duplicate delivery");
    assert!(error.message.contains("unfinished stream"), "{error:?}");
    assert_eq!(restored.world().entities().len(), before);
}

#[test]
fn an_unfinished_stream_without_progress_restarts_under_its_saved_id() {
    let (mut live, _, counters) = served();
    counters.hold.hold();
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();
    tick_until(&mut live, "stream dispatched", |world| {
        world.get::<InFlight>(effect).is_some()
    });
    assert_eq!(counters.stream_sends.load(Ordering::SeqCst), 0);
    let id = live.world().get::<Issued>(effect).unwrap().0;
    let saved = checkpoint(&mut live);
    let (mut resumed, _, resumed_counters) = served();
    let loaded = load_world(&saved, resumed.world_mut(), RestoreMode::Strict, []).unwrap();
    let loaded = loaded.with::<PendingEffect>(resumed.world())[0];
    tick_until(&mut resumed, "restarted stream completed", |world| {
        world.get::<EffectOutcome>(loaded).is_some()
    });
    assert_eq!(resumed.world().get::<Issued>(loaded).unwrap().0, id);
    assert_eq!(
        resumed_counters.stream_sends.load(Ordering::SeqCst),
        STREAM_CAP
    );
    assert_eq!(
        resumed.world().get::<Streamed>(loaded).unwrap().text,
        "tick ".repeat(STREAM_CAP)
    );
    drop(live);
    counters.hold.release();
}

#[test]
fn a_checkpoint_saves_intent_and_a_loaded_world_reissues_what_was_unanswered() {
    let (mut app, _, counters) = served();
    // Answer the first, then hold the rest.
    answered(&mut app, "first answered");
    counters.hold.hold();
    let taken = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "second in flight", |world| {
        world.get::<InFlight>(taken).is_some()
    });
    let child = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), ChildOf(taken)))
        .id();
    let waiting = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    app.update();
    let taken_id = app.world().get::<Issued>(taken).expect("issued").0;
    let saved = checkpoint(&mut app);
    drop(app);
    let _ = (child, waiting);

    // A fresh world, the checkpoint loaded, the same handler bound: the
    // answered effect stays answered, the two taken-or-waiting ones are
    // re-issued — the taken one under its saved id — and the child is a
    // child again.
    let counters = Arc::new(Counters::default());
    let mut app = bus_support::app();
    register(&mut app, "model", MockModel::saying(&counters, "again"));
    let loaded = load_world(&saved, app.world_mut(), RestoreMode::Strict, []).unwrap();
    let loaded = loaded.with::<PendingEffect>(app.world());
    assert_eq!(loaded.len(), 4);
    tick_until(&mut app, "all answered", |world| {
        loaded
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    let world = app.world();
    assert_eq!(
        counters.unary_served.load(Ordering::SeqCst),
        3,
        "the answered effect was never re-dispatched"
    );
    let kept = loaded
        .iter()
        .filter(|entity| {
            text_of(&world.get::<EffectOutcome>(**entity).expect("answered").0)
                == "hello from the world"
        })
        .count();
    assert_eq!(kept, 1, "the answered one keeps its answer");
    let reissued = loaded
        .iter()
        .find(|entity| world.get::<Issued>(**entity).map(|issued| issued.0) == Some(taken_id))
        .expect("re-issued under the saved id");
    assert!(world.get::<Reserved>(*reissued).is_none(), "consumed");
    let child_entity = loaded
        .iter()
        .find(|entity| world.get::<ChildOf>(**entity).is_some())
        .expect("a child");
    assert_eq!(
        world.get::<ChildOf>(*child_entity).expect("child").parent(),
        *reissued
    );
}

#[test]
fn three_goldens_replay_through_a_world_by_id() {
    for name in [
        "anthropic_completion_smoke",
        "anthropic_concurrent_tools_serial",
        "anthropic_streaming_with_events",
    ] {
        let log = golden(name);
        let mut app = serial_app();
        Replay::default()
            .register(app.world_mut(), &log)
            .expect("the golden registers");
        EffectLogResource::install(app.world_mut(), EffectLogRecorder::keeping_stream_events());
        let entities = Replay::load(app.world_mut(), &log);
        assert_eq!(entities.len(), log.records.len(), "{name}");
        tick_until(&mut app, name, |world| {
            entities
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        });
        let world = app.world();
        let mut records: Vec<_> = log.records.iter().collect();
        records.sort_by_key(|record| record.id);
        for (entity, record) in entities.iter().zip(records) {
            let issued = world.get::<Issued>(*entity).expect("issued");
            assert_eq!(issued.0, record.id, "{name}: the recorded id");
            let outcome = world.get::<EffectOutcome>(*entity).expect("answered");
            assert_eq!(
                serde_json::to_value(&outcome.0).expect("serde"),
                serde_json::to_value(&record.outcome).expect("serde"),
                "{name}: record {} replays its outcome",
                record.id
            );
            if let Some(events) = &record.events {
                let streamed = world
                    .get::<Streamed>(*entity)
                    .expect("a streamed record replays as a stream");
                assert_eq!(
                    serde_json::to_value(&streamed.events).expect("serde"),
                    serde_json::to_value(events).expect("serde"),
                    "{name}: record {} replays its events in order",
                    record.id
                );
            }
        }
        // The world's own log of the replay is the golden again, record for
        // record.
        // Both logs are in begin order; under serial serving neither is id
        // order, so compare by id.
        let mut replayed = world.resource::<EffectLogResource>().log().records;
        replayed.sort_by_key(|record| record.id);
        let mut theirs_sorted = log.records.clone();
        theirs_sorted.sort_by_key(|record| record.id);
        assert_eq!(replayed.len(), theirs_sorted.len(), "{name}");
        for (mine, theirs) in replayed.iter().zip(theirs_sorted.iter()) {
            assert_eq!(mine.id, theirs.id);
            assert_eq!(mine.key, theirs.key);
            assert_eq!(mine.parent, theirs.parent, "{name}: causality survives");
            assert_eq!(
                serde_json::to_value(&mine.outcome).expect("serde"),
                serde_json::to_value(&theirs.outcome).expect("serde")
            );
        }
    }
}

#[test]
fn a_checkpoint_and_the_logs_tail_resume_in_a_fresh_world() {
    let log = golden("anthropic_concurrent_tools_serial");
    // Split the log: the first record is "done", the rest is the tail a
    // fresh world must serve.
    let scene_state = serde_json::json!({ "world": "the host's" });
    let (checkpoint, tail) = log.checkpoint(1, scene_state.clone());
    assert_eq!(checkpoint.at, 1);
    assert_eq!(checkpoint.state, scene_state);
    let resumed = EffectLog::from_checkpoint(&checkpoint, tail.clone()).expect("joins");
    assert_eq!(
        resumed.records.len(),
        log.records.len() - 1,
        "the tail under the head's header"
    );

    let mut app = serial_app();
    Replay::default()
        .register(app.world_mut(), &resumed)
        .expect("registers");
    // Only the tail is re-issued: the head is spawned answered, as a scene
    // would spawn it.
    let head = app
        .world_mut()
        .spawn((
            PendingEffect::new(log.records[0].key.clone(), log.records[0].kind.clone()),
            Issued(log.records[0].id),
            EffectOutcome(log.records[0].outcome.clone()),
        ))
        .id();
    let entities = Replay::load(app.world_mut(), &tail);
    tick_until(&mut app, "tail replayed", |world| {
        entities
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    let world = app.world();
    assert!(world.get::<InFlight>(head).is_none(), "never re-dispatched");
    for (entity, record) in entities.iter().zip(tail.records.iter()) {
        assert_eq!(world.get::<Issued>(*entity).expect("issued").0, record.id);
        let outcome = world.get::<EffectOutcome>(*entity).expect("answered");
        assert_eq!(
            serde_json::to_value(&outcome.0).expect("serde"),
            serde_json::to_value(&record.outcome).expect("serde")
        );
    }
}

#[derive(Component)]
struct Model(Typed<family::Completion>);

#[derive(Resource, Default)]
struct Asked(Vec<Entity>);

fn ask_through_the_typed_key(
    models: Query<&Model>,
    mut asked: ResMut<Asked>,
    mut commands: Commands,
) {
    if asked.0.len() < 3
        && let Some(model) = models.iter().next()
    {
        let request: CompletionRequest = request();
        let pending = model.0.pending(request).expect("wraps");
        asked.0.push(commands.spawn(pending).id());
    }
}

#[test]
fn a_typed_key_dispatches_across_ticks() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let key: Key<family::Completion> = Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_typed::<family::Completion>("model", MockModel::new(&counters))
            .expect("the family is proven")
    })
    .expect("a bus");
    let wrong = Handlers::with(app.world_mut(), |handlers| {
        handlers.register_typed::<family::Tool>("model", MockModel::new(&counters))
    })
    .expect("a bus");
    assert!(wrong.is_err(), "a completion handler is not a tool");
    app.world_mut().spawn(Model(Typed(key)));
    app.init_resource::<Asked>();
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(
            rig_ecs::bus::RigSchedule,
            ask_through_the_typed_key.in_set(rig_ecs::bus::BusSet::Gate),
        );
    tick_until(&mut app, "three asked and answered", |world| {
        let asked = world.resource::<Asked>().0.clone();
        asked.len() == 3
            && asked
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    let world = app.world();
    for entity in &world.resource::<Asked>().0 {
        let response = world
            .get::<EffectOutcome>(*entity)
            .expect("answered")
            .typed::<family::Completion>()
            .expect("a completion");
        assert_eq!(response.provider, "mock");
    }
}

/// A tool handler, to try binding over a completion key.
struct Echo;

impl Serve for Echo {
    type Family = family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: rig_core::effect::FamilyDescriptor::Tool {
                name: "echo".to_owned(),
                description: "echoes".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let EffectKind::ToolCall { args, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "not a tool call",
            )));
        };
        rig_core::serve::Reply::Outcome(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::success(args.into()),
        }))
    }
}

#[test]
fn a_live_key_is_reserved_and_never_changes_family() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app();
    let first = register(&mut app, "model", MockModel::saying(&counters, "first"));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "in flight", |_| {
        counters.unary_started.load(Ordering::SeqCst) == 1
    });
    // Re-register over the live key with the same family: the bound entity
    // is re-served; the dispatch in flight keeps its handler.
    let second = register(&mut app, "model", MockModel::saying(&counters, "second"));
    assert_eq!(first, second, "the same handler entity");
    // Another family is refused while the key is bound.
    let refused = Handlers::with(app.world_mut(), |handlers| handlers.register("model", Echo))
        .expect("a bus");
    assert!(refused.is_err(), "a tool cannot take a completion key");
    let described = Handlers::with(app.world_mut(), |handlers| {
        handlers.descriptor(&HandlerKey::from("model"))
    })
    .expect("a bus")
    .expect("still bound");
    assert_eq!(described.family.family(), EffectFamily::Completion);
    counters.hold.release();
    tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        text_of(
            &app.world()
                .get::<EffectOutcome>(effect)
                .expect("answered")
                .0
        ),
        "first",
        "the dispatch in flight kept the handler that took it"
    );
    let next = answered(&mut app, "answered by the second");
    assert_eq!(
        text_of(&app.world().get::<EffectOutcome>(next).expect("answered").0),
        "second"
    );
}

#[test]
fn a_loaded_checkpoint_never_collides_with_minted_ids() {
    // Answered effects with ids 0..3, loaded into a fresh world: the next
    // minted id must be past them, not 0 again — the surviving ids alone
    // say so, whatever counter the checkpoint carries.
    let (mut live, _, _) = served();
    for _ in 0..3 {
        answered(&mut live, "saved answered");
    }
    let mut saved = checkpoint(&mut live);
    saved.counters.next_id = 0;
    let (mut app, _, _) = served();
    let loaded = load_world(&saved, app.world_mut(), RestoreMode::Strict, []).unwrap();
    assert_eq!(loaded.with::<PendingEffect>(app.world()).len(), 3);
    let fresh = answered(&mut app, "minted");
    let minted = app.world().get::<Issued>(fresh).expect("issued").0;
    assert_eq!(minted.as_u64(), 3, "past every saved id");
}

#[test]
fn pruned_effects_do_not_refund_issued_ids_after_checkpoint_load() {
    for retain_lower in [false, true] {
        let (mut live, _, _) = served();
        let mut effects = Vec::new();
        for expected in 0..3 {
            let effect = answered(&mut live, "completed before pruning");
            assert_eq!(
                live.world().get::<Issued>(effect).unwrap().0.as_u64(),
                expected
            );
            effects.push(effect);
        }
        for (index, effect) in effects.into_iter().enumerate() {
            if !retain_lower || index != 0 {
                live.world_mut().despawn(effect);
            }
        }
        let saved = checkpoint(&mut live);
        let (mut restored, _, _) = served();
        load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
        // A second checkpoint with no fresh dispatch must preserve the same
        // allocation history even when no effect entities survived.
        let saved = checkpoint(&mut restored);
        let (mut twice, _, _) = served();
        load_world(&saved, twice.world_mut(), RestoreMode::Strict, []).unwrap();
        let next = answered(&mut twice, "fresh dispatch after pruning");
        assert_eq!(
            twice.world().get::<Issued>(next).unwrap().0.as_u64(),
            3,
            "pruning is not an ID refund; lower survivor={retain_lower}"
        );
    }
}

#[test]
fn removed_reservations_remain_consumed_after_checkpoint_load() {
    let mut live = app();
    let reserved = live
        .world_mut()
        .spawn((
            PendingEffect::new("model", completion()),
            Reserved(rig_core::effect::EffectId::from_raw(20)),
        ))
        .id();
    live.world_mut().despawn(reserved);
    let saved = checkpoint(&mut live);
    let (mut restored, _, _) = served();
    load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let next = answered(&mut restored, "fresh dispatch after reservation");
    assert_eq!(restored.world().get::<Issued>(next).unwrap().0.as_u64(), 21);
}

#[test]
fn exhausting_fresh_ids_refuses_the_next_dispatch_without_wrapping() {
    let (mut live, _, _) = served();
    live.world_mut().resource_mut::<rig_ecs::bus::IdCounter>().0 = u64::MAX - 1;
    let last = answered(&mut live, "last available id");
    assert_eq!(
        live.world().get::<Issued>(last).unwrap().0.as_u64(),
        u64::MAX - 1
    );
    let refused = answered(&mut live, "allocator exhaustion");
    let error = live
        .world()
        .get::<EffectOutcome>(refused)
        .unwrap()
        .0
        .as_ref()
        .unwrap_err();
    assert_eq!(error.kind, rig_core::error::ErrorKind::Request);
    assert!(error.message.contains("exhausted"));
    assert!(live.world().get::<Issued>(refused).is_none());
    assert_eq!(
        live.world().resource::<rig_ecs::bus::IdCounter>().0,
        u64::MAX
    );
}

#[test]
fn maximum_reserved_id_is_refused_without_overflowing_the_insertion_hook() {
    let (mut live, _, _) = served();
    let reserved = live
        .world_mut()
        .spawn((
            PendingEffect::new("model", completion()),
            Reserved(rig_core::effect::EffectId::from_raw(u64::MAX)),
        ))
        .id();
    tick_until(&mut live, "invalid maximum reservation", |world| {
        world.get::<EffectOutcome>(reserved).is_some()
    });
    assert_eq!(
        live.world()
            .get::<EffectOutcome>(reserved)
            .unwrap()
            .0
            .as_ref()
            .unwrap_err()
            .kind,
        rig_core::error::ErrorKind::Request
    );
    assert!(live.world().get::<Issued>(reserved).is_none());
    assert_eq!(
        live.world().resource::<rig_ecs::bus::IdCounter>().0,
        u64::MAX
    );
}

#[test]
fn maximum_issued_id_never_wraps_the_insertion_hook() {
    let mut live = app();
    live.world_mut()
        .spawn(Issued(rig_core::effect::EffectId::from_raw(u64::MAX)));
    assert_eq!(
        live.world().resource::<rig_ecs::bus::IdCounter>().0,
        u64::MAX
    );
}

#[test]
fn cancelled_highest_id_remains_consumed_after_checkpoint_load() {
    let (mut live, _, counters) = served();
    EffectLogResource::install(live.world_mut(), EffectLogRecorder::new());
    answered(&mut live, "lower completed");
    counters.hold.hold();
    let highest = live
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut live, "highest in flight", |world| {
        world.get::<InFlight>(highest).is_some()
    });
    assert_eq!(live.world().get::<Issued>(highest).unwrap().0.as_u64(), 1);
    live.world_mut().despawn(highest);
    let log = live.world().resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 2);
    assert!(
        matches!(&log.records[1].outcome, Err(error) if error.kind == rig_core::error::ErrorKind::Cancelled)
    );
    let saved = checkpoint(&mut live);
    counters.hold.release();
    let mut restored = app();
    register(&mut restored, "model", MockModel::new(&counters));
    load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let fresh = answered(&mut restored, "after cancelled highest");
    assert_eq!(restored.world().get::<Issued>(fresh).unwrap().0.as_u64(), 2);
}

#[test]
fn checkpoint_counter_never_rewinds_a_used_destination_and_preserves_exhaustion() {
    for next_id in [0, 3, 101, u64::MAX] {
        let saved = Checkpoint {
            counters: SavedCounters {
                next_id,
                ..SavedCounters::default()
            },
            ..Checkpoint::default()
        };
        let saved = Checkpoint::from_json(&saved.to_json().unwrap()).unwrap();
        let mut restored = app();
        restored
            .world_mut()
            .resource_mut::<rig_ecs::bus::IdCounter>()
            .0 = 100;
        load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
        let expected = next_id.max(100);
        assert_eq!(
            restored.world().resource::<rig_ecs::bus::IdCounter>().0,
            expected
        );
        let counters = Arc::new(Counters::default());
        register(&mut restored, "model", MockModel::new(&counters));
        EffectLogResource::install(restored.world_mut(), EffectLogRecorder::new());
        let fresh = answered(&mut restored, "used destination");
        if expected == u64::MAX {
            assert!(restored.world().get::<Issued>(fresh).is_none());
            assert!(
                restored
                    .world()
                    .resource::<EffectLogResource>()
                    .log()
                    .records
                    .is_empty()
            );
            assert_eq!(
                restored
                    .world()
                    .get::<EffectOutcome>(fresh)
                    .unwrap()
                    .0
                    .as_ref()
                    .unwrap_err()
                    .kind,
                rig_core::error::ErrorKind::Request
            );
        } else {
            assert_eq!(
                restored.world().get::<Issued>(fresh).unwrap().0.as_u64(),
                expected
            );
        }
    }
}

#[test]
fn an_exhausted_checkpoint_can_resume_an_existing_reservation_but_cannot_mint() {
    for reserved_id in [5, u64::MAX - 1] {
        let (mut live, _, _) = served();
        live.world_mut().spawn((
            PendingEffect::new("model", completion()),
            Reserved(rig_core::effect::EffectId::from_raw(reserved_id)),
        ));
        live.world_mut().resource_mut::<rig_ecs::bus::IdCounter>().0 = u64::MAX;
        let saved = checkpoint(&mut live);
        assert_eq!(saved.counters.next_id, u64::MAX, "exhaustion is saved");
        let (mut restored, _, _) = served();
        EffectLogResource::install(restored.world_mut(), EffectLogRecorder::new());
        let loaded = load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
        let loaded = loaded.with::<PendingEffect>(restored.world())[0];
        tick_until(&mut restored, "resume with exhausted allocator", |world| {
            world.get::<EffectOutcome>(loaded).is_some()
        });
        assert_eq!(
            restored.world().get::<Issued>(loaded).unwrap().0.as_u64(),
            reserved_id
        );
        assert!(
            restored
                .world()
                .get::<EffectOutcome>(loaded)
                .unwrap()
                .0
                .is_ok()
        );
        let fresh = answered(&mut restored, "fresh allocation remains exhausted");
        assert!(restored.world().get::<Issued>(fresh).is_none());
        assert_eq!(
            restored
                .world()
                .resource::<EffectLogResource>()
                .log()
                .records
                .len(),
            1
        );
        assert_eq!(
            restored
                .world()
                .get::<EffectOutcome>(fresh)
                .unwrap()
                .0
                .as_ref()
                .unwrap_err()
                .kind,
            rig_core::error::ErrorKind::Request
        );
        assert_eq!(
            restored.world().resource::<rig_ecs::bus::IdCounter>().0,
            u64::MAX
        );
    }
}

#[test]
fn a_malformed_checkpoint_is_refused_before_world_mutation() {
    let mut live = app();
    for id in [7, 8] {
        live.world_mut().spawn((
            PendingEffect::new("model", completion()),
            Reserved(rig_core::effect::EffectId::from_raw(id)),
        ));
    }
    let original = checkpoint(&mut live);
    assert_eq!(original.entities.len(), 2);
    let child_of = std::any::type_name::<ChildOf>();
    for fault in ["missing-parent", "unknown-type", "malformed-value"] {
        let mut bad = original.clone();
        match fault {
            "missing-parent" => {
                bad.entities[0].insert(child_of.to_owned(), serde_json::json!(2));
            }
            "unknown-type" => {
                bad.entities[1].insert("host::Unknown".to_owned(), serde_json::json!({}));
            }
            "malformed-value" => {
                bad.entities[1].insert(
                    std::any::type_name::<Reserved>().to_owned(),
                    serde_json::json!("not an id"),
                );
            }
            _ => panic!("unknown malformed-checkpoint case: {fault}"),
        }
        let bad = Checkpoint::from_json(&bad.to_json().unwrap()).unwrap();
        let mut restored = app();
        restored
            .world_mut()
            .resource_mut::<rig_ecs::bus::IdCounter>()
            .0 = 5;
        let count = restored.world().entities().len();
        let error =
            load_world(&bad, restored.world_mut(), RestoreMode::Strict, []).expect_err(fault);
        assert_eq!(error.kind, rig_core::error::ErrorKind::Request, "{fault}");
        assert_eq!(
            restored.world().entities().len(),
            count,
            "no spawn on {fault}"
        );
        assert_eq!(restored.world().resource::<rig_ecs::bus::IdCounter>().0, 5);
    }
}
