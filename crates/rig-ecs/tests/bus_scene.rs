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
    checkpoint::{RestoreMode, load_world},
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
        assert_eq!(response.provider(), "mock");
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
