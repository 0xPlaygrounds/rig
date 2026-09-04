//! Scenes and replay: the fixture's proofs 10 and 13 in the native shape,
//! the goldens replayed through a world by id (the streamed one included),
//! the log as a resource, typed keys across ticks, and re-registration.
//!
//! | proof / behaviour | test |
//! |---|---|
//! | 10 scene round-trip: intent, ids, outcomes, causality | `a_scene_saves_intent_and_a_loaded_world_reissues_what_was_unanswered` |
//! | 13 checkpointed scene: the log's tail replayed over a fresh world | `a_checkpoint_and_the_logs_tail_resume_in_a_fresh_world` |
//! | §4.8 three goldens through a world, the streamed one asserting its events | `three_goldens_replay_through_a_world_by_id` |
//! | §4.8 `EffectLog` as a resource with a replayer inside the world | `three_goldens_replay_through_a_world_by_id` |
//! | §4.8 a typed key in a component across ticks | `a_typed_key_dispatches_across_ticks` |
//! | §4.8 register over a live key; the family-change refusal | `a_live_key_is_reserved_and_never_changes_family` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod bus_support;

use std::sync::{Arc, atomic::Ordering};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{
    completion::CompletionRequest,
    effect::{EffectFamily, EffectKind, HandlerKey, Key, Outcome, family},
    serve::{OutcomeSink, Serve},
};
use rig_ecs::bus::{
    EffectLogResource, EffectOutcome, Handlers, InFlight, Issued, PendingEffect, Replay, Reserved,
    Scene, Streamed, Typed,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};

/// A golden log from the corpus.
fn golden(name: &str) -> EffectLog {
    let path = format!(
        "{}/../rig-verify/fixtures/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden is committed");
    serde_json::from_str(&text).expect("the golden loads")
}

#[test]
fn a_scene_saves_intent_and_a_loaded_world_reissues_what_was_unanswered() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let answered = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    // Answer the first, then hold the rest.
    counters.hold.release();
    tick_until(&mut app, "first answered", |world| {
        world.get::<EffectOutcome>(answered).is_some()
    });
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

    let scene = Scene::save(app.world_mut());
    assert_eq!(scene.handlers.len(), 1);
    assert_eq!(scene.effects.len(), 4);
    let json = serde_json::to_string(&scene).expect("serde");
    assert!(!json.contains("Entity"), "no entity ids in a scene");
    let scene: Scene = serde_json::from_str(&json).expect("serde");
    let saved_child = scene
        .effects
        .iter()
        .find(|effect| effect.parent.is_some())
        .expect("the child");
    let parent_index = saved_child.parent.expect("has a parent");
    assert_eq!(scene.effects[parent_index].id, Some(taken_id));
    assert!(
        scene.effects[0].outcome.is_some(),
        "the answered one is answered"
    );
    drop(app);
    let _ = (child, waiting);

    // A fresh world, the scene loaded, the same handler bound: the answered
    // effect stays answered, the two taken-or-waiting ones are re-issued —
    // the taken one under its saved id — and the child is a child again.
    let counters = Arc::new(Counters::default());
    let mut app = bus_support::app();
    register(&mut app, "model", MockModel::saying(&counters, "again"));
    let loaded = scene.load(app.world_mut());
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
    assert_eq!(
        text_of(&world.get::<EffectOutcome>(loaded[0]).expect("kept").0),
        "hello from the world"
    );
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
        Handlers::with(app.world_mut(), |handlers| {
            Replay::default()
                .register(handlers, &log)
                .expect("the golden registers")
        })
        .expect("a bus");
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
    Handlers::with(app.world_mut(), |handlers| {
        Replay::default()
            .register(handlers, &resumed)
            .expect("registers")
    })
    .expect("a bus");
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

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let EffectKind::ToolCall { args, context, .. } = kind else {
            sink.resolve(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "not a tool call",
            )))
            .await;
            return;
        };
        sink.resolve(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::success(args.into()),
            context,
        }))
        .await;
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
    let next = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "answered by the second", |world| {
        world.get::<EffectOutcome>(next).is_some()
    });
    assert_eq!(
        text_of(&app.world().get::<EffectOutcome>(next).expect("answered").0),
        "second"
    );
}

#[test]
fn a_loaded_scene_never_collides_with_minted_ids() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    // A scene of answered effects with ids 0..3, loaded into a fresh world:
    // the next minted id must be past them, not 0 again.
    let scene = Scene {
        handlers: Vec::new(),
        effects: (0..3)
            .map(|n| rig_ecs::bus::SceneEffect {
                seq: rig_ecs::bus::Seq(n),
                key: HandlerKey::from("model"),
                kind: completion(),
                id: Some(rig_core::effect::EffectId::from_raw(n)),
                outcome: Some(Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Cancelled,
                    "saved answered",
                ))),
                parent: None,
                scope: None,
                held: false,
            })
            .collect(),
    };
    let loaded = scene.load(app.world_mut());
    let fresh = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "minted", |world| {
        world.get::<EffectOutcome>(fresh).is_some()
    });
    let minted = app.world().get::<Issued>(fresh).expect("issued").0;
    assert_eq!(minted.as_u64(), 3, "past every saved id");
    assert_eq!(loaded.len(), 3);
}
