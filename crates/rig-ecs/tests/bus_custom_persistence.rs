//! Typed custom answers retain every JSON shape through logs and scenes.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic
)]

mod bus_support;

use rig_core::effect::CustomEffect;
use rig_ecs::bus::{
    Answer, Asked, EffectLogResource, EffectOutcome, Handlers, PendingEffect, Replay, Scene,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use serde_json::{Value, json};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Echo;

impl CustomEffect for Echo {
    const KIND: &'static str = "echo_json";
    type Answer = Value;
}

#[test]
fn typed_json_answers_round_trip_through_log_replay_and_scene() {
    for value in [
        json!("approved"),
        json!(42),
        json!(true),
        Value::Null,
        json!([1, "two", null]),
        json!({"ok": true}),
        json!({"outcome": "user data", "payload": [false]}),
    ] {
        assert_persists(Echo, value);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Approval;

impl CustomEffect for Approval {
    const KIND: &'static str = "approval";
    type Answer = String;
}

#[test]
fn typed_string_answer_keeps_its_type_through_log_replay_and_scene() {
    assert_persists(Approval, "approved".to_owned());
}

fn assert_persists<E>(effect: E, value: E::Answer)
where
    E: CustomEffect + Send + Sync,
    E::Answer: Send + Sync + Clone + std::fmt::Debug + PartialEq,
{
    let mut live = bus_support::app();
    EffectLogResource::install(live.world_mut(), EffectLogRecorder::new());
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_world::<E>("echo")
    })
    .unwrap()
    .unwrap();
    let entity = live
        .world_mut()
        .spawn(PendingEffect::custom("echo", &effect).unwrap())
        .id();
    bus_support::tick_until(&mut live, "typed question", |world| {
        world.get::<Asked<E>>(entity).is_some()
    });
    live.world_mut()
        .entity_mut(entity)
        .insert(Answer::<E>(value.clone()));
    live.update();
    assert_eq!(
        live.world()
            .get::<EffectOutcome>(entity)
            .unwrap()
            .custom::<E>()
            .unwrap(),
        value
    );

    let log = live.world().resource::<EffectLogResource>().log();
    let log: EffectLog = serde_json::from_str(
        &serde_json::to_string(&log).expect("every JSON answer is persistable"),
    )
    .unwrap();
    let scene = Scene::save(live.world_mut());
    let scene: Scene = serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();

    let mut replay = bus_support::app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let replayed = Replay::load(replay.world_mut(), &log)[0];
    bus_support::tick_until(&mut replay, "replayed answer", |world| {
        world.get::<EffectOutcome>(replayed).is_some()
    });
    assert_eq!(
        replay
            .world()
            .get::<EffectOutcome>(replayed)
            .unwrap()
            .custom::<E>()
            .unwrap(),
        value
    );

    let mut restored = bus_support::app();
    let loaded = scene.load(restored.world_mut()).unwrap()[0];
    restored.update();
    assert_eq!(
        restored
            .world()
            .get::<EffectOutcome>(loaded)
            .unwrap()
            .custom::<E>()
            .unwrap(),
        value
    );
}
