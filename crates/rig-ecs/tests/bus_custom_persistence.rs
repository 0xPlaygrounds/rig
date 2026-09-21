//! Typed custom answers retain every JSON shape through logs and checkpoints.

use crate::bus_support;

use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::Replay;
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder};
use rig_core::effect::CustomEffect;
use rig_ecs::{
    bus::{Answer, Asked, EffectOutcome, Handlers, PendingEffect},
    checkpoint::{RestoreMode, load_world},
};
use serde_json::{Value, json};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Echo;

impl CustomEffect for Echo {
    const KIND: &'static str = "echo_json";
    type Answer = Value;
}

#[test]
fn typed_json_answers_round_trip_through_log_replay_and_checkpoint() {
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
fn typed_string_answer_keeps_its_type_through_log_replay_and_checkpoint() {
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
    let saved = bus_support::checkpoint(&mut live);

    let mut replay = bus_support::app();
    Replay::default()
        .register(replay.world_mut(), &log)
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
    // `echo` was served by a system, so the destination binds it the same way
    // rather than being handed a task handler.
    Handlers::with(restored.world_mut(), |handlers| {
        handlers.register_world::<E>("echo")
    })
    .unwrap()
    .unwrap();
    let loaded = load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let loaded = loaded.with::<PendingEffect>(restored.world())[0];
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
