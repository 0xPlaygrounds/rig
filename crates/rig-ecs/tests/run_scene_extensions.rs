//! Registered host state is durable; unregistered state is explicitly
//! host-owned. A host component reflected and registered with the world's
//! type registry travels with the entity it sits on, its `Entity` fields
//! remapped like the crate's own; one the host never registered is not in
//! the checkpoint, and one the destination never registered is refused.
use crate::{bus_support, run_support};

use std::any::type_name;

use bevy_ecs::{prelude::*, reflect::AppTypeRegistry};
use bevy_reflect::Reflect;
use rig_core::error::ErrorKind;
use rig_ecs::{
    agent::{Owner, Run, RunOf, Settled, Turn},
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::RunCommands,
};
use serde::{Deserialize, Serialize};

/// A host policy on a run.
#[derive(Component, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
struct RetryBudget(u32);

/// A host link from a run to the entity it blames: an `Entity` field the
/// checkpoint remaps.
#[derive(Component, Debug, PartialEq, Reflect)]
#[reflect(Component)]
struct Blames(Entity);

#[derive(Resource)]
struct HostState(u32);

fn register(world: &mut World) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let mut registry = registry.write();
    registry.register::<RetryBudget>();
    registry.register::<Blames>();
}

/// The checkpoint through its wire form.
fn round_trip(checkpoint: &Checkpoint) -> Checkpoint {
    Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap()
}

/// A settled run under `t/model:default`, its host policy on the run and
/// its first utterance, blaming its agent — saved through the wire form.
fn checkpoint() -> Checkpoint {
    let mut app = run_support::app();
    app.register_type::<RetryBudget>();
    app.register_type::<Blames>();
    let (agent, _) = run_support::capturing_agent(&mut app, "t/model:default", "t", "ok");
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "hello", false, Some(1));
    run_support::ended(&mut app, run, "the run");
    assert!(app.world().get::<Settled>(run).is_some());
    let utterance = run_support::utterances_of(app.world_mut(), run)[0];
    app.world_mut()
        .entity_mut(run)
        .insert((RetryBudget(3), Blames(agent)));
    app.world_mut().entity_mut(utterance).insert(RetryBudget(1));
    app.world_mut().insert_resource(HostState(8));
    round_trip(&save_world(app.world_mut()).unwrap())
}

#[test]
fn a_registered_host_component_round_trips_on_the_remapped_graph_with_host_owned_resources() {
    let saved = checkpoint();
    let mut app = run_support::app();
    app.register_type::<RetryBudget>();
    app.register_type::<Blames>();
    run_support::capturing_agent(&mut app, "t/model:default", "t", "ok");
    // Ensure the old entity ids cannot accidentally appear to work.
    for _ in 0..10 {
        app.world_mut().spawn_empty();
    }
    app.world_mut().insert_resource(HostState(99));
    let loaded = load_world(&saved, app.world_mut(), RestoreMode::Strict, []).unwrap();
    let world = app.world_mut();
    let run = loaded.with::<Run>(world)[0];
    assert_eq!(world.get::<RetryBudget>(run), Some(&RetryBudget(3)));
    let agent = world.get::<RunOf>(run).unwrap().0;
    assert!(
        loaded.with::<Owner>(world).contains(&agent),
        "the run's agent is the loaded one"
    );
    assert_eq!(world.get::<Owner>(agent).unwrap().0, "t");
    assert_eq!(
        world.get::<Blames>(run),
        Some(&Blames(agent)),
        "the host's entity field names the loaded agent"
    );
    let utterance = run_support::utterances_of(world, run)[0];
    assert_eq!(world.get::<RetryBudget>(utterance), Some(&RetryBudget(1)));
    let turns: Vec<Entity> = loaded.with::<Turn>(world);
    assert!(!turns.is_empty());
    assert!(
        turns
            .iter()
            .all(|turn| world.get::<RetryBudget>(*turn).is_none()),
        "only the entities that carried the policy carry it"
    );
    assert_eq!(
        world.resource::<HostState>().0,
        99,
        "resources are the host's"
    );
    // A subsequent policy consumes the restored state, not merely its JSON.
    let mut policy = Schedule::default();
    policy.add_systems(|mut budgets: Query<&mut RetryBudget, With<Run>>| {
        for mut budget in &mut budgets {
            budget.0 -= 1;
        }
    });
    policy.run(world);
    assert_eq!(world.get::<RetryBudget>(run), Some(&RetryBudget(2)));
}

#[test]
fn missing_registration_and_invalid_payload_are_refused_before_spawning() {
    let saved = checkpoint();
    let mut app = run_support::app();
    run_support::capturing_agent(&mut app, "t/model:default", "t", "ok");
    let count = app.world().entities().len();
    let error = load_world(&saved, app.world_mut(), RestoreMode::Strict, []).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);
    assert!(
        error.message.contains(type_name::<RetryBudget>())
            || error.message.contains(type_name::<Blames>()),
        "names an unregistered host type: {error:?}"
    );
    assert_eq!(app.world().entities().len(), count);
    register(app.world_mut());
    let run = saved
        .entities
        .iter()
        .position(|entity| entity.contains_key(type_name::<Run>()))
        .unwrap();
    let mut invalid = saved.clone();
    invalid.entities[run].insert(
        type_name::<RetryBudget>().to_owned(),
        serde_json::json!("not a budget"),
    );
    let error = load_world(&invalid, app.world_mut(), RestoreMode::Strict, []).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request, "{error:?}");
    assert_eq!(app.world().entities().len(), count);
    let mut invalid = saved;
    invalid.entities[run].insert(
        type_name::<Blames>().to_owned(),
        serde_json::json!(usize::MAX),
    );
    let error = load_world(&invalid, app.world_mut(), RestoreMode::Strict, []).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request, "{error:?}");
    assert_eq!(app.world().entities().len(), count);
}

#[test]
fn unregistered_state_is_outside_the_checkpoint_contract() {
    let mut app = run_support::app();
    app.world_mut()
        .spawn((Owner("test".into()), RetryBudget(5)));
    let saved = save_world(app.world_mut()).unwrap();
    assert!(
        !saved
            .to_json()
            .unwrap()
            .contains(type_name::<RetryBudget>()),
        "an unregistered component is not in the checkpoint"
    );
    let mut restored = run_support::app();
    register(restored.world_mut());
    let loaded = load_world(&saved, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let agent = loaded.with::<Owner>(restored.world())[0];
    assert!(restored.world().get::<RetryBudget>(agent).is_none());
}

#[test]
fn a_stream_checkpoint_restores_completed_state_and_refuses_an_unfinished_prefix() {
    use rig_ecs::bus::{EffectOutcome, PendingEffect, Streamed};
    use std::sync::{Arc, atomic::Ordering};

    for completed in [false, true] {
        let mut live = bus_support::app();
        register(live.world_mut());
        let counters = Arc::new(bus_support::Counters::default());
        let model = if completed {
            bus_support::MockModel::new(&counters)
        } else {
            bus_support::MockModel::endless(&counters)
        };
        bus_support::register(&mut live, "model", model);
        let agent = live.world_mut().spawn(Owner("test".into())).id();
        let run = live
            .world_mut()
            .spawn((Run, RunOf(agent), ChildOf(agent), RetryBudget(2)))
            .id();
        let effect = live
            .world_mut()
            .spawn((
                PendingEffect::new("model", bus_support::streaming()),
                ChildOf(run),
            ))
            .id();
        bus_support::tick_until(&mut live, "stream checkpoint cut", |world| {
            if completed {
                world.get::<EffectOutcome>(effect).is_some()
            } else {
                world
                    .get::<Streamed>(effect)
                    .is_some_and(|streamed| !streamed.text.is_empty())
            }
        });
        let expected = serde_json::to_value(live.world().get::<Streamed>(effect).unwrap()).unwrap();
        let saved = bus_support::checkpoint(&mut live);
        let mut restored = bus_support::app();
        register(restored.world_mut());
        let counters = Arc::new(bus_support::Counters::default());
        bus_support::register(
            &mut restored,
            "model",
            bus_support::MockModel::new(&counters),
        );
        let count = restored.world().entities().len();
        let loaded = load_world(&saved, restored.world_mut(), RestoreMode::Strict, []);
        if completed {
            let loaded = loaded.unwrap();
            let effect = loaded.with::<PendingEffect>(restored.world())[0];
            restored.update();
            assert_eq!(
                serde_json::to_value(restored.world().get::<Streamed>(effect).unwrap()).unwrap(),
                expected
            );
            assert_eq!(
                restored.world().get::<ChildOf>(effect).map(ChildOf::parent),
                Some(loaded.with::<Run>(restored.world())[0]),
                "the effect is the loaded run's"
            );
            let budgets: Vec<_> = restored
                .world_mut()
                .query::<&RetryBudget>()
                .iter(restored.world())
                .map(|budget| budget.0)
                .collect();
            assert_eq!(budgets, [2]);
            assert_eq!(counters.stream_sends.load(Ordering::SeqCst), 0);
        } else {
            let error = loaded.unwrap_err();
            assert_eq!(error.kind, ErrorKind::Request, "{error:?}");
            assert_eq!(restored.world().entities().len(), count);
            assert_eq!(
                restored
                    .world_mut()
                    .query::<&RetryBudget>()
                    .iter(restored.world())
                    .count(),
                0
            );
        }
    }
}
