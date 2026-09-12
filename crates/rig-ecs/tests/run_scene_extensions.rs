//! Registered application state is durable; omitted state is explicitly host-owned.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

mod bus_support;

use bevy_ecs::prelude::*;
use rig_ecs::agent::{
    Owner, Run, RunOf,
    scene::{SceneExtensions, WorldScene, load_world, save_world},
};
use serde::{Deserialize, Serialize};

#[derive(Component, Debug, PartialEq, Serialize, Deserialize)]
struct RetryBudget(u32);

#[derive(Resource)]
struct HostState(u32);

fn register(world: &mut World) {
    let mut registry = SceneExtensions::default();
    registry
        .register_component::<RetryBudget>("test/retry-budget/v1")
        .unwrap();
    world.insert_resource(registry);
}

fn scene() -> WorldScene {
    let mut world = World::new();
    register(&mut world);
    let agent = world.spawn(Owner("test".into())).id();
    world.spawn((Run, RunOf(agent), ChildOf(agent), RetryBudget(3)));
    world.insert_resource(HostState(8));
    let saved = save_world(&mut world).unwrap();
    serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap()
}

#[test]
fn custom_policy_roundtrips_on_remapped_graph_with_host_owned_resources() {
    let saved = scene();
    let mut world = World::new();
    register(&mut world);
    // Ensure the old entity ids cannot accidentally appear to work.
    for _ in 0..10 {
        world.spawn_empty();
    }
    world.insert_resource(HostState(99));
    let loaded = load_world(&saved, &mut world).unwrap();
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|e| world.get::<Run>(*e).is_some())
        .unwrap();
    assert_eq!(world.get::<RetryBudget>(run), Some(&RetryBudget(3)));
    let agent = world.get::<RunOf>(run).unwrap().0;
    assert_eq!(world.get::<ChildOf>(run).unwrap().parent(), agent);
    assert_eq!(world.get::<Owner>(agent).unwrap().0, "test");
    assert_eq!(world.resource::<HostState>().0, 99);
    // A subsequent policy consumes the restored state, not merely its JSON.
    let mut policy = Schedule::default();
    policy.add_systems(|mut budgets: Query<&mut RetryBudget, With<Run>>| {
        for mut budget in &mut budgets {
            budget.0 -= 1;
        }
    });
    policy.run(&mut world);
    assert_eq!(world.get::<RetryBudget>(run), Some(&RetryBudget(2)));
}

#[test]
fn missing_registration_and_invalid_payload_are_refused_before_spawning() {
    let saved = scene();
    let mut world = World::new();
    let count = world.entities().len();
    assert!(
        load_world(&saved, &mut world)
            .unwrap_err()
            .message
            .contains("unregistered")
    );
    assert_eq!(world.entities().len(), count);
    register(&mut world);
    let mut invalid = saved.clone();
    for components in invalid.extensions.values_mut() {
        components.insert(
            "test/retry-budget/v1".into(),
            serde_json::json!("not a budget"),
        );
    }
    assert!(load_world(&invalid, &mut world).is_err());
    assert_eq!(world.entities().len(), count);
    let mut invalid = saved;
    invalid.extensions.insert(usize::MAX, Default::default());
    assert!(
        load_world(&invalid, &mut world)
            .unwrap_err()
            .message
            .contains("index")
    );
    assert_eq!(world.entities().len(), count);
}

#[test]
fn unregistered_state_is_outside_the_scene_contract() {
    let mut world = World::new();
    world.spawn((Owner("test".into()), RetryBudget(5)));
    let saved = save_world(&mut world).unwrap();
    assert!(saved.extensions.is_empty());
    let mut restored = World::new();
    let loaded = load_world(&saved, &mut restored).unwrap();
    assert!(restored.get::<RetryBudget>(loaded.graph[0]).is_none());
}

#[test]
fn names_must_be_nonempty_and_unique() {
    let mut registry = SceneExtensions::default();
    assert!(registry.register_component::<RetryBudget>(" ").is_err());
    registry
        .register_component::<RetryBudget>("budget/v1")
        .unwrap();
    assert!(
        registry
            .register_component::<RetryBudget>("budget/v1")
            .is_err()
    );
}

#[test]
fn paired_graph_stream_scene_restores_completed_state_and_refuses_an_unfinished_prefix() {
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
        bus_support::tick_until(&mut live, "stream scene cut", |world| {
            if completed {
                world.get::<EffectOutcome>(effect).is_some()
            } else {
                world
                    .get::<Streamed>(effect)
                    .is_some_and(|streamed| !streamed.text.is_empty())
            }
        });
        let expected = serde_json::to_value(live.world().get::<Streamed>(effect).unwrap()).unwrap();
        let saved = save_world(live.world_mut()).unwrap();
        let saved: WorldScene =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        let mut restored = bus_support::app();
        register(restored.world_mut());
        let counters = Arc::new(bus_support::Counters::default());
        bus_support::register(
            &mut restored,
            "model",
            bus_support::MockModel::new(&counters),
        );
        let count = restored.world().entities().len();
        let loaded = load_world(&saved, restored.world_mut());
        if completed {
            let loaded = loaded.unwrap();
            restored.update();
            assert_eq!(
                serde_json::to_value(restored.world().get::<Streamed>(loaded.effects[0]).unwrap())
                    .unwrap(),
                expected
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
            assert!(loaded.unwrap_err().message.contains("unfinished stream"));
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
