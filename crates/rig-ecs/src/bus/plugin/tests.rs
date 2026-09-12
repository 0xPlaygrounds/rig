#![allow(clippy::unwrap_used)]
use super::*;
use crate::bus::{Handlers, Issued, PendingEffect};
use rig_core::effect::{EffectKind, FamilyDescriptor};

fn custom() -> EffectKind {
    EffectKind::Custom {
        kind: "test".into(),
        payload: serde_json::Value::Null,
    }
}

fn world_with_capacity(command_capacity: usize, pending: usize) -> World {
    let mut world = World::new();
    install_bus(
        &mut world,
        ServingPolicy {
            command_capacity,
            ..ServingPolicy::default()
        },
    );
    Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "open",
            FamilyDescriptor::Custom {
                kind: "test".into(),
            },
        )
    })
    .unwrap()
    .unwrap();
    for _ in 0..pending {
        world.spawn(PendingEffect::new("open", custom()));
    }
    world
}

fn issued(world: &mut World) -> usize {
    world.query::<&Issued>().iter(world).count()
}

/// The intake bound measures issues per tick: the runner takes at most
/// `command_capacity` effects in one tick, however many passes the tick
/// runs, and the rest wait for later ticks rather than for anything else.
#[test]
fn the_runner_bounds_intake_per_tick_and_the_rest_wait_for_later_ticks() {
    let mut world = world_with_capacity(2, 5);
    run_to_quiescence(&mut world);
    assert_eq!(issued(&mut world), 2, "one tick issues at most the bound");
    run_to_quiescence(&mut world);
    assert_eq!(issued(&mut world), 4);
    run_to_quiescence(&mut world);
    assert_eq!(issued(&mut world), 5);
}

/// A host that runs `RigSchedule` itself, pass by pass (as `DeliveryBatch`
/// and the collection budget support), never calls the runner that resets
/// the tick's intake. Each of its passes is a tick for the bound: a queue
/// longer than `command_capacity` drains over several passes instead of
/// stalling at the bound for the life of the world.
#[test]
fn a_host_driven_pass_is_a_tick_for_the_intake_bound() {
    let mut world = world_with_capacity(2, 5);
    world.run_schedule(RigSchedule);
    assert_eq!(issued(&mut world), 2, "one pass issues at most the bound");
    for _ in 0..4 {
        world.run_schedule(RigSchedule);
    }
    assert_eq!(
        issued(&mut world),
        5,
        "the intake bound must reset for every host-driven pass"
    );
}
