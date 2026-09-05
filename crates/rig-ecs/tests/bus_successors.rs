//! The successors of the rig-bus tests whose subjects left with the bus's
//! side channels: each keeps the old test's name and asserts what the
//! world-shaped bus does in the place of what the channel-shaped bus did.
//!
//! | rig-bus test (deleted) | subject there | here |
//! |---|---|---|
//! | `a_reopened_bus_serves_handles_bound_before_the_drop` | `Bus::reopen` after a driver died | nothing dies: the handler entity serves after every effect it served is gone |
//! | `reopen_while_a_driver_is_alive_is_refused` | the reopen race | there is no driver to race: another pass while effects are in flight is just a pass |
//! | `pendings_and_streams_created_while_closed_stay_closed_after_reopen` | no resurrection across a restart | an effect answered `HandlerUnavailable` stays answered once its key is bound |
//! | `a_rebind_before_registration_fails_at_first_dispatch_not_at_bind` | `Handle::rebind` before the handler | a scene loaded before its handlers are bound fails at the first pass, by key, never at load |
//! | `a_rebind_of_the_wrong_family_panics_at_the_hosts_line` | the family assertion at rebind | `Scene::first_gap` names the key whose bound family differs |
//! | `the_inbox_names_every_dispatch_that_ended_since_the_last_drain` | the completion inbox | `Added<EffectOutcome>` names every effect that ended since the system last ran |
//! | `the_inbox_is_bounded_and_counts_what_it_dropped` | the inbox's bound and drop count | nothing is dropped: a system that skips passes still sees every outcome |
//!
//! The loom model `loom_reopen_races_the_last_in_flight_reply` has no
//! successor: the race it proved safe (a reply from a dying driver against
//! a reopen) does not exist when the driver is a system.

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
    effect::{EffectFamily, FamilyDescriptor, HandlerDescriptor, HandlerKey},
    error::ErrorKind,
};
use rig_ecs::bus::{BusSet, EffectOutcome, InFlight, PendingEffect, RigSchedule, Scene};

#[test]
fn a_reopened_bus_serves_handles_bound_before_the_drop() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let before = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "served once", |world| {
        world.get::<EffectOutcome>(before).is_some()
    });
    // The fixture's moment: the driver's task despawned. Here every effect
    // the handler served is despawned instead, and the handler entity is
    // untouched — there is no driver whose death could take it.
    app.world_mut().despawn(before);
    tick(&mut app, 2);
    let after = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "served again", |world| {
        world.get::<EffectOutcome>(after).is_some()
    });
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 2);
}

#[test]
fn reopen_while_a_driver_is_alive_is_refused() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "in flight", |_| {
        counters.unary_started.load(Ordering::SeqCst) == 1
    });
    // Nothing to refuse: running the schedule again with an effect in
    // flight is another pass, and the effect stays exactly where it was.
    for _ in 0..5 {
        rig_ecs::bus::run_to_quiescence(app.world_mut());
        assert!(app.world().get::<InFlight>(effect).is_some());
    }
    counters.hold.release();
    tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        counters.unary_started.load(Ordering::SeqCst),
        1,
        "taken once"
    );
}

#[test]
fn pendings_and_streams_created_while_closed_stay_closed_after_reopen() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    // No handler bound: both are answered unavailable at the first pass.
    let unary = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    let stream = app
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();
    app.update();
    for effect in [unary, stream] {
        let outcome = app.world().get::<EffectOutcome>(effect).expect("answered");
        assert_eq!(
            outcome.0.as_ref().expect_err("unavailable").kind,
            ErrorKind::HandlerUnavailable
        );
    }
    // The key bound afterwards: nothing is resurrected — an answered
    // effect is never re-dispatched.
    register(&mut app, "model", MockModel::new(&counters));
    tick(&mut app, 3);
    assert_eq!(counters.unary_started.load(Ordering::SeqCst), 0);
    for effect in [unary, stream] {
        let outcome = app
            .world()
            .get::<EffectOutcome>(effect)
            .expect("still answered");
        assert_eq!(
            outcome.0.as_ref().expect_err("still unavailable").kind,
            ErrorKind::HandlerUnavailable
        );
    }
}

#[test]
fn a_rebind_before_registration_fails_at_first_dispatch_not_at_bind() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    // What a scene stored: the descriptor, and one effect, no handler yet.
    let scene = Scene {
        handlers: vec![HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: "gpt".into(),
                capabilities: Default::default(),
            },
            layers: Vec::new(),
        }],
        effects: vec![rig_ecs::bus::SceneEffect {
            streamed: None,
            seq: rig_ecs::bus::Seq(0),
            key: HandlerKey::from("model"),
            kind: completion(),
            id: None,
            outcome: None,
            parent: None,
            parent_ref: None,
            scope: None,
            held: false,
            tool_inputs: None,
            tool_outputs: None,
        }],
    };
    let loaded = scene.load(app.world_mut()).unwrap();
    assert_eq!(loaded.len(), 1, "the load succeeds with nothing bound");
    assert!(
        app.world().get::<EffectOutcome>(loaded[0]).is_none(),
        "not failed at load"
    );
    app.update();
    let outcome = app
        .world()
        .get::<EffectOutcome>(loaded[0])
        .expect("answered");
    assert_eq!(
        outcome.0.as_ref().expect_err("nothing serves it").kind,
        ErrorKind::HandlerUnavailable,
        "failed at the first pass, by key"
    );
    // Bound afterwards, a fresh effect on the same key works: the
    // descriptor a scene keeps is a claim about the key, not a binding.
    register(&mut app, "model", MockModel::new(&counters));
    let next = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "served", |world| {
        world.get::<EffectOutcome>(next).is_some()
    });
}

#[test]
fn a_rebind_of_the_wrong_family_panics_at_the_hosts_line() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let stored = HandlerDescriptor {
        key: HandlerKey::from("model"),
        family: FamilyDescriptor::Tool {
            name: "add".to_owned(),
            description: "adds".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            embedding: None,
        },
        layers: Vec::new(),
    };
    let scene = Scene {
        handlers: vec![stored],
        effects: Vec::new(),
    };
    let bound = rig_ecs::bus::Handlers::with(app.world_mut(), |handlers| handlers.descriptors())
        .expect("a bus");
    // No panic anywhere: the gap is data, named at the host's line by the
    // host's own check.
    let gap = scene.first_gap(&bound).expect("the family differs");
    assert_eq!(gap.key, HandlerKey::from("model"));
    assert_eq!(gap.family.family(), EffectFamily::Tool);
}

#[derive(Resource, Default)]
struct Ended(Vec<Entity>);

fn see_what_ended(landed: Query<Entity, Added<EffectOutcome>>, mut ended: ResMut<Ended>) {
    ended.0.extend(landed.iter());
}

#[test]
fn the_inbox_names_every_dispatch_that_ended_since_the_last_drain() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    app.init_resource::<Ended>();
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, see_what_ended.after(BusSet::Judge));
    let effects: Vec<Entity> = (0..3)
        .map(|_| {
            app.world_mut()
                .spawn(PendingEffect::new("model", completion()))
                .id()
        })
        .collect();
    tick_until(&mut app, "all seen", |world| {
        world.resource::<Ended>().0.len() == 3
    });
    let mut seen = app.world().resource::<Ended>().0.clone();
    seen.sort();
    let mut expected = effects.clone();
    expected.sort();
    assert_eq!(seen, expected, "each named once, as it ended");
    tick(&mut app, 3);
    assert_eq!(
        app.world().resource::<Ended>().0.len(),
        3,
        "and never again"
    );
}

#[test]
fn the_inbox_is_bounded_and_counts_what_it_dropped() {
    let counters = Arc::new(Counters::default());
    let mut app = app_with(rig_core::serve::ServingPolicy {
        command_capacity: 10_000,
        ..Default::default()
    });
    register(&mut app, "model", MockModel::new(&counters));
    let effects: Vec<Entity> = (0..2_000)
        .map(|_| {
            app.world_mut()
                .spawn(PendingEffect::new("model", completion()))
                .id()
        })
        .collect();
    // The observing system is added only after everything ended: it still
    // sees every outcome on its first run — `Added` is per system, and the
    // world dropped nothing while nobody was looking.
    tick_until(&mut app, "all answered", |world| {
        effects
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    app.init_resource::<Ended>();
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, see_what_ended.after(BusSet::Judge));
    app.update();
    assert_eq!(
        app.world().resource::<Ended>().0.len(),
        2_000,
        "nothing dropped"
    );
}
