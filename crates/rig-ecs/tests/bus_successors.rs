//! The successors of the rig-bus tests whose subjects left with the bus's
//! side channels: each keeps the old test's name and asserts what the
//! world-shaped bus does in the place of what the channel-shaped bus did.
//!
//! | rig-bus test (deleted) | subject there | here |
//! |---|---|---|
//! | `reopen_while_a_driver_is_alive_is_refused` | the reopen race | there is no driver to race: another pass while effects are in flight is just a pass |
//! | `pendings_and_streams_created_while_closed_stay_closed_after_reopen` | no resurrection across a restart | an effect answered `HandlerUnavailable` stays answered once its key is bound |
//! | `an_unfinished_effect_without_a_saved_contract_is_refused_before_loading` | checkpoint restoration | an unfinished effect without a saved handler contract refuses before any resumed execution |
//! | `a_rebind_of_the_wrong_family_panics_at_the_hosts_line` | the family assertion at rebind | a checkpoint whose key is bound here to another family is refused as data, before anything is spawned |
//! | `the_inbox_names_every_dispatch_that_ended_since_the_last_drain` | the completion inbox | `Added<EffectOutcome>` names every effect that ended since the system last ran |
//! | `the_inbox_is_bounded_and_counts_what_it_dropped` | the inbox's bound and drop count | nothing is dropped: a system that skips passes still sees every outcome |
//!
//! The loom model `loom_reopen_races_the_last_in_flight_reply` has no
//! successor: the race it proved safe (a reply from a dying driver against
//! a reopen) does not exist when the driver is a system.

use crate::bus_support;

use std::sync::{Arc, atomic::Ordering};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{effect::HandlerKey, error::ErrorKind, serve::ErasedHandler};
use rig_ecs::{
    bus::{BusSet, EffectOutcome, InFlight, PendingEffect, RigSchedule},
    checkpoint::{RestoreMode, load_world},
};

#[test]
fn reopen_while_a_driver_is_alive_is_refused() {
    let (mut app, _, counters) = served();
    counters.hold.hold();
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
        app.update();
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
fn an_unfinished_effect_without_a_saved_contract_is_refused_before_loading() {
    let counters = Arc::new(Counters::default());
    let (mut live, _, _) = served();
    live.world_mut()
        .spawn(PendingEffect::new("later", completion()));
    let saved = checkpoint(&mut live);
    let mut app = app();
    let before = checkpoint(&mut app).to_json().unwrap();
    let error = load_world(
        &saved,
        app.world_mut(),
        RestoreMode::Strict,
        [(
            HandlerKey::from("model"),
            ErasedHandler::new(MockModel::new(&counters)),
        )],
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing saved handler `later`"));
    assert_eq!(checkpoint(&mut app).to_json().unwrap(), before);
    assert_eq!(counters.unary_started.load(Ordering::SeqCst), 0);
}

#[test]
fn a_rebind_of_the_wrong_family_panics_at_the_hosts_line() {
    let (mut app, _, _) = served();
    let mut live = bus_support::app();
    register(
        &mut live,
        "model",
        crate::run_support::NeverCalled {
            name: "add".to_owned(),
        },
    );
    let saved = checkpoint(&mut live);
    let before = app.world().entities().len();
    // No panic anywhere: the gap is data, refused at the host's line.
    let error = load_world(&saved, app.world_mut(), RestoreMode::Strict, [])
        .expect_err("the family differs");
    assert_eq!(error.kind, ErrorKind::Request);
    assert!(error.message.contains("model"), "{error:?}");
    assert_eq!(app.world().entities().len(), before, "nothing spawned");
}

#[derive(Resource, Default)]
struct Ended(Vec<Entity>);

fn see_what_ended(landed: Query<Entity, Added<EffectOutcome>>, mut ended: ResMut<Ended>) {
    ended.0.extend(landed.iter());
}

#[test]
fn the_inbox_names_every_dispatch_that_ended_since_the_last_drain() {
    let (mut app, _, _) = served();
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
