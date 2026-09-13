//! What a run leaves in the world, and what the bus takes per tick.
//!
//! | claim | test |
//! |---|---|
//! | a zero intake bound is a bound of one: the world makes progress, it never sits | `a_zero_command_capacity_still_takes_an_effect_per_tick` |
//! | `despawn_run` takes an ended run and its whole graph out of the world; it refuses a run that has not ended | `despawn_run_removes_an_ended_run_and_refuses_a_live_one` |
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_app::{App, Update};
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::serve::ServingPolicy;
use rig_ecs::{
    agent::{Runs, Settled},
    bus::{Bus, Policy, run_to_quiescence},
    systems::{RunBusy, despawn_run, install_agent, spawn_run},
};
use run_support::*;

fn app_with_capacity(command_capacity: usize) -> App {
    let mut app = App::new();
    Bus::with_policy(ServingPolicy {
        command_capacity,
        ..ServingPolicy::default()
    })
    .ambiguity_detection(LogLevel::Error)
    .install(app.world_mut());
    install_agent(app.world_mut());
    app.add_systems(Update, run_to_quiescence);
    app.finish();
    app.cleanup();
    app
}

#[test]
fn a_zero_command_capacity_still_takes_an_effect_per_tick() {
    let mut app = app_with_capacity(0);
    assert_eq!(
        app.world().resource::<Policy>().0.command_capacity,
        1,
        "the bus clamps the intake bound like rig-agent's driver does"
    );
    let (model, _) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "hi", false, None);
    tick_until(&mut app, "the run settles under a zero bound", |world| {
        world.get::<Settled>(run).is_some()
    });
}

#[test]
fn despawn_run_removes_an_ended_run_and_refuses_a_live_one() {
    let mut app = app_with_capacity(16);
    let (model, _) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    // Live entities, not allocation slots: a despawned slot stays counted
    // by `Entities::len` until reused.
    let live = |app: &mut App| app.world_mut().query::<Entity>().iter(app.world()).count();
    let before = live(&mut app);

    let run = spawn_run(app.world_mut(), agent, &[], "hi", false, None);
    assert_eq!(
        despawn_run(app.world_mut(), run),
        Err(RunBusy::Unsettled),
        "a run that has not ended stays"
    );
    assert_eq!(
        despawn_run(app.world_mut(), agent),
        Err(RunBusy::NotARun),
        "only a run is despawned this way"
    );
    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    assert!(
        live(&mut app) > before,
        "a settled run left its graph in the world"
    );
    assert!(
        app.world()
            .get::<Runs>(agent)
            .is_some_and(|runs| runs.iter().any(|r| r == run))
    );

    despawn_run(app.world_mut(), run).expect("an ended run despawns");
    app.update();
    assert_eq!(
        live(&mut app),
        before,
        "the run, its turns, utterances and effects are gone"
    );
    assert!(
        app.world()
            .get::<Runs>(agent)
            .is_none_or(|runs| !runs.iter().any(|r| r == run)),
        "the agent no longer lists the run"
    );
}

/// A run cancelled while its model call is in flight has ended, but the
/// call is still the handler's: `despawn_run` refuses it until the effect
/// settles, so a host cancels, lets the run drain, then despawns.
#[test]
fn despawn_run_refuses_a_run_whose_effect_is_still_in_flight() {
    let mut app = app_with_capacity(16);
    let model = register(
        &mut app,
        "t/model:never",
        NeverAnswers {
            label: "never".into(),
        },
    );
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "hi", false, None);
    tick_until(&mut app, "the model call is in flight", |world| {
        world
            .query_filtered::<Entity, With<rig_ecs::bus::InFlight>>()
            .iter(world)
            .next()
            .is_some()
    });
    app.world_mut()
        .entity_mut(run)
        .insert(rig_ecs::agent::Cancelled("host stop".into()));
    tick_until(&mut app, "the run fails cancelled", |world| {
        world.get::<rig_ecs::agent::Failed>(run).is_some()
    });
    assert_eq!(
        despawn_run(app.world_mut(), run),
        Err(RunBusy::InFlight),
        "the never-answering call is still in flight"
    );
}
