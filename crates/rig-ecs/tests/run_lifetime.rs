//! What a run leaves in the world, and what the bus takes per tick.
//!
//! | claim | test |
//! |---|---|
//! | a zero intake bound is a bound of one: the world makes progress, it never sits | `a_zero_command_capacity_still_takes_an_effect_per_tick` |
//! | `despawn_run` takes an ended run and its whole graph out of the world; it refuses a run that has not ended | `despawn_run_removes_an_ended_run_and_refuses_a_live_one` |
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

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
