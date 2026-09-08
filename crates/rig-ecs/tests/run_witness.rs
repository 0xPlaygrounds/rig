//! The agent runtime's facts reach the witness: endings, cancellation
//! requests, invalid-call resolutions and the tool batch's holds, each with
//! the run's scope as its subject.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod run_support;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use rig_core::observe::{Action, ObservationLog, Stage};
use rig_ecs::{
    agent::{Cancelled, Failed, Grant, Order, Settled, ToolPolicy},
    bus::{Scope, Witnessing},
    systems::spawn_run,
};
use run_support::*;

fn witnessed(app: &mut bevy_app::App) -> Arc<ObservationLog> {
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log.clone());
    log
}

fn endings(log: &ObservationLog) -> Vec<(Option<String>, String)> {
    log.trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Ended { ending } => Some((o.subject.scope.clone(), ending.code.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_settled_run_ends_in_the_trace_under_its_scope() {
    let mut app = app();
    let log = witnessed(&mut app);
    let (model, _) = Capturing::new("m", "fine");
    let model = register(&mut app, "m", model);
    let agent = spawn_agent(app.world_mut(), "app", model);
    let run = spawn_run(app.world_mut(), agent, &[], "hi", false, Some(1));
    tick_until(&mut app, "settled", |world| {
        world.get::<Settled>(run).is_some()
    });
    app.update();

    let scope = app.world().get::<Scope>(run).unwrap().0.clone();
    assert_eq!(endings(&log), [(Some(scope.clone()), "settled".to_owned())]);
    let trace = log.trace();
    let issued = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::Issued))
        .expect("the completion was issued");
    assert_eq!(issued.subject.scope.as_deref(), Some(scope.as_str()));
    assert!(
        trace
            .observations
            .iter()
            .any(|o| matches!(o.action, Action::Landed { .. }))
    );
}

#[test]
fn a_cancelled_run_requests_then_ends_and_leaves_its_flight_to_the_handler() {
    let mut app = app();
    let log = witnessed(&mut app);
    let model = register(
        &mut app,
        "never",
        NeverAnswers {
            label: "never".into(),
        },
    );
    let agent = spawn_agent(app.world_mut(), "app", model);
    let run = spawn_run(app.world_mut(), agent, &[], "hi", false, Some(1));
    tick_until(&mut app, "in flight", |world| {
        world
            .query_filtered::<(), With<rig_ecs::bus::InFlight>>()
            .iter(world)
            .count()
            == 1
    });
    app.world_mut()
        .entity_mut(run)
        .insert(Cancelled("operator stop".into()));
    tick_until(&mut app, "failed", |world| {
        world.get::<Failed>(run).is_some()
    });
    app.update();

    let trace = log.trace();
    let kinds: Vec<String> = trace
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::CancelRequested { reason } => {
                Some(format!("requested:{}", reason.detail.clone().unwrap()))
            }
            Action::Ended { ending } => Some(format!("ended:{}", ending.code)),
            _ => None,
        })
        .collect();
    assert_eq!(kinds, ["requested:operator stop", "ended:cancelled"]);
    // An issued effect is left to its handler: no in-flight cancellation
    // is observed, and the world still shows it in flight.
    assert!(
        !trace
            .observations
            .iter()
            .any(|o| matches!(o.action, Action::Cancelled { .. }))
    );
    assert_eq!(
        app.world_mut()
            .query_filtered::<(), With<rig_ecs::bus::InFlight>>()
            .iter(app.world())
            .count(),
        1
    );
    assert_eq!(
        trace
            .observations
            .iter()
            .find(|o| matches!(o.action, Action::CancelRequested { .. }))
            .unwrap()
            .stage,
        Stage::Runtime
    );
}

#[test]
fn an_invalid_tool_call_is_resolved_in_the_trace() {
    let mut app = app();
    let log = witnessed(&mut app);
    let (model, _) = Scripted::new(
        "s",
        vec![vec![call("c1", "no_such_tool", serde_json::json!({}))]],
    );
    let model = register(&mut app, "s", model);
    let agent = spawn_agent(app.world_mut(), "app", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, Some(2));
    tick_until(&mut app, "failed", |world| {
        world.get::<Failed>(run).is_some()
    });
    app.update();

    let trace = log.trace();
    let invalid = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::InvalidCall { .. }))
        .expect("the resolution is observed");
    let Action::InvalidCall { name, resolution } = &invalid.action else {
        panic!("matched above")
    };
    assert_eq!(name, "no_such_tool");
    assert_eq!(resolution.code, "fail");
    assert_eq!(
        endings(&log).last().map(|(_, code)| code.as_str()),
        Some("unknown_tool_call")
    );
}

#[test]
fn a_tool_batch_beyond_its_concurrency_is_held_then_released() {
    let mut app = app();
    let log = witnessed(&mut app);
    let (model, _) = Scripted::new(
        "s",
        vec![vec![
            call("c1", "add", serde_json::json!({"x": 1, "y": 2})),
            call("c2", "add", serde_json::json!({"x": 3, "y": 4})),
        ]],
    );
    let model = register(&mut app, "s", model);
    let adder = Adder::new("add");
    let tool = register(&mut app, "tool:add", adder);
    let agent = spawn_agent(app.world_mut(), "app", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(ToolPolicy { concurrency: 1 });
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    let run = spawn_run(app.world_mut(), agent, &[], "add things", false, Some(3));
    tick_until(&mut app, "settled", |world| {
        world.get::<Settled>(run).is_some()
    });
    app.update();

    let trace = log.trace();
    let held_then_released: Vec<&str> = trace
        .observations
        .iter()
        .filter(|o| o.subject.family == Some(rig_core::effect::EffectFamily::Tool))
        .filter_map(|o| match &o.action {
            Action::Held { .. } => Some("held"),
            Action::Released => Some("released"),
            Action::Issued => Some("issued"),
            _ => None,
        })
        .collect();
    assert_eq!(
        held_then_released,
        ["held", "issued", "released", "issued"],
        "{trace:?}"
    );
    assert_eq!(
        endings(&log).last().map(|(_, c)| c.as_str()),
        Some("settled")
    );
}
