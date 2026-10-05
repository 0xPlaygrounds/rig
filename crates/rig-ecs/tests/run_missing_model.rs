//! A removed model binding produces a terminal diagnostic before dispatch.

use crate::run_support;

use rig_core::{effect::HandlerKey, error::ErrorKind};
use rig_ecs::{
    agent::{Failed, Failure, RunPhase, Settled},
    bus::{Handlers, PendingEffect},
    systems::RunCommands,
};
use run_support::*;

#[test]
fn deregistered_model_before_first_dispatch_fails_without_issuing_an_effect() {
    let mut app = app();
    let (handler, requests) = Capturing::new("model", "ok");
    let model = register(&mut app, "model", handler);
    let agent = spawn_agent(app.world_mut(), "test", model);
    let run = app.world_mut().spawn_run(agent, &[], "go", false, None);
    Handlers::with(app.world_mut(), |handlers| {
        handlers.deregister(&HandlerKey::from("model"))
    })
    .unwrap();
    app.update();
    let failed = app
        .world()
        .get::<Failed>(run)
        .expect("a missing selected model is terminal");
    assert!(
        matches!(&failed.0, Failure::Provider(report) if report.kind == ErrorKind::HandlerUnavailable)
    );
    assert!(app.world().get::<Settled>(run).is_none());
    assert!(app.world().get::<RunPhase>(run).is_none());
    assert_eq!(
        app.world_mut()
            .query::<&PendingEffect>()
            .iter(app.world())
            .count(),
        0
    );
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn selected_model_with_a_non_completion_descriptor_fails_with_its_key() {
    let mut app = app();
    let tool = register(
        &mut app,
        "wrong-model",
        NeverCalled {
            name: "unused".into(),
        },
    );
    let agent = spawn_agent(app.world_mut(), "test", tool);
    let run = app.world_mut().spawn_run(agent, &[], "go", false, None);
    app.update();
    assert!(matches!(&app.world().get::<Failed>(run).unwrap().0,
        Failure::Provider(report) if report.message.contains("wrong-model")
    ));
    assert_eq!(
        app.world_mut()
            .query::<&PendingEffect>()
            .iter(app.world())
            .count(),
        0
    );
}
