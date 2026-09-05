//! A removed model binding produces a terminal diagnostic before dispatch.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

mod run_support;

struct ParkedAdder {
    inner: Adder,
    gate: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    started: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl rig_core::serve::Serve for ParkedAdder {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        self.inner.descriptor()
    }
    async fn serve(&self, kind: rig_core::effect::EffectKind, sink: rig_core::serve::OutcomeSink) {
        self.started
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let gate = self.gate.lock().unwrap().take();
        if let Some(gate) = gate {
            gate.await.unwrap();
        }
        self.inner.serve(kind, sink).await;
    }
}

use rig_core::{effect::HandlerKey, error::ErrorKind};
use rig_ecs::{
    agent::{Assembling, Failed, Failure, Settled},
    bus::{Bound, Handlers, PendingEffect},
    systems::spawn_run,
};
use run_support::*;

#[test]
fn deregistered_model_before_first_dispatch_fails_without_issuing_an_effect() {
    let mut app = app();
    let (handler, requests) = Capturing::new("model", "ok");
    let model = register(&mut app, "model", handler);
    let agent = spawn_agent(app.world_mut(), "test", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
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
    assert!(app.world().get::<Assembling>(run).is_none());
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
fn selected_model_without_a_bound_descriptor_fails() {
    let mut app = app();
    let (handler, requests) = Capturing::new("model", "ok");
    let model = register(&mut app, "model", handler);
    let agent = spawn_agent(app.world_mut(), "test", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    app.world_mut().entity_mut(model).remove::<Bound>();
    app.update();
    assert!(app.world().get::<Failed>(run).is_some());
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
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
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

#[test]
fn deregistered_model_between_turns_fails_after_the_outstanding_tool_finishes() {
    use bevy_ecs::prelude::ChildOf;
    use rig_ecs::agent::{Grant, MaxTurns, Order};
    use std::sync::{Arc, Mutex, atomic::Ordering};
    let mut app = app();
    let (handler, requests) = Scripted::new(
        "model",
        vec![vec![call(
            "add-1",
            "add",
            serde_json::json!({"x": 1, "y": 2}),
        )]],
    );
    let model = register(&mut app, "model", handler);
    let (release, gate) = futures::channel::oneshot::channel();
    let adder = Adder::new("tool:add");
    let in_flight = adder.in_flight.clone();
    let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let tool = register(
        &mut app,
        "tool:add",
        ParkedAdder {
            inner: adder,
            gate: Mutex::new(Some(gate)),
            started: started.clone(),
        },
    );
    let agent = spawn_agent(app.world_mut(), "test", model);
    app.world_mut().entity_mut(agent).insert(MaxTurns(2));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    tick_until(&mut app, "tool parked", |_| started.load(Ordering::SeqCst));
    Handlers::with(app.world_mut(), |handlers| {
        handlers.deregister(&HandlerKey::from("model"))
    })
    .unwrap();
    app.update();
    assert!(
        app.world().get::<Failed>(run).is_none(),
        "the current tool may finish"
    );
    release.send(()).unwrap();
    tick_until(&mut app, "next model unavailable", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert!(
        matches!(&app.world().get::<Failed>(run).unwrap().0, Failure::Provider(report) if report.kind == ErrorKind::HandlerUnavailable)
    );
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(in_flight.load(Ordering::SeqCst), 0);
}
