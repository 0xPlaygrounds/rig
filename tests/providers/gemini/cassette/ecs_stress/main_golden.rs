//! Exact named producer setup for the Gemini effect golden; real provider IO.
use crate::ecs_agent::RuntimeHandler;
use bevy_app::App;
use bevy_ecs::prelude::*;
use rig::{
    completion::CompletionModel,
    serve::adapters::{CompletionAdapter, ToolAdapter},
    tool::Tool,
};
use rig_ecs::{
    agent::{
        DefaultMaxTurns, Failed, Grant, MaxTurns, Order, Owner, Preamble, RunResult, Settled,
        Temperature, UsesModel,
    },
    bus::{BusPlugin, Handlers, Recording},
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use std::{sync::Arc, time::Duration};

fn tool<T: Tool + 'static>(app: &mut App, agent: Entity, tool: T, order: u64) {
    let handler = Handlers::with(app.world_mut(), |handlers| {
        handlers.register(
            format!("stress-agent/tool:{}#{order}", T::NAME),
            RuntimeHandler {
                inner: Arc::new(ToolAdapter::new(tool)),
                runtime: tokio::runtime::Handle::current(),
            },
        )
    })
    .expect("bus installed")
    .expect("unique named tool");
    app.world_mut()
        .spawn((Grant(handler), Order(order), ChildOf(agent)));
}
pub(super) async fn run(
    model: impl CompletionModel + 'static,
    preamble: &str,
    prompt: &str,
    add: impl Tool + 'static,
    subtract: impl Tool + 'static,
) -> (bool, EffectLog) {
    let mut app = App::new();
    app.add_plugins((BusPlugin::default(), AgentPlugin::default()));
    app.finish();
    app.cleanup();
    // Original record_effects() does not retain stream events.
    let recorder = EffectLogRecorder::new();
    Recording::install(app.world_mut(), recorder.clone());
    let model = Handlers::with(app.world_mut(), |handlers| {
        handlers.register(
            "stress-agent/model:default",
            RuntimeHandler {
                inner: Arc::new(CompletionAdapter::new("default", model)),
                runtime: tokio::runtime::Handle::current(),
            },
        )
    })
    .expect("bus installed")
    .expect("unique named model");
    let agent = app
        .world_mut()
        .spawn((
            Owner("stress-agent".into()),
            Preamble(Some(preamble.into())),
            DefaultMaxTurns(None),
            MaxTurns(1),
            Temperature(Some(0.0)),
            UsesModel(model),
        ))
        .id();
    tool(&mut app, agent, add, 0);
    tool(&mut app, agent, subtract, 1);
    let run = spawn_run(app.world_mut(), agent, &[], prompt, true, Some(6));
    let bus = app.world().resource::<rig_ecs::bus::Policy>().0;
    rig_ecs::replay::stamp_header(app.world_mut(), agent, &recorder, Some(bus), vec![]);
    rig_ecs::replay::stamp_run(app.world_mut(), run, &recorder);
    let saw_final = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            app.update();
            assert!(
                app.world().get::<Failed>(run).is_none(),
                "native golden run failed: {:?}",
                app.world().get::<Failed>(run)
            );
            if app.world().get::<Settled>(run).is_some() {
                // The original collector ignores item errors but requires a final.
                // Do not silently substitute the stricter shared success collector.
                return app.world().get::<RunResult>(run).is_some();
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("native golden run exceeded deadline");
    (saw_final, recorder.log())
}
