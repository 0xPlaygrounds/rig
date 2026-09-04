//! The agent on `wasm32-unknown-unknown`, executed: two goldens of the
//! completion-only corpus — the smoke and the two-record text reprompt —
//! replayed through the run graph on the browser target, driven by hand
//! as `bus_wasm` is. Every other wasm claim about the agent modules is
//! `cargo check`.

#![cfg(target_arch = "wasm32")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{effect::HandlerKey, serve::ServingPolicy};
use rig_ecs::{
    agent::{
        AdditionalParams, DefaultMaxTurns, Failed, InvalidCalls, MaxTokens, MaxTurns, Output,
        OutputKind, Owner, Preamble, RunResult, Settled, Temperature, ToolChoiceSpec, UsesModel,
    },
    bus::{BusPlugin, EffectLogResource, Handlers, IdCounter, run_to_quiescence},
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use wasm_bindgen_test::wasm_bindgen_test;

const SMOKE: &str = include_str!("../../rig-verify/fixtures/anthropic_completion_smoke.effects.json");
const REPROMPT: &str =
    include_str!("../../rig-verify/fixtures/mock_output_tool_text_reprompt.effects.json");

fn app(log: &EffectLog) -> (App, Entity) {
    bevy_tasks::ComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let mut app = App::new();
    app.add_plugins((
        BusPlugin::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
    ));
    app.finish();
    app.cleanup();
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let key = HandlerKey::from("golden/model:default");
    let model = Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_erased(
                key.clone(),
                rig_core::serve::ErasedHandler::new(
                    EffectLogReplayer::for_key(log, &key).expect("records"),
                ),
            )
            .expect("a fresh key")
    })
    .expect("a bus");
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    (app, model)
}

async fn tick(app: &mut App) {
    run_to_quiescence(app.world_mut());
    bevy_tasks::futures_lite::future::yield_now().await;
    bevy_tasks::tick_global_task_pools_on_main_thread();
}

async fn settle(app: &mut App, run: Entity) -> String {
    for _ in 0..500 {
        tick(app).await;
        if let Some(result) = app.world().get::<RunResult>(run) {
            assert!(app.world().get::<Settled>(run).is_some());
            return result.0.clone();
        }
        assert!(
            app.world().get::<Failed>(run).is_none(),
            "{:?}",
            app.world().get::<Failed>(run)
        );
    }
    panic!("the run never settled");
}

fn same_records(replayed: &EffectLog, log: &EffectLog) {
    assert_eq!(replayed.records.len(), log.records.len());
    for (mine, theirs) in replayed.records.iter().zip(&log.records) {
        assert_eq!(mine.id, theirs.id);
        assert_eq!(mine.key, theirs.key);
        assert_eq!(
            serde_json::to_value(&mine.kind).expect("serde"),
            serde_json::to_value(&theirs.kind).expect("serde")
        );
        assert_eq!(
            serde_json::to_value(&mine.outcome).expect("serde"),
            serde_json::to_value(&theirs.outcome).expect("serde")
        );
    }
}

#[wasm_bindgen_test]
async fn the_smoke_golden_replays_through_the_graph_on_wasm() {
    let log: EffectLog = serde_json::from_str(SMOKE).expect("the golden loads");
    let (mut app, model) = app(&log);
    let agent = app
        .world_mut()
        .spawn((
            Owner("golden".to_owned()),
            Preamble(Some("You are a concise assistant. Answer directly.".to_owned())),
            Temperature(None),
            MaxTokens(None),
            AdditionalParams(None),
            ToolChoiceSpec(None),
            Output::default(),
            DefaultMaxTurns(None),
            MaxTurns(1),
            InvalidCalls::default(),
            UsesModel(model),
        ))
        .id();
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        "In one or two sentences, explain what Rust programming language is and why memory safety matters.",
        false,
        Some(1),
    );
    let answer = settle(&mut app, run).await;
    assert!(answer.starts_with("Rust is a systems programming language"));
    same_records(&app.world().resource::<EffectLogResource>().log(), &log);
}

#[wasm_bindgen_test]
async fn the_text_reprompt_golden_replays_through_the_graph_on_wasm() {
    let log: EffectLog = serde_json::from_str(REPROMPT).expect("the golden loads");
    let (mut app, model) = app(&log);
    let agent = app
        .world_mut()
        .spawn((
            Owner("golden".to_owned()),
            Preamble(Some("You are a concise assistant. Answer directly.".to_owned())),
            Temperature(None),
            MaxTokens(None),
            AdditionalParams(None),
            ToolChoiceSpec(None),
            Output {
                mode: OutputKind::Tool,
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "category": {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "required": ["title", "category", "summary"]
                })),
            },
            DefaultMaxTurns(None),
            MaxTurns(3),
            InvalidCalls::default(),
            UsesModel(model),
        ))
        .id();
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        "Return a concise event object for a local Rust meetup in Seattle.",
        false,
        Some(3),
    );
    let answer = settle(&mut app, run).await;
    assert!(answer.contains("Seattle Rust Meetup"));
    same_records(&app.world().resource::<EffectLogResource>().log(), &log);
}
