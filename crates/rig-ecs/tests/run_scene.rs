//! The graph is the state (prompt ruling 6): a run saved after its first
//! turn was answered resumes in a fresh world to the same second request
//! and the same answer — the two-record golden
//! `mock_output_tool_text_reprompt`, split at its first record.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod run_support;

use std::time::Instant;

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{effect::HandlerKey, serve::ServingPolicy};
use rig_ecs::{
    agent::{
        AdditionalParams, Assembling, Cursor, DefaultMaxTurns, InvalidCalls, MaxTokens, MaxTurns,
        Output, OutputKind, Owner, Preamble, RunResult, Settled, Temperature, ToolChoiceSpec,
        UsesModel, Utterance, scene::RunScene,
    },
    bus::{BusPlugin, EffectLogResource, Handlers, IdCounter, RigSchedule},
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use run_support::GUARD;

fn golden(name: &str) -> EffectLog {
    let path = format!(
        "{}/../rig-verify/fixtures/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("the golden is committed"))
        .expect("the golden loads")
}

fn world_with(log: &EffectLog) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((
        BusPlugin::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
    ));
    app.finish();
    app.cleanup();
    let key = HandlerKey::from("golden/model:default");
    let model = Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_erased(
                key.clone(),
                rig_core::serve::ErasedHandler::new(
                    EffectLogReplayer::for_key(log, &key).expect("the model's records"),
                ),
            )
            .expect("a fresh key")
    })
    .expect("a bus");
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    (app, model)
}

fn event_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "category": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["title", "category", "summary"]
    })
}

#[test]
fn a_run_saved_mid_turn_resumes_to_the_same_request_and_answer() {
    let log = golden("mock_output_tool_text_reprompt");
    assert_eq!(log.records.len(), 2);
    let (mut app, model) = world_with(&log);
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let agent = app
        .world_mut()
        .spawn((
            Owner("golden".to_owned()),
            Preamble(Some(
                "You are a concise assistant. Answer directly.".to_owned(),
            )),
            Temperature(None),
            MaxTokens(None),
            AdditionalParams(None),
            ToolChoiceSpec(None),
            Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema()),
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

    // One pass at a time, until the first turn was read and the run wants
    // its second: the reprompt is in the graph, the second request is not
    // yet folded.
    let start = Instant::now();
    loop {
        app.world_mut().run_schedule(RigSchedule);
        let world = app.world_mut();
        let wants_second = world.get::<Assembling>(run).is_some()
            && world
                .get::<Cursor>(run)
                .is_some_and(|cursor| cursor.turn == 1);
        if wants_second {
            break;
        }
        assert!(
            world.get::<Settled>(run).is_none(),
            "settled before the reprompt"
        );
        assert!(start.elapsed() < GUARD, "the first turn never came back");
        std::thread::yield_now();
    }
    let utterances = app
        .world_mut()
        .query::<&Utterance>()
        .iter(app.world())
        .count();
    assert_eq!(utterances, 3, "prompt, the text answer, the reprompt");
    let saved = RunScene::save(app.world_mut());
    let json = serde_json::to_string(&saved).expect("serde");
    assert!(!json.contains("Entity"), "no entity ids in a scene");
    let head = app.world().resource::<EffectLogResource>().log();
    assert_eq!(head.records.len(), 1, "the first record was recorded here");
    drop(app);

    // A fresh world over the log's tail: the graph loaded, the second turn
    // folded from it, answered by record 2, settled to the golden's answer.
    let saved: RunScene = serde_json::from_str(&json).expect("serde");
    let (mut app, _model) = world_with(&log.tail(1));
    app.world_mut().resource_mut::<IdCounter>().0 = 2;
    let loaded = saved.load(app.world_mut()).expect("the model is bound");
    let run = loaded
        .into_iter()
        .find(|entity| app.world().get::<rig_ecs::agent::Run>(*entity).is_some())
        .expect("the run");
    let start = Instant::now();
    loop {
        app.update();
        if app.world().get::<Settled>(run).is_some() {
            break;
        }
        assert!(
            app.world().get::<rig_ecs::agent::Failed>(run).is_none(),
            "{:?}",
            app.world().get::<rig_ecs::agent::Failed>(run)
        );
        assert!(start.elapsed() < GUARD, "the resumed run never settled");
        std::thread::yield_now();
    }
    let answer = app
        .world()
        .get::<RunResult>(run)
        .expect("settled")
        .0
        .clone();
    let expected = match &log.records[1].outcome {
        Ok(rig_core::effect::Outcome::Completion(response)) => response
            .choice
            .iter()
            .find_map(|part| match part {
                rig_core::message::AssistantContent::ToolCall(call) => {
                    Some(call.function.arguments.to_string())
                }
                rig_core::message::AssistantContent::Text(_)
                | rig_core::message::AssistantContent::Reasoning(_)
                | rig_core::message::AssistantContent::Image(_) => None,
            })
            .expect("the output tool's call"),
        other => panic!("a completion: {other:?}"),
    };
    assert_eq!(answer, expected, "the golden's answer");
    let tail = app.world().resource::<EffectLogResource>().log();
    assert_eq!(tail.records.len(), 1);
    let mine = &tail.records[0];
    let theirs = &log.records[1];
    assert_eq!(mine.id, theirs.id);
    assert_eq!(
        serde_json::to_value(&mine.kind).expect("serde"),
        serde_json::to_value(&theirs.kind).expect("serde"),
        "the second request, folded from the loaded graph, is the golden's"
    );
    assert_eq!(
        serde_json::to_value(&mine.outcome).expect("serde"),
        serde_json::to_value(&theirs.outcome).expect("serde")
    );
}
