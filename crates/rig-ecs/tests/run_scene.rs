//! The graph is the state (prompt ruling 6): a run saved after its first
//! turn was answered resumes in a fresh world to the same second request
//! and the same answer — the two-record golden
//! `mock_output_tool_text_reprompt`, split at its first record. And the
//! checkpoint of a run mid-flight (stage 3 ruling 1a): a run saved with its
//! model call in flight resumes in a fresh world where the effect,
//! `ChildOf` its turn again, is re-issued under its saved id and answered
//! there.

use crate::run_support;

use std::{any::type_name, time::Instant};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::Replay;
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use rig_core::{effect::HandlerKey, serve::ServingPolicy};
use rig_ecs::{
    agent::{
        AdditionalParams, Cursor, DefaultMaxTurns, InvalidCalls, MaxTokens, MaxTurns, Output,
        OutputKind, Owner, Preamble, Run, RunResult, Settled, Temperature, ToolChoiceSpec, Turn,
        UsesModel, Utterance,
    },
    bus::{
        EffectOutcome, Handlers, IdCounter, InFlight, Issued, PendingEffect, Reserved, RigSchedule,
    },
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::RunCommands,
};
use run_support::{GUARD, NeverAnswers};

fn golden(name: &str) -> EffectLog {
    let path = format!(
        "{}/../rig-cassette/fixtures/effects/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("the golden is committed"))
        .expect("the golden loads")
}

fn world_with(log: &EffectLog) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(
        rig_ecs::RigPlugin::with_policy(ServingPolicy::default())
            .ambiguity_detection(LogLevel::Error),
    );
    app.add_plugins(rig_cassette::ecs::ReplayPlugin);
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

/// The checkpoint entities carrying `C`, by index.
fn entities_with<C: Component>(checkpoint: &Checkpoint) -> Vec<usize> {
    checkpoint
        .entities
        .iter()
        .enumerate()
        .filter(|(_, entity)| entity.contains_key(type_name::<C>()))
        .map(|(index, _)| index)
        .collect()
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
    let run = app.world_mut().spawn_run(
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
        let wants_second = world.get::<rig_ecs::agent::RunPhase>(run)
            == Some(&rig_ecs::agent::RunPhase::Assembling)
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
    let saved = save_world(app.world_mut()).expect("every component serializes");
    let json = saved.to_json().expect("serde");
    let head = app.world().resource::<EffectLogResource>().log();
    assert_eq!(head.records.len(), 1, "the first record was recorded here");
    drop(app);

    // A fresh world over the log's tail: the graph loaded, the second turn
    // folded from it, answered by record 2, settled to the golden's answer.
    let saved = Checkpoint::from_json(&json).expect("serde");
    let (mut app, _model) = world_with(&log.tail(1));
    app.world_mut().resource_mut::<IdCounter>().0 = 2;
    let loaded =
        load_world(&saved, app.world_mut(), RestoreMode::Strict, []).expect("the model is bound");
    let run = loaded.with::<Run>(app.world())[0];
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
                    Some(call.function.arguments_value().to_string())
                }
                rig_core::message::AssistantContent::Text(_)
                | rig_core::message::AssistantContent::Reasoning(_)
                | rig_core::message::AssistantContent::Image(_)
                | rig_core::message::AssistantContent::Opaque(_) => None,
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

#[test]
fn a_run_saved_with_its_effect_in_flight_resumes_and_the_effect_is_answered_there() {
    let log = golden("anthropic_completion_smoke");
    assert_eq!(log.records.len(), 1);
    let key = HandlerKey::from("golden/model:default");

    // The first world: the model never answers, so the run's one effect is
    // in flight when the world is saved.
    let mut app = run_support::app();
    let model = run_support::register(
        &mut app,
        "golden/model:default",
        NeverAnswers {
            label: "golden/model:default".to_owned(),
        },
    );
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
            Output::default(),
            DefaultMaxTurns(None),
            MaxTurns(1),
            InvalidCalls::default(),
            UsesModel(model),
        ))
        .id();
    let _run = app.world_mut().spawn_run(agent,
        &[],
        "In one or two sentences, explain what Rust programming language is and why memory safety matters.",
        false,
        Some(1),
    );
    run_support::tick_until(&mut app, "the effect in flight", |world| {
        world
            .query_filtered::<Entity, (With<PendingEffect>, With<InFlight>)>()
            .iter(world)
            .next()
            .is_some()
    });
    let saved = save_world(app.world_mut()).expect("every component serializes");
    let effects = entities_with::<PendingEffect>(&saved);
    assert_eq!(effects.len(), 1);
    let effect = &saved.entities[effects[0]];
    assert_eq!(
        effect.get(type_name::<Issued>()),
        Some(&serde_json::json!(log.records[0].id)),
        "issued under the golden's id"
    );
    assert!(
        !effect.contains_key(type_name::<EffectOutcome>()),
        "in flight: intent, no answer"
    );
    let json = saved.to_json().expect("serde");
    drop(app);

    // A fresh world with the golden's by-id replayer: the effect is
    // `ChildOf` its turn again, re-issued under its saved id, answered
    // from the record, and the run settles on the golden's answer.
    let saved = Checkpoint::from_json(&json).expect("serde");
    let mut app = run_support::app();
    Replay::default()
        .register(app.world_mut(), &log)
        .expect("the golden's replayers");
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    // The saved descriptor is the never-answering model's; the golden's
    // replayer advertises the recorded model instead. Same family, a
    // deliberately different implementation, so the change is declared.
    let loaded =
        load_world(&saved, app.world_mut(), RestoreMode::Replace, []).expect("the model is bound");
    let run = loaded.with::<Run>(app.world())[0];
    let effect = loaded.with::<PendingEffect>(app.world())[0];
    let turn = app
        .world()
        .get::<ChildOf>(effect)
        .map(ChildOf::parent)
        .expect("the effect is the turn's child again");
    assert!(app.world().get::<Turn>(turn).is_some());
    assert_eq!(
        app.world().get::<ChildOf>(turn).map(ChildOf::parent),
        Some(run),
        "the turn is the run's"
    );
    assert_eq!(
        app.world()
            .get::<Reserved>(effect)
            .map(|reserved| reserved.0),
        Some(log.records[0].id),
        "re-issued under the saved id"
    );
    run_support::tick_until(&mut app, "the resumed run settles", |world| {
        assert!(
            world.get::<rig_ecs::agent::Failed>(run).is_none(),
            "{:?}",
            world.get::<rig_ecs::agent::Failed>(run)
        );
        world.get::<Settled>(run).is_some()
    });
    assert_eq!(
        app.world().get::<Issued>(effect).map(|issued| issued.0),
        Some(log.records[0].id),
        "re-issued under the saved id"
    );
    let answer = app
        .world()
        .get::<RunResult>(run)
        .expect("settled")
        .0
        .clone();
    assert!(
        answer.starts_with("Rust is a systems programming language"),
        "{answer}"
    );
    let replayed = app.world().resource::<EffectLogResource>().log();
    assert_eq!(replayed.records.len(), 1, "answered in this world");
    assert_eq!(replayed.records[0].id, log.records[0].id);
    assert_eq!(replayed.records[0].key, key);
    assert_eq!(
        serde_json::to_value(&replayed.records[0].outcome).expect("serde"),
        serde_json::to_value(&log.records[0].outcome).expect("serde")
    );
}
