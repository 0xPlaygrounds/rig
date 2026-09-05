//! The graph is the state (prompt ruling 6): a run saved after its first
//! turn was answered resumes in a fresh world to the same second request
//! and the same answer — the two-record golden
//! `mock_output_tool_text_reprompt`, split at its first record. And the
//! paired scene (stage 3 ruling 1a): a run saved with its model call in
//! flight resumes in a fresh world where the effect, `ChildOf` its turn
//! again, is re-issued under its saved id and answered there.

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
        UsesModel, Utterance,
        scene::{RunScene, WorldScene, load_world, save_world},
    },
    bus::{
        BusPlugin, EffectLogResource, Handlers, IdCounter, InFlight, Issued, PendingEffect, Replay,
        RigSchedule,
    },
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use run_support::{GUARD, NeverAnswers};

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
    let saved = RunScene::save(app.world_mut()).expect("every component serializes");
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
    let _run = spawn_run(
        app.world_mut(),
        agent,
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
    assert_eq!(saved.effects.effects.len(), 1);
    let effect = &saved.effects.effects[0];
    assert_eq!(
        effect.id,
        Some(log.records[0].id),
        "issued under the golden's id"
    );
    assert!(effect.outcome.is_none(), "in flight: intent, no answer");
    let turn_index = effect
        .parent_ref
        .expect("the effect names its turn in the graph");
    assert_eq!(
        saved.graph.entities[turn_index].kind,
        rig_ecs::agent::scene::SceneKind::Turn
    );
    let json = serde_json::to_string(&saved).expect("serde");
    assert!(!json.contains("Entity"), "no entity ids in a scene");
    drop(app);

    // A fresh world with the golden's by-id replayer: the effect is
    // `ChildOf` its turn again, re-issued under its saved id, answered
    // from the record, and the run settles on the golden's answer.
    let saved: WorldScene = serde_json::from_str(&json).expect("serde");
    let mut app = run_support::app();
    Handlers::with(app.world_mut(), |handlers| {
        Replay::default()
            .register(handlers, &log)
            .expect("the golden's replayers")
    })
    .expect("a bus");
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let loaded = load_world(&saved, app.world_mut()).expect("the model is bound");
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| app.world().get::<rig_ecs::agent::Run>(*entity).is_some())
        .expect("the run");
    let effect = loaded.effects[0];
    assert_eq!(
        app.world().get::<ChildOf>(effect).map(ChildOf::parent),
        Some(loaded.graph[turn_index]),
        "the effect is the turn's child again"
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

/// A run saved while its retrievals are out resumes: the scene carries
/// each retrieval effect's `Retrieval`, so the loaded turn attaches the
/// results and folds (the review's P1).
#[test]
fn a_run_saved_while_retrieving_resumes_and_attaches() {
    use rig_core::{
        effect::{FamilyDescriptor, HandlerDescriptor, RetrieveQuery, RetrievedDocuments},
        serve::{OutcomeSink, Serve},
    };
    use rig_ecs::agent::{Retrieval, RetrievalKind, Retrieves, Retrieving};

    /// An index that never answers in the first world and answers at once
    /// in the second.
    struct Index {
        answers: bool,
    }
    impl Serve for Index {
        type Family = rig_core::effect::family::Retrieve;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("t/retrieve:context#0"),
                family: FamilyDescriptor::Retrieve {},
                layers: Vec::new(),
            }
        }
        async fn serve(&self, kind: rig_core::effect::EffectKind, sink: OutcomeSink) {
            let rig_core::effect::EffectKind::Retrieve {
                query: RetrieveQuery::TopN { .. },
            } = kind
            else {
                return;
            };
            if !self.answers {
                std::future::pending::<()>().await;
            }
            sink.resolve(Ok(rig_core::effect::Outcome::Documents(
                RetrievedDocuments::Scored(vec![(0.9, "d1".to_owned(), serde_json::json!("a"))]),
            )))
            .await;
        }
    }
    fn world(answers: bool) -> (App, Entity, Entity) {
        let mut app = run_support::app();
        let (model, _) = run_support::Capturing::new("t/model:default", "ok");
        let model = run_support::register(&mut app, "t/model:default", model);
        let index = run_support::register(&mut app, "t/retrieve:context#0", Index { answers });
        (app, model, index)
    }

    let (mut app, model, index) = world(false);
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let agent = run_support::spawn_agent(app.world_mut(), "t", model);
    app.world_mut().spawn((
        Retrieves(index),
        Retrieval {
            samples: 1,
            what: RetrievalKind::Documents,
        },
        rig_ecs::agent::Order(0),
        ChildOf(agent),
    ));
    let _run = spawn_run(app.world_mut(), agent, &[], "what?", false, Some(1));
    run_support::tick_until(&mut app, "the retrieval out", |world| {
        world
            .query_filtered::<(), (
                With<rig_ecs::agent::Retrieval>,
                With<rig_ecs::bus::InFlight>,
            )>()
            .iter(world)
            .count()
            == 1
    });
    assert!(
        app.world_mut()
            .query_filtered::<(), With<Retrieving>>()
            .iter(app.world())
            .count()
            == 1
    );
    let saved = rig_ecs::agent::scene::save_world(app.world_mut()).expect("serializes");
    assert_eq!(
        saved.retrievals.len(),
        1,
        "the retrieval effect's marker is saved"
    );
    let json = serde_json::to_string(&saved).expect("serde");
    drop(app);

    let (mut app, _, _) = world(true);
    let saved: rig_ecs::agent::scene::WorldScene = serde_json::from_str(&json).expect("serde");
    let loaded = rig_ecs::agent::scene::load_world(&saved, app.world_mut()).expect("bound");
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| app.world().get::<rig_ecs::agent::Run>(*entity).is_some())
        .expect("the run");
    run_support::tick_until(&mut app, "the resumed run", |world| {
        world.get::<Settled>(run).is_some()
    });
    let attachments = app
        .world_mut()
        .query::<&rig_ecs::agent::Attachment>()
        .iter(app.world())
        .count();
    assert_eq!(attachments, 1, "the retrieved document is attached");
}

#[test]
fn contradictory_effect_ids_refuse_the_paired_graph_before_spawning() {
    use rig_core::effect::{EffectId, EffectKind};
    use rig_ecs::{
        agent::scene::{SceneEntity, SceneKind},
        bus::{Reserved, Scene},
    };
    let mut live = run_support::app();
    live.world_mut().spawn((
        PendingEffect::new(
            "custom",
            EffectKind::Custom {
                kind: "scene-preflight".into(),
                payload: serde_json::Value::Null,
            },
        ),
        Reserved(EffectId::from_raw(8)),
    ));
    let scene = Scene::save(live.world_mut());
    for (id, next_id) in [(8, Some(4)), (u64::MAX, None)] {
        let mut bad = scene.clone();
        bad.effects[0].id = Some(EffectId::from_raw(id));
        bad.next_id = next_id;
        let bad: Scene = serde_json::from_value(serde_json::to_value(bad).unwrap()).unwrap();
        let mut restored = run_support::app();
        restored.world_mut().resource_mut::<IdCounter>().0 = 5;
        let before = restored.world().entities().len();
        let world_scene = WorldScene {
            effects: bad,
            graph: RunScene {
                entities: vec![SceneEntity {
                    kind: SceneKind::Agent,
                    components: Default::default(),
                    parent: None,
                    relations: Vec::new(),
                }],
                ..RunScene::default()
            },
            ..WorldScene::default()
        };
        assert!(load_world(&world_scene, restored.world_mut()).is_err());
        assert_eq!(
            restored.world().entities().len(),
            before,
            "paired graph preflight must also refuse"
        );
        assert_eq!(restored.world().resource::<rig_ecs::bus::IdCounter>().0, 5);
    }
}
