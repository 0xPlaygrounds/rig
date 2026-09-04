//! The graph's wins, as tests (prompt ruling 8): what the request-as-graph
//! gives that a request builder cannot.
//!
//! | claim (CONTRACT §derivation / design §2) | test |
//! |---|---|
//! | chat history is entities: despawn one and the next request lacks it | `an_utterance_despawned_before_assemble_leaves_the_next_request` |
//! | documents are shared: one entity, two runs, both requests | `one_document_attached_to_two_runs_appears_in_both_requests` |
//! | tools are the handler entities: a grant link advertises, its removal un-advertises | `a_tool_granted_by_a_relationship_is_advertised_and_gone_after_removal` |
//! | the model is a relationship: swap it on the run and the next request's key follows | `uses_model_swapped_on_a_run_changes_the_next_requests_key` |
//! | the second slot: a `Patch` system rewrites the folded effect and the record holds the patch | `a_patch_system_rewrites_the_folded_request_and_the_record_holds_it` |
//! | the first slot: a system before `Assemble` rewrites an utterance and the request carries it | `a_system_before_assemble_rewrites_an_utterance` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    effect::EffectKind,
    message::{Message, UserContent},
};
use rig_ecs::{
    agent::{
        Context, DocumentId, DocumentText, Grant, MessageParts, Order, Parts, RunResult, Settled,
        UsesModel, Utterance,
    },
    bus::{EffectLogResource, PendingEffect, RigSchedule},
    systems::{RigSet, spawn_run},
};
use rig_effect_log::EffectLogRecorder;
use run_support::*;

fn add_before_assemble<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, system.before(RigSet::Assemble));
}

fn settle(app: &mut bevy_app::App, run: Entity, what: &str) -> String {
    tick_until(app, what, |world| world.get::<Settled>(run).is_some());
    app.world()
        .get::<RunResult>(run)
        .expect("settled with an answer")
        .0
        .clone()
}

#[derive(Resource, Default)]
struct Once(bool);

fn despawn_the_assistant_utterance(
    utterances: Query<(Entity, &Parts), With<Utterance>>,
    mut once: ResMut<Once>,
    mut commands: Commands,
) {
    if once.0 {
        return;
    }
    for (entity, Parts(parts)) in &utterances {
        if let MessageParts::Assistant { .. } = parts {
            commands.entity(entity).despawn();
            once.0 = true;
        }
    }
}

#[test]
fn an_utterance_despawned_before_assemble_leaves_the_next_request() {
    let mut app = app();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.init_resource::<Once>();
    add_before_assemble(&mut app, despawn_the_assistant_utterance);
    let history = vec![
        MessageParts::User {
            content: vec![UserContent::text("A")],
        },
        MessageParts::Assistant {
            id: None,
            content: vec![rig_core::message::AssistantContent::text("B")],
        },
    ];
    let run = spawn_run(app.world_mut(), agent, &history, "C", false, Some(1));
    settle(&mut app, run, "answered");
    let requests = requests.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        texts(&requests[0]),
        vec!["system:You are terse.", "user:A", "user:C"],
        "the assistant utterance left the history before the fold"
    );
}

#[test]
fn one_document_attached_to_two_runs_appears_in_both_requests() {
    let mut app = app();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let document = app
        .world_mut()
        .spawn((
            DocumentId("shared".to_owned()),
            DocumentText("one document, many turns".to_owned()),
        ))
        .id();
    app.world_mut()
        .spawn((Context(document), Order(0), ChildOf(agent)));
    let first = spawn_run(app.world_mut(), agent, &[], "first?", false, Some(1));
    let second = spawn_run(app.world_mut(), agent, &[], "second?", false, Some(1));
    settle(&mut app, first, "first");
    settle(&mut app, second, "second");
    let requests = requests.lock().expect("requests");
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(request.documents.len(), 1);
        assert_eq!(request.documents[0].id, "shared");
        assert_eq!(request.documents[0].text, "one document, many turns");
    }
    let documents = app
        .world_mut()
        .query::<&DocumentId>()
        .iter(app.world())
        .count();
    assert_eq!(documents, 1, "one entity, two requests");
}

#[test]
fn a_tool_granted_by_a_relationship_is_advertised_and_gone_after_removal() {
    let mut app = app();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    let tool = register(
        &mut app,
        "t/tool:add#0",
        NeverCalled {
            name: "add".to_owned(),
        },
    );
    let agent = spawn_agent(app.world_mut(), "t", model);
    let before = spawn_run(app.world_mut(), agent, &[], "one", false, Some(1));
    settle(&mut app, before, "before the grant");
    let grant = app
        .world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)))
        .id();
    let during = spawn_run(app.world_mut(), agent, &[], "two", false, Some(1));
    settle(&mut app, during, "with the grant");
    app.world_mut().despawn(grant);
    let after = spawn_run(app.world_mut(), agent, &[], "three", false, Some(1));
    settle(&mut app, after, "after the grant");
    let requests = requests.lock().expect("requests");
    let advertised: Vec<Vec<String>> = requests
        .iter()
        .map(|request| request.tools.iter().map(|tool| tool.name.clone()).collect())
        .collect();
    assert_eq!(
        advertised,
        vec![Vec::<String>::new(), vec!["add".to_owned()], Vec::new()]
    );
}

#[test]
fn uses_model_swapped_on_a_run_changes_the_next_requests_key() {
    let mut app = app();
    let (default_model, default_requests) = Capturing::new("t/model:default", "from default");
    let (fast_model, fast_requests) = Capturing::new("t/model:fast", "from fast");
    let default_model = register(&mut app, "t/model:default", default_model);
    let fast_model = register(&mut app, "t/model:fast", fast_model);
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let agent = spawn_agent(app.world_mut(), "t", default_model);
    let first = spawn_run(app.world_mut(), agent, &[], "one", false, Some(1));
    assert_eq!(settle(&mut app, first, "default"), "from default");
    // A routing system's spelling: the run's own model, before Assemble.
    let second = spawn_run(app.world_mut(), agent, &[], "two", false, Some(1));
    app.world_mut()
        .entity_mut(second)
        .insert(UsesModel(fast_model));
    assert_eq!(settle(&mut app, second, "fast"), "from fast");
    assert_eq!(default_requests.lock().expect("requests").len(), 1);
    assert_eq!(fast_requests.lock().expect("requests").len(), 1);
    let log = app.world().resource::<EffectLogResource>().log();
    let keys: Vec<String> = log
        .records
        .iter()
        .map(|record| record.key.to_string())
        .collect();
    assert_eq!(keys, vec!["t/model:default", "t/model:fast"]);
}

fn patch_temperature(mut effects: Query<&mut PendingEffect, Added<PendingEffect>>) {
    for mut effect in &mut effects {
        if let EffectKind::Completion { request, .. } = &mut effect.kind {
            request.temperature = Some(0.5);
        }
    }
}

#[test]
fn a_patch_system_rewrites_the_folded_request_and_the_record_holds_it() {
    let mut app = app();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, patch_temperature.in_set(RigSet::Patch));
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "one", false, Some(1));
    settle(&mut app, run, "patched");
    assert_eq!(
        requests.lock().expect("requests")[0].temperature,
        Some(0.5),
        "the handler saw the patch"
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let EffectKind::Completion { request, .. } = &log.records[0].kind else {
        panic!("a completion");
    };
    assert_eq!(
        request.temperature,
        Some(0.5),
        "the record is the patched request"
    );
    // And the graph is untouched: the agent's own temperature is unset.
    assert_eq!(
        app.world()
            .get::<rig_ecs::agent::Temperature>(agent)
            .expect("a temperature")
            .0,
        None
    );
}

fn shout_the_prompt(mut utterances: Query<&mut Parts, With<Utterance>>) {
    for mut parts in &mut utterances {
        if let MessageParts::User { content } = &mut parts.0 {
            for part in content.iter_mut() {
                if let UserContent::Text(text) = part
                    && !text.text.ends_with('!')
                {
                    text.text.push('!');
                }
            }
        }
    }
}

#[test]
fn a_system_before_assemble_rewrites_an_utterance() {
    let mut app = app();
    let (model, requests) = Capturing::new("t/model:default", "ok");
    let model = register(&mut app, "t/model:default", model);
    add_before_assemble(&mut app, shout_the_prompt);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "quietly", false, Some(1));
    settle(&mut app, run, "shouted");
    let requests = requests.lock().expect("requests");
    assert_eq!(
        texts(&requests[0]),
        vec!["system:You are terse.", "user:quietly!"]
    );
    let Message::User { content } = &requests[0].chat_history[1] else {
        panic!("the prompt");
    };
    assert_eq!(content.len(), 1);
}
