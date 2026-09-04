//! Tools are effect entities; the batch is the turn's children (CONTRACT
//! §8). What the shape gives, as tests:
//!
//! | claim | test |
//! |---|---|
//! | two calls in one turn are two children, dispatched in call order, and one user utterance of results in that order | `a_turn_with_two_calls_is_a_batch_and_the_results_are_one_utterance` |
//! | `ToolPolicy { concurrency }` on the agent lets two calls fly at once; the default holds the second until the first lands | `tool_policy_sets_how_many_calls_are_in_flight` |
//! | a `Judge` system replaces a tool child's outcome: history holds the replacement, the record the answer | `a_judge_system_replaces_a_tool_result_and_the_record_keeps_the_answer` |
//! | a `Gate` denial is a skipped result the model sees, and no record | `a_gate_denial_is_a_skipped_result_and_no_record` |
//! | a tool child despawned fails the run `Cancelled` | `despawning_a_tool_child_fails_the_run_cancelled` |
//! | `Resolution::Repair` written by a system renames the call and dispatches it | `a_system_repairs_an_invalid_call_to_a_granted_tool` |
//! | `Resolution::Retry` retries the turn with feedback and the invalid-peer notice | `a_system_retries_an_invalid_call_with_feedback` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod run_support;

use std::sync::{Arc, atomic::Ordering};

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message, UserContent},
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{
        Failed, Failure, Grant, InvalidCall, InvalidCalls, Order, Resolution, RunResult, Settled,
        ToolPolicy,
    },
    bus::{BusSet, EffectLogResource, EffectOutcome, Issued, PendingEffect, RigSchedule},
    systems::{RigSet, spawn_run},
};
use rig_effect_log::EffectLogRecorder;
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

fn add_system<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, system);
}

/// An app with a scripted model and the adder granted to one agent.
fn tooling(turns: Vec<Vec<AssistantContent>>) -> (bevy_app::App, Entity, Arc<Adder>, RequestsSeen) {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, requests) = Scripted::new(MODEL, turns);
    let model = register(&mut app, MODEL, model);
    let adder = Arc::new(Adder::new(ADD));
    let tool = register(&mut app, ADD, Arc::clone(&adder));
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(4));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    (app, agent, adder, requests)
}

type RequestsSeen = Arc<std::sync::Mutex<Vec<rig_core::completion::CompletionRequest>>>;

fn two_calls_then_text() -> Vec<Vec<AssistantContent>> {
    vec![
        vec![
            call("c1", "add", serde_json::json!({"x": 1, "y": 2})),
            call("c2", "add", serde_json::json!({"x": 3, "y": 4})),
        ],
        vec![AssistantContent::text("3 and 7")],
    ]
}

fn ended(app: &mut bevy_app::App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
}

fn tool_results(request: &rig_core::completion::CompletionRequest) -> Vec<(String, String)> {
    request
        .chat_history
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::ToolResult(result) => Some((
                        result.call.to_string(),
                        result
                            .content
                            .iter()
                            .map(|c| match c {
                                rig_core::message::ToolResultContent::Text(text) => {
                                    text.text.clone()
                                }
                                rig_core::message::ToolResultContent::Json { value, .. } => {
                                    value.to_string()
                                }
                                rig_core::message::ToolResultContent::Image(_) => {
                                    "<image>".to_owned()
                                }
                            })
                            .collect::<String>(),
                    )),
                    UserContent::Text(_)
                    | UserContent::Image(_)
                    | UserContent::Audio(_)
                    | UserContent::Video(_)
                    | UserContent::Document(_) => None,
                })
                .collect::<Vec<_>>(),
            Message::System { .. } | Message::Assistant { .. } => Vec::new(),
        })
        .collect()
}

#[test]
fn a_turn_with_two_calls_is_a_batch_and_the_results_are_one_utterance() {
    let (mut app, agent, adder, requests) = tooling(two_calls_then_text());
    let run = spawn_run(app.world_mut(), agent, &[], "add twice", false, None);
    ended(&mut app, run, "answered");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("3 and 7")
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let keys: Vec<&str> = log.records.iter().map(|r| r.key.as_str()).collect();
    assert_eq!(
        keys,
        [MODEL, ADD, ADD, MODEL],
        "the batch between the two completions"
    );
    let args: Vec<&str> = log
        .records
        .iter()
        .filter_map(|r| match &r.kind {
            EffectKind::ToolCall { args, .. } => Some(args.as_str()),
            EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Custom { .. } => None,
        })
        .collect();
    assert_eq!(args, [r#"{"x":1,"y":2}"#, r#"{"x":3,"y":4}"#], "call order");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        tool_results(&requests[1]),
        [
            ("c1".to_owned(), "3".to_owned()),
            ("c2".to_owned(), "7".to_owned())
        ],
        "one user utterance, the results in call order"
    );
    assert_eq!(adder.peak.load(Ordering::SeqCst), 1, "serial by default");
}

#[test]
fn tool_policy_sets_how_many_calls_are_in_flight() {
    let (mut app, agent, adder, _) = tooling(two_calls_then_text());
    app.world_mut()
        .entity_mut(agent)
        .insert(ToolPolicy { concurrency: 2 });
    let run = spawn_run(app.world_mut(), agent, &[], "add twice", false, None);
    ended(&mut app, run, "answered");
    assert!(app.world().get::<Settled>(run).is_some());
    assert_eq!(
        adder.peak.load(Ordering::SeqCst),
        2,
        "both calls in flight at once"
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let keys: Vec<&str> = log.records.iter().map(|r| r.key.as_str()).collect();
    assert_eq!(
        keys,
        [MODEL, ADD, ADD, MODEL],
        "the trace does not change with the policy"
    );
}

fn replace_tool_results(
    mut landed: Query<(&PendingEffect, &mut EffectOutcome), Added<EffectOutcome>>,
) {
    for (effect, mut outcome) in &mut landed {
        if let EffectKind::ToolCall { .. } = effect.kind {
            outcome.0 = Ok(Outcome::ToolResult {
                result: ToolResult::success(ToolOutput::text("99")),
            });
        }
    }
}

#[test]
fn a_judge_system_replaces_a_tool_result_and_the_record_keeps_the_answer() {
    let (mut app, agent, _, requests) = tooling(vec![
        vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
        vec![AssistantContent::text("99")],
    ]);
    add_system(&mut app, replace_tool_results.in_set(BusSet::Judge));
    let run = spawn_run(app.world_mut(), agent, &[], "add", false, None);
    ended(&mut app, run, "answered");
    let requests = requests.lock().unwrap();
    assert_eq!(
        tool_results(&requests[1]),
        [("c1".to_owned(), "99".to_owned())]
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let Ok(Outcome::ToolResult { result }) = &log.records[1].outcome else {
        panic!("the tool's record");
    };
    assert_eq!(
        result.output().render(),
        "3",
        "the record keeps the handler's answer"
    );
}

fn deny_tool_calls(
    fresh: Query<(Entity, &PendingEffect), (Without<Issued>, Without<EffectOutcome>)>,
    mut commands: Commands,
) {
    for (entity, effect) in &fresh {
        if let EffectKind::ToolCall { .. } = effect.kind {
            commands
                .entity(entity)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Denied,
                    "not today",
                ))));
        }
    }
}

#[test]
fn a_gate_denial_is_a_skipped_result_and_no_record() {
    let (mut app, agent, adder, requests) = tooling(vec![
        vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
        vec![AssistantContent::text("I could not add them.")],
    ]);
    add_system(&mut app, deny_tool_calls.in_set(BusSet::Gate));
    let run = spawn_run(app.world_mut(), agent, &[], "add", false, None);
    ended(&mut app, run, "answered");
    let requests = requests.lock().unwrap();
    assert_eq!(
        tool_results(&requests[1]),
        [("c1".to_owned(), "not today".to_owned())]
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let keys: Vec<&str> = log.records.iter().map(|r| r.key.as_str()).collect();
    assert_eq!(keys, [MODEL, MODEL], "a denial is no record");
    assert_eq!(adder.peak.load(Ordering::SeqCst), 0, "the tool never ran");
}

fn despawn_tool_calls(
    fresh: Query<(Entity, &PendingEffect), Without<Issued>>,
    mut commands: Commands,
) {
    for (entity, effect) in &fresh {
        if let EffectKind::ToolCall { .. } = effect.kind {
            commands.entity(entity).despawn();
        }
    }
}

#[test]
fn despawning_a_tool_child_fails_the_run_cancelled() {
    let (mut app, agent, _, _) = tooling(two_calls_then_text());
    add_system(&mut app, despawn_tool_calls.in_set(BusSet::Gate));
    let run = spawn_run(app.world_mut(), agent, &[], "add", false, None);
    ended(&mut app, run, "cancelled");
    assert!(matches!(
        app.world().get::<Failed>(run),
        Some(Failed(Failure::Cancelled(report))) if report.kind == ErrorKind::Cancelled
    ));
}

fn repair_to_add(
    invalid: Query<Entity, (With<InvalidCall>, Without<Resolution>)>,
    mut commands: Commands,
) {
    for entity in &invalid {
        commands.entity(entity).insert(Resolution::Repair {
            to: "add".to_owned(),
        });
    }
}

#[test]
fn a_system_repairs_an_invalid_call_to_a_granted_tool() {
    let (mut app, agent, _, requests) = tooling(vec![
        vec![call("c1", "multiply", serde_json::json!({"x": 2, "y": 3}))],
        vec![AssistantContent::text("5")],
    ]);
    add_system(&mut app, repair_to_add.before(RigSet::Materialise));
    let run = spawn_run(app.world_mut(), agent, &[], "multiply", false, None);
    ended(&mut app, run, "answered");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("5")
    );
    let log = app.world().resource::<EffectLogResource>().log();
    assert!(matches!(&log.records[1].kind, EffectKind::ToolCall { name, .. } if name == "add"));
    let requests = requests.lock().unwrap();
    let Message::Assistant { content, .. } = &requests[1].chat_history[2] else {
        panic!("the assistant turn");
    };
    assert!(
        matches!(&content[0], AssistantContent::ToolCall(call) if call.function.name == "add"),
        "history carries the repaired name"
    );
    assert_eq!(
        tool_results(&requests[1]),
        [("c1".to_owned(), "5".to_owned())]
    );
}

fn retry_with_feedback(
    invalid: Query<(Entity, &InvalidCall), Without<Resolution>>,
    mut commands: Commands,
) {
    for (entity, call) in &invalid {
        commands.entity(entity).insert(Resolution::Retry {
            feedback: format!("there is no tool named {}; use add", call.name),
        });
    }
}

#[test]
fn a_system_retries_an_invalid_call_with_feedback() {
    let (mut app, agent, _, requests) = tooling(vec![
        vec![
            call("c1", "multiply", serde_json::json!({"x": 2, "y": 3})),
            call("c2", "add", serde_json::json!({"x": 2, "y": 3})),
        ],
        vec![call("c3", "add", serde_json::json!({"x": 2, "y": 3}))],
        vec![AssistantContent::text("5")],
    ]);
    app.world_mut().entity_mut(agent).insert(InvalidCalls {
        retries: 1,
        unhandled: rig_ecs::agent::Unhandled::Fail,
    });
    add_system(&mut app, retry_with_feedback.before(RigSet::Materialise));
    let run = spawn_run(app.world_mut(), agent, &[], "multiply", false, None);
    ended(&mut app, run, "answered");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("5")
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let keys: Vec<&str> = log.records.iter().map(|r| r.key.as_str()).collect();
    assert_eq!(
        keys,
        [MODEL, MODEL, ADD, MODEL],
        "nothing dispatched for the retried turn"
    );
    let requests = requests.lock().unwrap();
    assert_eq!(
        tool_results(&requests[1]),
        [
            (
                "c1".to_owned(),
                "there is no tool named multiply; use add".to_owned()
            ),
            (
                "c2".to_owned(),
                rig_ecs::policy::text::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_owned()
            ),
        ],
        "the feedback for the invalid call, the notice for its peer"
    );
    let _ = HandlerKey::from(ADD);
}
