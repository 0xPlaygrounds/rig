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
                        result
                            .call
                            .explicit()
                            .expect("explicit provider test ID")
                            .to_owned(),
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

#[derive(Resource)]
struct ActiveTools(Vec<String>);

fn narrow_tools(
    fresh: Query<Entity, With<rig_ecs::systems::Fresh>>,
    allowed: Res<ActiveTools>,
    mut commands: Commands,
) {
    for turn in &fresh {
        commands.entity(turn).insert(rig_ecs::agent::RequestPatch {
            active_tools: Some(allowed.0.clone()),
            ..Default::default()
        });
    }
}

fn assert_active_tools(allowed: &[&str], executable: bool) {
    let (mut app, agent, adder, requests) = tooling(vec![
        vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
        vec![AssistantContent::text("done")],
    ]);
    let other = register(
        &mut app,
        "other",
        NeverCalled {
            name: "other".into(),
        },
    );
    app.world_mut()
        .spawn((Grant(other), Order(1), ChildOf(agent)));
    app.insert_resource(ActiveTools(
        allowed.iter().map(|name| (*name).to_owned()).collect(),
    ));
    add_system(
        &mut app,
        narrow_tools.after(RigSet::Advance).before(RigSet::Assemble),
    );
    let run = spawn_run(app.world_mut(), agent, &[], "add numbers", false, None);
    ended(&mut app, run, "restricted tool decision");
    let requests = requests.lock().expect("requests");
    let advertised: Vec<_> = requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(advertised, allowed);
    if executable {
        assert!(app.world().get::<Settled>(run).is_some());
        assert_eq!(adder.peak.load(Ordering::SeqCst), 1);
    } else {
        assert_eq!(
            adder.peak.load(Ordering::SeqCst),
            0,
            "excluded tool must never execute"
        );
        assert!(matches!(
            app.world().get::<Failed>(run),
            Some(Failed(Failure::UnknownToolCall { name })) if name == "add"
        ));
        assert!(app.world().get::<RunResult>(run).is_none());
        assert!(
            app.world()
                .resource::<EffectLogResource>()
                .log()
                .records
                .iter()
                .all(|record| { !matches!(record.kind, EffectKind::ToolCall { .. }) })
        );
    }
}

#[test]
fn active_tools_blocks_an_excluded_granted_tool() {
    assert_active_tools(&["other"], false);
}

#[test]
fn active_tools_empty_blocks_every_granted_tool() {
    assert_active_tools(&[], false);
}

#[test]
fn active_tools_keeps_an_allowed_tool_executable() {
    assert_active_tools(&["add"], true);
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

/// Each call owns a gate, so the host decides when it may finish. An
/// executor may poll the calls in either order without changing the proof.
struct GatedAdder {
    adder: Arc<Adder>,
    gates:
        std::sync::Mutex<std::collections::BTreeMap<i64, futures::channel::oneshot::Receiver<()>>>,
    entered: std::sync::atomic::AtomicUsize,
    outstanding: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}

impl rig_core::serve::Serve for GatedAdder {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::serve::Serve::descriptor(&*self.adder)
    }

    async fn serve(&self, kind: EffectKind, sink: rig_core::serve::OutcomeSink) {
        let EffectKind::ToolCall { args, .. } = &kind else {
            panic!("the gated adder accepts tool calls");
        };
        let args: serde_json::Value = serde_json::from_str(args).unwrap();
        let gate = self
            .gates
            .lock()
            .unwrap()
            .remove(&args["x"].as_i64().unwrap())
            .unwrap();
        let now = self.outstanding.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.entered.fetch_add(1, Ordering::SeqCst);
        gate.await
            .expect("the host releases each call independently");
        rig_core::serve::Serve::serve(&*self.adder, kind, sink).await;
        self.outstanding.fetch_sub(1, Ordering::SeqCst);
    }
}

fn in_flight_tools(world: &mut World) -> usize {
    world
        .query::<&rig_ecs::bus::InFlight>()
        .iter(world)
        .filter(|flight| flight.key.as_str() == ADD)
        .count()
}

#[test]
fn tool_policy_sets_how_many_calls_are_in_flight() {
    for concurrency in [1, 2] {
        for capacity in [1, 16] {
            for reverse in [false, true] {
                let (mut app, agent, adder, requests) = tooling(two_calls_then_text());
                let (first_release, first_gate) = futures::channel::oneshot::channel();
                let (second_release, second_gate) = futures::channel::oneshot::channel();
                let gated = Arc::new(GatedAdder {
                    adder,
                    gates: std::sync::Mutex::new([(1, first_gate), (3, second_gate)].into()),
                    entered: 0.into(),
                    outstanding: 0.into(),
                    peak: 0.into(),
                });
                register(&mut app, ADD, gated.clone());
                app.world_mut()
                    .resource_mut::<rig_ecs::bus::Policy>()
                    .0
                    .command_capacity = capacity;
                app.world_mut()
                    .entity_mut(agent)
                    .insert(ToolPolicy { concurrency });
                let run = spawn_run(app.world_mut(), agent, &[], "add twice", false, None);
                tick_until(&mut app, "requested calls entered their gates", |_| {
                    gated.entered.load(Ordering::SeqCst) == concurrency
                });
                assert_eq!(in_flight_tools(app.world_mut()), concurrency);
                assert_eq!(gated.outstanding.load(Ordering::SeqCst), concurrency);
                assert!(app.world().get::<Settled>(run).is_none());
                if concurrency == 1 {
                    // The first cannot finish while held. The second is a
                    // pending batch child, not a task merely polled late.
                    let calls: Vec<_> = app
                        .world_mut()
                        .query::<(&PendingEffect, Option<&rig_ecs::bus::InFlight>)>()
                        .iter(app.world())
                        .filter(|(effect, _)| effect.key.as_str() == ADD)
                        .map(|(_, flight)| flight.is_some())
                        .collect();
                    assert_eq!(calls.len(), 2);
                    assert_eq!(calls.iter().filter(|flight| **flight).count(), 1);
                    first_release.send(()).unwrap();
                    tick_until(
                        &mut app,
                        "second call starts after the first lands",
                        |world| {
                            gated.entered.load(Ordering::SeqCst) == 2 && in_flight_tools(world) == 1
                        },
                    );
                    assert_eq!(gated.outstanding.load(Ordering::SeqCst), 1);
                    second_release.send(()).unwrap();
                } else {
                    let (early, late) = if reverse {
                        (second_release, first_release)
                    } else {
                        (first_release, second_release)
                    };
                    early.send(()).unwrap();
                    tick_until(
                        &mut app,
                        "one result landed while its peer stays held",
                        |world| in_flight_tools(world) == 1,
                    );
                    assert_eq!(gated.outstanding.load(Ordering::SeqCst), 1);
                    late.send(()).unwrap();
                }
                ended(&mut app, run, "answered");
                assert!(app.world().get::<Settled>(run).is_some());
                assert_eq!(gated.peak.load(Ordering::SeqCst), concurrency);
                let log = app.world().resource::<EffectLogResource>().log();
                let keys: Vec<&str> = log.records.iter().map(|r| r.key.as_str()).collect();
                assert_eq!(
                    keys,
                    [MODEL, ADD, ADD, MODEL],
                    "policy preserves dispatch trace"
                );
                let requests = requests.lock().unwrap();
                assert_eq!(
                    tool_results(&requests[1]),
                    [("c1".into(), "3".into()), ("c2".into(), "7".into())]
                );
            }
        }
    }
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
    add_system(&mut app, repair_to_add.in_set(RigSet::Judge));
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

struct PeerAdder(Adder);

impl rig_core::serve::Serve for PeerAdder {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        let mut descriptor = rig_core::serve::Serve::descriptor(&self.0);
        if let rig_core::effect::FamilyDescriptor::Tool { name, .. } = &mut descriptor.family {
            *name = "peer_add".into();
        }
        descriptor
    }

    async fn serve(&self, kind: EffectKind, sink: rig_core::serve::OutcomeSink) {
        rig_core::serve::Serve::serve(&self.0, kind, sink).await;
    }
}

#[test]
fn repair_keeps_same_spelling_identity_namespaces_distinct() {
    use rig_core::message::{ToolCall, ToolCallId, ToolFunction};
    for generated_invalid in [false, true] {
        let generated = ToolCall::new(
            ToolCallId::minted(0),
            ToolFunction {
                name: if generated_invalid {
                    "multiply"
                } else {
                    "peer_add"
                }
                .into(),
                arguments: serde_json::json!({"x": 2, "y": 3}),
            },
        );
        let explicit = ToolCall::from_wire(
            generated.id.wire_hint(),
            ToolFunction {
                name: if generated_invalid {
                    "peer_add"
                } else {
                    "multiply"
                }
                .into(),
                arguments: serde_json::json!({"x": 4, "y": 5}),
            },
        );
        let original = [generated, explicit];
        let (mut app, agent, _, requests) = tooling(vec![
            original
                .iter()
                .cloned()
                .map(AssistantContent::ToolCall)
                .collect(),
            vec![AssistantContent::text("done")],
        ]);
        let peer = register(
            &mut app,
            "t/tool:peer_add",
            PeerAdder(Adder::new("t/tool:peer_add")),
        );
        app.world_mut()
            .spawn((Grant(peer), Order(1), ChildOf(agent)));
        add_system(&mut app, repair_to_add.in_set(RigSet::Judge));
        let run = spawn_run(
            app.world_mut(),
            agent,
            &[],
            "repair only the invalid call",
            false,
            None,
        );
        ended(&mut app, run, "typed repair complete");
        assert!(app.world().get::<Failed>(run).is_none());
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let calls: Vec<_> = requests[1]
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        let results: Vec<_> = requests[1]
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(results.len(), 2);
        for (index, original) in original.iter().enumerate() {
            assert_eq!(calls[index].id, original.id);
            assert_eq!(calls[index].provider, original.provider);
            assert_eq!(calls[index].function.arguments, original.function.arguments);
            let expected_name = if original.function.name == "multiply" {
                "add"
            } else {
                "peer_add"
            };
            assert_eq!(calls[index].function.name, expected_name);
            let result = results
                .iter()
                .find(|result| result.call == original.id)
                .unwrap();
            assert_eq!(result.provider, original.provider);
            assert_eq!(result.name, expected_name);
            assert!(
                matches!(result.content.as_slice(), [rig_core::message::ToolResultContent::Json { value }] if value == &serde_json::json!(if index == 0 { 5 } else { 9 }))
            );
        }
        let log = app.world().resource::<EffectLogResource>().log();
        assert_eq!(
            log.records
                .iter()
                .filter(|record| matches!(record.kind, EffectKind::ToolCall { .. }))
                .count(),
            2
        );
    }
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
    add_system(&mut app, retry_with_feedback.in_set(RigSet::Judge));
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

#[test]
fn retry_feedback_targets_only_the_invalid_identity_namespace() {
    use rig_core::message::{ToolCall, ToolCallId, ToolFunction, ToolResultContent};
    for generated_invalid in [false, true] {
        let generated = ToolCall::new(
            ToolCallId::minted(0),
            ToolFunction {
                name: if generated_invalid { "multiply" } else { "add" }.into(),
                arguments: serde_json::json!({"x":2,"y":3}),
            },
        );
        let explicit = ToolCall::from_wire(
            generated.id.wire_hint(),
            ToolFunction {
                name: if generated_invalid { "add" } else { "multiply" }.into(),
                arguments: serde_json::json!({"x":4,"y":5}),
            },
        );
        let calls = [generated, explicit];
        let (mut app, agent, _, requests) = tooling(vec![
            calls
                .iter()
                .cloned()
                .map(AssistantContent::ToolCall)
                .collect(),
            vec![AssistantContent::text("done")],
        ]);
        app.world_mut().entity_mut(agent).insert(InvalidCalls {
            retries: 1,
            unhandled: rig_ecs::agent::Unhandled::Fail,
        });
        add_system(&mut app, retry_with_feedback.in_set(RigSet::Judge));
        let run = spawn_run(
            app.world_mut(),
            agent,
            &[],
            "retry invalid identity",
            false,
            None,
        );
        ended(&mut app, run, "typed retry complete");
        assert!(app.world().get::<Failed>(run).is_none());
        let log = app.world().resource::<EffectLogResource>().log();
        assert!(
            log.records
                .iter()
                .all(|record| !matches!(record.kind, EffectKind::ToolCall { .. })),
            "neither invalid call nor its peer executes on retry"
        );
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let results: Vec<_> = requests[1]
            .chat_history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 2);
        for call in &calls {
            let result = results
                .iter()
                .find(|result| result.call == call.id)
                .unwrap();
            assert_eq!(result.provider, call.provider);
            assert_eq!(result.name, call.function.name);
            let expected = if call.function.name == "multiply" {
                "there is no tool named multiply; use add"
            } else {
                rig_ecs::policy::text::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
            };
            assert!(
                matches!(result.content.as_slice(), [ToolResultContent::Text(text)] if text.text == expected)
            );
        }
    }
}
