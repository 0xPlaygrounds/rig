//! Checkpoint holds are durable tool-batch boundaries, independent of tick granularity.
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{
        Cancelled, Failed, Failure, Grant, MaxTurns, Run, Settled,
        checkpoint::{
            CheckpointError, ToolTurnCommit, ToolTurnCommitted, ToolTurnHolds, TurnAssistant,
            TurnResults, hold_after_tool_turn, release_tool_turn_hold,
        },
        content::parts::read_message,
    },
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::RunCommands,
};
use run_support::*;
use std::sync::{Arc, Mutex};
const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

fn setup(turns: usize, limit: usize) -> (bevy_app::App, Entity, RequestsSeen) {
    let mut app = app();
    let script = (0..turns)
        .map(|i| {
            vec![call(
                &format!("call-{i}"),
                "add",
                serde_json::json!({"x":i,"y":1}),
            )]
        })
        .collect();
    let (agent, requests) = scripted_agent(&mut app, MODEL, script);
    let add = register(&mut app, ADD, Adder::new(ADD));
    app.world_mut().entity_mut(agent).insert(MaxTurns(limit));
    app.world_mut().spawn((Grant(add), ChildOf(agent)));
    let run = app.world_mut().spawn_run(agent, &[], "count", false, None);
    (app, run, requests)
}
fn committed(world: &mut World, run: Entity, number: usize) -> bool {
    world
        .query::<(&ChildOf, &ToolTurnCommit)>()
        .iter(world)
        .any(|(parent, c)| parent.parent() == run && c.turn == number)
}
fn assert_stays_held(app: &mut bevy_app::App, requests: &RequestsSeen, count: usize) {
    for _ in 0..8 {
        app.update();
    }
    assert_eq!(
        requests.lock().unwrap().len(),
        count,
        "ordinary quiescence updates dispatched through hold"
    );
}

#[test]
fn every_selected_turn_holds_under_quiescence_and_release_continues_once() {
    let (mut app, run, requests) = setup(4, 5);
    for turn in 1..=4 {
        hold_after_tool_turn(app.world_mut(), run, "checkpoint", turn).unwrap();
        tick_until(&mut app, "committed turn", |w| committed(w, run, turn));
        assert_stays_held(&mut app, &requests, turn);
        assert!(release_tool_turn_hold(app.world_mut(), run, "checkpoint").unwrap());
    }
    tick_until(&mut app, "settled", |w| w.get::<Settled>(run).is_some());
    assert_eq!(requests.lock().unwrap().len(), 5);
}

#[test]
fn owners_are_independent_and_invalid_hold_requests_are_rejected() {
    let (mut app, run, requests) = setup(1, 2);
    assert_eq!(
        hold_after_tool_turn(app.world_mut(), run, "", 1),
        Err(CheckpointError::EmptyOwner)
    );
    assert_eq!(
        hold_after_tool_turn(app.world_mut(), run, "a", 0),
        Err(CheckpointError::ZeroTurn)
    );
    assert!(hold_after_tool_turn(app.world_mut(), run, "a", 1).unwrap());
    assert!(!hold_after_tool_turn(app.world_mut(), run, "a", 2).unwrap());
    hold_after_tool_turn(app.world_mut(), run, "b", 1).unwrap();
    tick_until(&mut app, "held", |w| committed(w, run, 1));
    assert!(!release_tool_turn_hold(app.world_mut(), run, "stranger").unwrap());
    release_tool_turn_hold(app.world_mut(), run, "a").unwrap();
    assert_stays_held(&mut app, &requests, 1);
    release_tool_turn_hold(app.world_mut(), run, "b").unwrap();
    tick_until(&mut app, "settled", |w| w.get::<Settled>(run).is_some());
    assert!(app.world().get::<ToolTurnHolds>(run).is_none());
    assert_eq!(
        hold_after_tool_turn(app.world_mut(), run, "a", 1),
        Err(CheckpointError::NotLiveRun)
    );
}

#[test]
fn cancellation_and_turn_budget_end_even_with_an_armed_hold() {
    let (mut app, run, requests) = setup(1, 2);
    hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
    tick_until(&mut app, "held", |w| committed(w, run, 1));
    app.world_mut()
        .entity_mut(run)
        .insert(Cancelled("stop".into()));
    tick_until(&mut app, "cancelled", |w| w.get::<Failed>(run).is_some());
    assert!(matches!(
        &app.world().get::<Failed>(run).unwrap().0,
        Failure::Cancelled(_)
    ));
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert!(release_tool_turn_hold(app.world_mut(), run, "checkpoint").unwrap());
    let (mut app, run, requests) = setup(1, 1);
    hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
    tick_until(&mut app, "budget ended", |w| w.get::<Failed>(run).is_some());
    assert!(committed(app.world_mut(), run, 1));
    assert!(matches!(
        app.world().get::<Failed>(run).unwrap().0,
        Failure::MaxTurns { limit: 1 }
    ));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

fn observe(app: &mut bevy_app::App) -> Arc<Mutex<Vec<usize>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let captured = seen.clone();
    app.world_mut().add_observer(
        move |event: On<ToolTurnCommitted>,
              turns: Query<(&ToolTurnCommit, &TurnAssistant, &TurnResults)>| {
            let (commit, assistant, results) = turns
                .get(event.turn)
                .expect("commit and links visible at notification");
            assert_ne!(assistant.0, results.0);
            captured.lock().unwrap().push(commit.turn);
        },
    );
    seen
}

#[test]
fn fresh_world_restore_preserves_holds_links_and_emits_only_new_commits() {
    let (mut first, run, requests) = setup(2, 3);
    let a = observe(&mut first);
    let b = observe(&mut first);
    hold_after_tool_turn(first.world_mut(), run, "workspace", 1).unwrap();
    tick_until(&mut first, "held", |w| committed(w, run, 1));
    assert_eq!(*a.lock().unwrap(), vec![1]);
    assert_eq!(*b.lock().unwrap(), vec![1]);
    assert_stays_held(&mut first, &requests, 1);
    let checkpoint = save_world(first.world_mut()).unwrap();
    let checkpoint = Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
    drop(first);
    let mut restored = app();
    let (model, requests) = Scripted::new(
        MODEL,
        vec![
            vec![call("call-1", "add", serde_json::json!({"x":1,"y":1}))],
            vec![AssistantContent::text("done")],
        ],
    );
    register(&mut restored, MODEL, model);
    register(&mut restored, ADD, Adder::new(ADD));
    let seen = observe(&mut restored);
    let loaded = load_world(&checkpoint, restored.world_mut(), RestoreMode::Strict, []).unwrap();
    let run = loaded.with::<Run>(restored.world())[0];
    assert!(committed(restored.world_mut(), run, 1));
    let links: Vec<_> = restored
        .world_mut()
        .query::<(&TurnAssistant, &TurnResults)>()
        .iter(restored.world())
        .map(|(a, r)| (a.0, r.0))
        .collect();
    assert_eq!(links.len(), 1);
    for (a, r) in links {
        read_message(restored.world(), a).unwrap();
        read_message(restored.world(), r).unwrap();
    }
    assert_stays_held(&mut restored, &requests, 0);
    assert!(seen.lock().unwrap().is_empty());
    release_tool_turn_hold(restored.world_mut(), run, "workspace").unwrap();
    tick_until(&mut restored, "settled", |w| {
        w.get::<Settled>(run).is_some()
    });
    assert_eq!(*seen.lock().unwrap(), vec![2]);
    assert_eq!(requests.lock().unwrap().len(), 2);
    // A run spawned after the load is sequenced after the loaded one.
    let agent = loaded.with::<rig_ecs::agent::Owner>(restored.world())[0];
    let later = restored
        .world_mut()
        .spawn_run(agent, &[], "later", false, None);
    let seq = |world: &World, run: Entity| world.get::<rig_ecs::agent::RunSeq>(run).unwrap().0;
    assert!(seq(restored.world(), later) > seq(restored.world(), run));
}

#[test]
fn one_held_run_does_not_stop_an_unrelated_run() {
    let (mut app, held, requests) = setup(1, 2);
    hold_after_tool_turn(app.world_mut(), held, "workspace", 1).unwrap();
    tick_until(&mut app, "held", |w| committed(w, held, 1));
    let (other_model, other_requests) = Capturing::new("other/model", "free");
    let model = register(&mut app, "other/model", other_model);
    let agent = spawn_agent(app.world_mut(), "other", model);
    let other = app.world_mut().spawn_run(agent, &[], "other", false, None);
    tick_until(&mut app, "other settled", |w| {
        w.get::<Settled>(other).is_some()
    });
    assert_eq!(other_requests.lock().unwrap().len(), 1);
    assert_stays_held(&mut app, &requests, 1);
    assert!(app.world().get::<Settled>(held).is_none());
}

struct GatedTool {
    gates: Mutex<std::collections::BTreeMap<usize, futures::channel::oneshot::Receiver<()>>>,
    completed: Arc<Mutex<Vec<usize>>>,
}
impl rig_core::serve::Serve for GatedTool {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::serve::Serve::descriptor(&Adder::new(ADD))
    }
    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        _: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        let rig_core::effect::EffectKind::ToolCall { args, .. } = kind else {
            panic!("tool request")
        };
        let args: serde_json::Value = serde_json::from_str(&args).unwrap();
        let index = usize::try_from(args["x"].as_u64().unwrap()).unwrap();
        let gate = self.gates.lock().unwrap().remove(&index).unwrap();
        gate.await.unwrap();
        self.completed.lock().unwrap().push(index);
        let outcome = if index == 1 {
            Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Internal,
                "expected tool failure",
            ))
        } else {
            Ok(rig_core::effect::Outcome::ToolResult {
                result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::text(
                    format!("result-{index}"),
                )),
            })
        };
        rig_core::serve::Reply::Outcome(outcome)
    }
}

#[test]
fn partial_out_of_order_parallel_batch_has_no_commit_until_every_result_lands() {
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![
            (0..3)
                .map(|i| call(&format!("c{i}"), "add", serde_json::json!({"x":i,"y":0})))
                .collect(),
        ],
    );
    let completed = Arc::new(Mutex::new(Vec::new()));
    let mut gates = std::collections::BTreeMap::new();
    let mut senders = std::collections::BTreeMap::new();
    for i in 0..3 {
        let (tx, rx) = futures::channel::oneshot::channel();
        gates.insert(i, rx);
        senders.insert(i, tx);
    }
    let tool = register(
        &mut app,
        ADD,
        GatedTool {
            gates: Mutex::new(gates),
            completed: completed.clone(),
        },
    );
    app.world_mut()
        .entity_mut(agent)
        .insert((MaxTurns(2), rig_ecs::agent::ToolPolicy { concurrency: 3 }));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "parallel", false, None);
    let seen = observe(&mut app);
    hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
    for (count, index) in [2, 1, 0].into_iter().enumerate() {
        senders.remove(&index).unwrap().send(()).unwrap();
        tick_until(&mut app, "tool completed", |_| {
            completed.lock().unwrap().len() == count + 1
        });
        if count < 2 {
            assert!(!committed(app.world_mut(), run, 1));
            assert!(seen.lock().unwrap().is_empty());
            assert_eq!(requests.lock().unwrap().len(), 1);
        }
    }
    tick_until(&mut app, "batch committed", |w| committed(w, run, 1));
    assert_eq!(*completed.lock().unwrap(), vec![2, 1, 0]);
    assert_eq!(*seen.lock().unwrap(), vec![1]);
    let results = app
        .world_mut()
        .query::<&TurnResults>()
        .single(app.world())
        .unwrap()
        .0;
    let parts = read_message(app.world(), results).unwrap();
    let rig_ecs::agent::MessageParts::User { content } = parts else {
        panic!("results must be user content")
    };
    let ids: Vec<_> = content
        .iter()
        .map(|part| {
            let rig_core::message::UserContent::ToolResult(r) = part else {
                panic!("tool result")
            };
            r.call.to_string()
        })
        .collect();
    assert_eq!(ids, vec!["explicit:c0", "explicit:c1", "explicit:c2"]);
    let encoded = serde_json::to_string(&content).unwrap();
    assert!(encoded.contains("expected tool failure"));
    assert!(encoded.contains("result-0"));
    assert!(encoded.contains("result-2"));
    assert_stays_held(&mut app, &requests, 1);
    release_tool_turn_hold(app.world_mut(), run, "checkpoint").unwrap();
    tick_until(&mut app, "settled", |w| w.get::<Settled>(run).is_some());
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(*completed.lock().unwrap(), vec![2, 1, 0]);
}

#[test]
fn corrupt_commits_links_and_hold_owners_are_rejected_before_destination_mutation() {
    use std::any::type_name;
    let (mut source, run, _) = setup(1, 2);
    hold_after_tool_turn(source.world_mut(), run, "workspace", 1).unwrap();
    tick_until(&mut source, "held", |w| committed(w, run, 1));
    let good = save_world(source.world_mut()).unwrap();
    let turn = good
        .entities
        .iter()
        .position(|e| e.contains_key(type_name::<ToolTurnCommit>()))
        .unwrap();
    let run = good
        .entities
        .iter()
        .position(|e| e.contains_key(type_name::<Run>()))
        .unwrap();
    let holds = good.entities[run][type_name::<ToolTurnHolds>()].clone();
    // The value's first object, whatever the reflected shape wraps it in.
    fn object(value: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
        match value {
            serde_json::Value::Object(map) => map,
            serde_json::Value::Array(items) => object(&mut items[0]),
            other => panic!("no object in {other}"),
        }
    }
    for defect in [
        "missing_results",
        "wrong_role",
        "zero_commit",
        "future_commit",
        "orphan_results",
        "empty_owner",
        "zero_hold",
        "hold_on_turn",
        "results_off_an_utterance",
    ] {
        let mut checkpoint = good.clone();
        match defect {
            "missing_results" => {
                checkpoint.entities[turn].remove(type_name::<TurnResults>());
            }
            "wrong_role" => {
                let assistant = checkpoint.entities[turn][type_name::<TurnAssistant>()].clone();
                checkpoint.entities[turn].insert(type_name::<TurnResults>().into(), assistant);
            }
            "zero_commit" | "future_commit" => {
                let commit = checkpoint.entities[turn]
                    .get_mut(type_name::<ToolTurnCommit>())
                    .unwrap();
                object(commit)["turn"] = (if defect == "zero_commit" { 0 } else { 99 }).into();
            }
            "orphan_results" => {
                checkpoint.entities[turn].remove(type_name::<ToolTurnCommit>());
            }
            "empty_owner" => {
                let mut holds = holds.clone();
                let turn = object(&mut holds).remove("workspace").unwrap();
                object(&mut holds).insert(String::new(), turn);
                checkpoint.entities[run].insert(type_name::<ToolTurnHolds>().into(), holds);
            }
            "zero_hold" => {
                let mut holds = holds.clone();
                object(&mut holds)["workspace"] = 0.into();
                checkpoint.entities[run].insert(type_name::<ToolTurnHolds>().into(), holds);
            }
            "hold_on_turn" => {
                checkpoint.entities[turn]
                    .insert(type_name::<ToolTurnHolds>().into(), holds.clone());
            }
            // A link escaping the utterances is invalid even when its target
            // is a real checkpoint entity.
            "results_off_an_utterance" => {
                checkpoint.entities[turn].insert(type_name::<TurnResults>().into(), run.into());
            }
            _ => panic!("unknown corruption"),
        }
        let mut destination = app();
        let (model, _) = Scripted::new(MODEL, vec![]);
        register(&mut destination, MODEL, model);
        register(&mut destination, ADD, Adder::new(ADD));
        let sentinel = destination.world_mut().spawn_empty().id();
        let before = destination.world().entities().len();
        assert!(
            load_world(
                &checkpoint,
                destination.world_mut(),
                RestoreMode::Strict,
                []
            )
            .is_err(),
            "accepted {defect}"
        );
        assert_eq!(
            destination.world().entities().len(),
            before,
            "partial destination mutation for {defect}"
        );
        assert!(destination.world().get_entity(sentinel).is_ok());
    }
}

struct CountedAdder(Arc<std::sync::atomic::AtomicUsize>);
impl rig_core::serve::Serve for CountedAdder {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::serve::Serve::descriptor(&Adder::new(ADD))
    }
    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        rig_core::serve::Serve::serve(&Adder::new(ADD), kind, dispatch).await
    }
}
struct RetryModel {
    requests: RequestsSeen,
}
impl rig_core::serve::Serve for RetryModel {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::serve::Serve::descriptor(&Scripted::new(MODEL, vec![]).0)
    }
    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        _: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        use rig_core::{
            completion::{CompletionResponse, Usage},
            effect::{EffectKind, Outcome},
            error::{ErrorKind, ErrorReport},
            serve::Reply,
        };
        let EffectKind::Completion { request, .. } = kind else {
            panic!("completion")
        };
        let index = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len()
        };
        Reply::Outcome(match index {
            1 => Ok(Outcome::Completion(CompletionResponse::new(
                vec![call("c1", "add", serde_json::json!({"x":1,"y":2}))],
                Usage::default(),
                "retry-model",
                serde_json::json!({}),
            ))),
            2 => Err(ErrorReport::new(ErrorKind::ProviderResponse, "transient")
                .with_http_status(503)
                .with_retryable(true)),
            3 => Ok(Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("done")],
                Usage::default(),
                "retry-model",
                serde_json::json!({}),
            ))),
            _ => panic!("unexpected repeated request"),
        })
    }
}

#[test]
fn released_checkpoint_provider_retry_preserves_request_and_does_not_repeat_tool() {
    let mut app = app();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = register(
        &mut app,
        MODEL,
        RetryModel {
            requests: requests.clone(),
        },
    );
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tool = register(&mut app, ADD, CountedAdder(calls.clone()));
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert((MaxTurns(2), rig_ecs::agent::ProviderRetries(1)));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    let seen = observe(&mut app);
    hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
    tick_until(&mut app, "held", |w| committed(w, run, 1));
    assert_stays_held(&mut app, &requests, 1);
    release_tool_turn_hold(app.world_mut(), run, "checkpoint").unwrap();
    tick_until(&mut app, "retry settled", |w| {
        w.get::<Settled>(run).is_some()
    });
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(*seen.lock().unwrap(), vec![1]);
    assert_eq!(
        app.world()
            .get::<rig_ecs::agent::ProviderRetried>(run)
            .unwrap()
            .0,
        1
    );
    assert_eq!(
        app.world().get::<rig_ecs::agent::Cursor>(run).unwrap().turn,
        2
    );
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        serde_json::to_value(&requests[1]).unwrap(),
        serde_json::to_value(&requests[2]).unwrap()
    );
}

#[test]
fn output_tool_settlement_commits_only_a_real_mixed_batch_and_ignores_hold() {
    use rig_ecs::agent::{Output, OutputKind, OutputToolConfig, RunResult};
    for mixed in [false, true] {
        let mut app = app();
        let mut parts = Vec::new();
        if mixed {
            parts.push(call("real", "add", serde_json::json!({"x":1,"y":2})));
        }
        parts.push(call("output", "submit", serde_json::json!({"answer":42})));
        let (agent, requests) = scripted_agent(&mut app, MODEL, vec![parts]);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = register(&mut app, ADD, CountedAdder(calls.clone()));
        app.world_mut().entity_mut(agent).insert((Output {mode:OutputKind::Tool,schema:Some(serde_json::json!({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}))},OutputToolConfig {name:Some("submit".into()),description:None,augment_preamble:false}));
        app.world_mut().spawn((Grant(tool), ChildOf(agent)));
        app.world_mut().add_observer(move |event: On<Add, Settled>, commits: Query<(&ChildOf, &ToolTurnCommit, &TurnAssistant, &TurnResults)>| {
            let count = commits.iter().filter(|(parent, _, assistant, results)| {
                assert_ne!(assistant.0, results.0);
                parent.parent() == event.entity
            }).count();
            assert_eq!(count, usize::from(mixed), "terminal observers must see the complete commit");
        });
        let run = app
            .world_mut()
            .spawn_run(agent, &[], "extract", false, None);
        let seen = observe(&mut app);
        hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
        tick_until(&mut app, "output settled", |w| {
            w.get::<Settled>(run).is_some()
        });
        assert_eq!(
            app.world().get::<RunResult>(run).unwrap().0,
            "{\"answer\":42}"
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            usize::from(mixed)
        );
        assert_eq!(committed(app.world_mut(), run, 1), mixed);
        assert_eq!(*seen.lock().unwrap(), if mixed { vec![1] } else { vec![] });
        assert_eq!(requests.lock().unwrap().len(), 1);
        if mixed {
            let results = app
                .world_mut()
                .query::<&TurnResults>()
                .single(app.world())
                .unwrap()
                .0;
            let rig_ecs::agent::MessageParts::User { content } =
                read_message(app.world(), results).unwrap()
            else {
                panic!("results")
            };
            assert_eq!(
                content.len(),
                1,
                "output call is not an executed tool result"
            );
        }
        save_world(app.world_mut()).unwrap();
    }
}

#[test]
fn invalid_call_retry_feedback_is_not_a_completed_tool_batch() {
    use rig_ecs::{
        agent::{InvalidCall, InvalidCalls, InvalidRetries, Resolution, Unhandled},
        bus::RigSchedule,
        systems::RigSet,
    };
    fn retry(invalid: Query<(Entity, &InvalidCall), Without<Resolution>>, mut commands: Commands) {
        for (entity, _) in &invalid {
            commands.entity(entity).insert(Resolution::Retry {
                feedback: "use add".into(),
            });
        }
    }
    let mut app = app();
    app.world_mut()
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .unwrap()
        .add_systems(retry.in_set(RigSet::Judge));
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![
            vec![
                call("bad", "missing", serde_json::json!({})),
                call("skipped", "add", serde_json::json!({"x":1,"y":2})),
            ],
            vec![call("good", "add", serde_json::json!({"x":1,"y":2}))],
            vec![AssistantContent::text("done")],
        ],
    );
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tool = register(&mut app, ADD, CountedAdder(calls.clone()));
    app.world_mut().entity_mut(agent).insert((
        MaxTurns(3),
        InvalidCalls {
            retries: 1,
            unhandled: Unhandled::Fail,
        },
    ));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    let seen = observe(&mut app);
    hold_after_tool_turn(app.world_mut(), run, "checkpoint", 1).unwrap();
    tick_until(&mut app, "real batch committed", |w| committed(w, run, 2));
    assert!(!committed(app.world_mut(), run, 1));
    assert_eq!(*seen.lock().unwrap(), vec![2]);
    assert_stays_held(&mut app, &requests, 2);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(app.world().get::<InvalidRetries>(run).unwrap().0, 1);
    assert!(
        serde_json::to_string(&requests.lock().unwrap()[1])
            .unwrap()
            .contains("use add")
    );
    release_tool_turn_hold(app.world_mut(), run, "checkpoint").unwrap();
    tick_until(&mut app, "settled", |w| w.get::<Settled>(run).is_some());
    assert_eq!(requests.lock().unwrap().len(), 3);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn terminal_cleanup_suppresses_commit_notification_for_deleted_run() {
    use rig_ecs::agent::{Output, OutputKind, OutputToolConfig};
    let mut app = app();
    let (agent, requests) = scripted_agent(
        &mut app,
        MODEL,
        vec![vec![
            call("real", "add", serde_json::json!({"x":1,"y":2})),
            call("output", "submit", serde_json::json!({"answer":42})),
        ]],
    );
    let tool = register(&mut app, ADD, Adder::new(ADD));
    app.world_mut().entity_mut(agent).insert((Output {mode:OutputKind::Tool,schema:Some(serde_json::json!({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}))},OutputToolConfig {name:Some("submit".into()),description:None,augment_preamble:false}));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    app.world_mut()
        .add_observer(|event: On<Add, Settled>, mut commands: Commands| {
            commands.entity(event.entity).despawn();
        });
    let seen = observe(&mut app);
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "extract", false, None);
    tick_until(&mut app, "terminal cleanup", |w| w.get_entity(run).is_err());
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert!(
        seen.lock().unwrap().is_empty(),
        "deleted graph must not generate stale notification"
    );
}

#[test]
fn forked_held_run_remaps_committed_utterances_and_releases_independently() {
    let (mut app, original, requests) = setup(1, 2);
    hold_after_tool_turn(app.world_mut(), original, "workspace", 1).unwrap();
    tick_until(&mut app, "held", |w| committed(w, original, 1));
    let seen = observe(&mut app);
    let fork = rig_ecs::agent::fork(app.world_mut(), original);
    assert!(committed(app.world_mut(), fork, 1));
    let links: Vec<_> = app
        .world_mut()
        .query::<(&ChildOf, &TurnAssistant, &TurnResults)>()
        .iter(app.world())
        .map(|(p, a, r)| (p.parent(), a.0, r.0))
        .collect();
    assert_eq!(links.len(), 2);
    let original_links = links.iter().find(|(r, _, _)| *r == original).unwrap();
    let fork_links = links.iter().find(|(r, _, _)| *r == fork).unwrap();
    assert_ne!(original_links.1, fork_links.1);
    assert_ne!(original_links.2, fork_links.2);
    for (owner, assistant, results) in links {
        for entity in [assistant, results] {
            assert_eq!(app.world().get::<ChildOf>(entity).unwrap().parent(), owner);
            read_message(app.world(), entity).unwrap();
        }
    }
    assert_stays_held(&mut app, &requests, 1);
    assert!(seen.lock().unwrap().is_empty());
    release_tool_turn_hold(app.world_mut(), fork, "workspace").unwrap();
    tick_until(&mut app, "fork settled", |w| {
        w.get::<Settled>(fork).is_some()
    });
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert!(app.world().get::<Settled>(original).is_none());
    assert!(
        app.world()
            .get::<ToolTurnHolds>(original)
            .unwrap()
            .blocks(1)
    );
    save_world(app.world_mut()).unwrap();
    release_tool_turn_hold(app.world_mut(), original, "workspace").unwrap();
    tick_until(&mut app, "original settled", |w| {
        w.get::<Settled>(original).is_some()
    });
    assert_eq!(requests.lock().unwrap().len(), 3);
}
