//! Checkpoint holds are durable tool-batch boundaries, independent of tick granularity.
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{
        Grant, MaxTurns, Run, Settled,
        checkpoint::{
            ToolTurnCommit, ToolTurnCommitted, ToolTurnHolds, TurnAssistant, TurnHoldError,
            TurnResults, hold_after_tool_turn, release_tool_turn_hold,
        },
        content::parts::read_message,
    },
    checkpoint::{RestoreMode, load_world, save_world},
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
fn owners_are_independent_and_invalid_hold_requests_are_rejected() {
    let (mut app, run, requests) = setup(1, 2);
    assert_eq!(
        hold_after_tool_turn(app.world_mut(), run, "", 1),
        Err(TurnHoldError::EmptyOwner)
    );
    assert_eq!(
        hold_after_tool_turn(app.world_mut(), run, "a", 0),
        Err(TurnHoldError::ZeroTurn)
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
        Err(TurnHoldError::NotLiveRun)
    );
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
                checkpoint.entities[turn].shift_remove(type_name::<TurnResults>());
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
                checkpoint.entities[turn].shift_remove(type_name::<ToolTurnCommit>());
            }
            "empty_owner" => {
                let mut holds = holds.clone();
                let turn = object(&mut holds).shift_remove("workspace").unwrap();
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
