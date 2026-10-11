use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// The model every subagent is answered by; one at a time takes its turns.
#[derive(Resource)]
struct SubagentModel(MockCompletionModel);

/// Each subagent's turn starts and its finish, in order.
#[derive(Resource, Default)]
struct Steps(Vec<(&'static str, Entity)>);

/// A session under `policy` with a parent answered by `parent`, and a
/// `read` tool; every subagent is answered by `subagents`.
struct Session {
    app: App,
    wakes: Receiver<()>,
}

impl Session {
    fn new(
        policy: SubagentPolicy,
        parent: &MockCompletionModel,
        subagents: &MockCompletionModel,
    ) -> Self {
        let (mut app, wakes) = app_on(&MemoryStore::default());
        let read = |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
            let output = ToolOutput(ToolResult::success("Nothing here.".into()));
            commands.entity(called.call).insert(output);
        };
        app.insert_resource(Models(Connector::new(Catalog::default())))
            .insert_resource(policy)
            .insert_resource(SubagentModel(subagents.clone()))
            .init_resource::<Steps>()
            .add_plugins((AgentPlugin, JournalPlugin, SubagentPolicyPlugin))
            .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
            .add_observer(
                |spawned: On<Add<SpawnedBy>>, model: Res<SubagentModel>, mut commands: Commands| {
                    if let Some(connection) = connect(&model.0) {
                        commands.entity(spawned.entity).insert(connection);
                    }
                },
            )
            .add_observer(
                |started: On<Add<ActiveTurn>>,
                 sub: Query<(), With<SpawnedBy>>,
                 mut steps: ResMut<Steps>| {
                    if sub.contains(started.entity) {
                        steps.0.push(("started", started.entity));
                    }
                },
            )
            .add_observer(
                |step: On<Insert<Lifecycle>>,
                 lifecycles: Query<&Lifecycle>,
                 mut steps: ResMut<Steps>| {
                    if let Ok(Lifecycle::Finished(_)) = lifecycles.get(step.entity) {
                        steps.0.push(("finished", step.entity));
                    }
                },
            );
        if let Some(connection) = connect(parent) {
            let choice = ModelChoice(connection.spec.reference());
            app.world_mut().spawn((Agent, connection, choice));
        }
        app.finish();
        Self { app, wakes }
    }

    /// Asks the parent to start, and runs until its model was asked `calls`
    /// times.
    fn run(&mut self, parent: &MockCompletionModel, calls: usize) {
        let world = self.app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let first = roots.iter(world).next();
        if let Some(agent) = first {
            world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));
        }
        run_until(&mut self.app, &self.wakes, |_| {
            parent.request_count() >= calls
        });
    }

    /// The subagents.
    fn subagents(&mut self) -> Vec<Entity> {
        let world = self.app.world_mut();
        let mut subagents = world.query_filtered::<Entity, With<SpawnedBy>>();
        subagents.iter(world).collect()
    }
}

fn task(id: &str, prompt: &str) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, TASK, json!({ "prompt": prompt }))
}

#[test]
fn a_spawn_beyond_max_depth_is_refused_for_the_model_to_read() {
    let policy = SubagentPolicy {
        max_depth: 1,
        ..SubagentPolicy::default()
    };
    let parent = MockCompletionModel::from_stream_turns([
        calls("t1", TASK, json!({ "prompt": "x" })),
        reply("Waiting."),
        reply("Done."),
    ]);
    let subagent = MockCompletionModel::from_stream_turns([
        calls("t2", TASK, json!({ "prompt": "y" })),
        reply("Did it myself."),
    ]);
    let mut session = Session::new(policy, &parent, &subagent);
    session.run(&parent, 3);

    assert_eq!(session.subagents().len(), 1);
    let requests = subagent.requests();
    let second = requests
        .get(1)
        .map(|request| json!(request.chat_history).to_string());
    let second = second.unwrap_or_default();
    assert!(
        second.contains("subagent depth limit 1 reached; do it yourself"),
        "{second}"
    );
    let report = parent
        .requests()
        .get(2)
        .map(|r| json!(r.chat_history).to_string());
    assert!(report.unwrap_or_default().contains("Did it myself."));
}

#[test]
fn a_new_subagent_gets_the_default_tools() {
    let policy = SubagentPolicy {
        default_tools: Some(vec!["read".to_owned()]),
        ..SubagentPolicy::default()
    };
    let parent = MockCompletionModel::from_stream_turns([
        calls("t1", TASK, json!({ "prompt": "x" })),
        reply("Waiting."),
        reply("Done."),
    ]);
    let subagent = MockCompletionModel::from_stream_turns([reply("Read it.")]);
    let mut session = Session::new(policy, &parent, &subagent);
    session.run(&parent, 3);

    let first = subagent.requests().into_iter().next();
    let offered: Vec<String> = first
        .map(|request| {
            request
                .tools
                .iter()
                .map(|tool| tool.name.to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(offered, ["read"]);
}

#[test]
fn spawns_beyond_max_concurrent_wait_for_a_slot() {
    let policy = SubagentPolicy {
        max_concurrent: Some(1),
        ..SubagentPolicy::default()
    };
    let both = vec![
        task("t1", "First."),
        task("t2", "Second."),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let parent =
        MockCompletionModel::from_stream_turns([both, reply("Waiting."), reply("Both done.")]);
    let subagents =
        MockCompletionModel::from_stream_turns([reply("First done."), reply("Second done.")]);
    let mut session = Session::new(policy, &parent, &subagents);
    session.run(&parent, 3);

    // The second was queued, and started only once the first finished.
    let results = parent
        .requests()
        .get(1)
        .map(|r| json!(r.chat_history).to_string());
    assert!(results.unwrap_or_default().contains("Queued subagent"));
    let steps = &session.app.world().resource::<Steps>().0;
    let order: Vec<&str> = steps.iter().map(|(step, _)| *step).collect();
    assert_eq!(order, ["started", "finished", "started", "finished"]);
    let agents: Vec<Entity> = steps.iter().map(|(_, agent)| *agent).collect();
    assert_eq!(agents.first(), agents.get(1));
    assert_ne!(agents.first(), agents.get(2));
    let report = parent
        .requests()
        .get(2)
        .map(|r| json!(r.chat_history).to_string());
    let report = report.unwrap_or_default();
    assert!(report.contains("First done.") && report.contains("Second done."));
}

#[test]
fn a_subagent_over_max_turn_calls_is_stopped_and_reported_with_its_partial_answer() {
    let policy = SubagentPolicy {
        max_turn_calls: Some(1),
        ..SubagentPolicy::default()
    };
    let parent = MockCompletionModel::from_stream_turns([
        calls("t1", TASK, json!({ "prompt": "x" })),
        reply("Waiting."),
        reply("Done."),
    ]);
    let read = vec![
        MockStreamEvent::text("Half way."),
        MockStreamEvent::tool_call("r1", "read", json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let subagent = MockCompletionModel::from_stream_turns([read, reply("Never sent.")]);
    let mut session = Session::new(policy, &parent, &subagent);
    session.run(&parent, 3);

    let report = parent
        .requests()
        .get(2)
        .map(|r| json!(r.chat_history).to_string());
    let report = report.unwrap_or_default();
    assert!(
        report.contains("max_turn_calls") && report.contains("Half way."),
        "{report}"
    );
    assert!(!report.contains("Never sent."));
}
