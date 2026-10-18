use std::cell::Cell;
use std::sync::mpsc::Receiver;

use rig_core::completion::{Cost, Usage};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    ToolOutputs, app_on, calls, log_tool_outputs, reply, run_until, scripted_models,
};

/// The parent's model.
const PARENT: &str = "corpus-policy/parent";
/// The policy's default model, which answers every subagent.
const SUBAGENT: &str = "corpus-policy/subagent";

/// Each subagent's turn starts and its finish, in order.
#[derive(Resource, Default)]
struct Steps(Vec<(&'static str, Entity)>);

/// A session under `policy` with a parent answered by `parent`, and a
/// `read` tool; every subagent is answered by `subagents`, the policy's
/// default model.
struct Session {
    app: App,
    wakes: Receiver<()>,
    parent: Entity,
}

impl Session {
    fn new(
        policy: SubagentsConfig,
        parent: &MockCompletionModel,
        subagents: &MockCompletionModel,
    ) -> Self {
        let policy = SubagentsConfig {
            default_model: Some(SUBAGENT.to_owned()),
            ..policy
        };
        let models = scripted_models([(PARENT, parent), (SUBAGENT, subagents)]);
        let (mut app, wakes) = app_on(&MemoryStore::default());
        let read = |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
            let output = Outcome::from(ToolResult::success("Nothing here.".into()));
            commands.entity(called.call).insert_if_new(output);
        };
        app.insert_resource(models)
            .insert_resource(policy)
            .init_resource::<Steps>()
            .init_resource::<ToolOutputs>()
            .add_plugins((AgentPlugin, SessionLogPlugin, SubagentPolicyPlugin))
            .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
            .add_observer(log_tool_outputs)
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
        let choice = ModelChoice(PARENT.to_owned());
        let parent = app.world_mut().spawn((Agent, choice)).id();
        app.finish();
        Self { app, wakes, parent }
    }

    /// Asks the parent to start, and runs until `done`.
    fn run(&mut self, done: impl Fn(&World) -> bool) {
        let go = Deliver::new(self.parent, "Go.", DeliveryMode::Queue);
        self.app.world_mut().trigger(go);
        run_until(&mut self.app, &self.wakes, done);
    }

    /// The subagents.
    fn subagents(&mut self) -> Vec<Entity> {
        let world = self.app.world_mut();
        let mut subagents = world.query_filtered::<Entity, With<SpawnedBy>>();
        subagents.iter(world).collect()
    }

    /// The output of the tool call `id`, as its model read it.
    fn output(&self, id: &str) -> String {
        let outputs = &self.app.world().resource::<ToolOutputs>().0;
        outputs.get(id).cloned().unwrap_or_default()
    }
}

/// A reply that calls `task` once.
fn task(id: &str, prompt: &str) -> Vec<MockStreamEvent> {
    calls(id, TASK, json!({ "description": id, "prompt": prompt }))
}

/// What the parent's model read in its request `n` (from 0).
fn read_by(model: &MockCompletionModel, n: usize) -> String {
    let request = model.requests().into_iter().nth(n);
    request.map_or_else(String::new, |r| json!(r.chat_history).to_string())
}

#[test]
fn a_spawn_beyond_max_depth_is_refused_for_the_model_to_read() {
    let policy = SubagentsConfig {
        max_depth: 1,
        ..SubagentsConfig::default()
    };
    let parent = MockCompletionModel::from_stream_turns([
        task("t1", "x"),
        reply("Waiting."),
        reply("Done."),
    ]);
    let subagent =
        MockCompletionModel::from_stream_turns([task("t2", "y"), reply("Did it myself.")]);
    let mut session = Session::new(policy, &parent, &subagent);
    session.run(|_| parent.request_count() >= 3);

    assert_eq!(session.subagents().len(), 1);
    let nested = session.output("t2");
    // At the depth limit the subagent is not given `task`, so the kernel
    // refuses the call before any gate; either way the model reads why.
    assert!(
        nested.contains("subagent depth limit 1 reached")
            || nested.contains("no tool named `task`"),
        "{nested}"
    );
    assert!(read_by(&parent, 2).contains("Did it myself."));
}

#[test]
fn a_new_subagent_gets_the_default_tools() {
    let policy = SubagentsConfig {
        default_tools: Some(vec!["read".to_owned()]),
        ..SubagentsConfig::default()
    };
    let parent = MockCompletionModel::from_stream_turns([
        task("t1", "x"),
        reply("Waiting."),
        reply("Done."),
    ]);
    let subagent = MockCompletionModel::from_stream_turns([reply("Read it.")]);
    let mut session = Session::new(policy, &parent, &subagent);
    session.run(|_| parent.request_count() >= 3);

    // The default model answered it, offered only the default tools.
    let first = subagent.requests().into_iter().next();
    let offered: Vec<String> = first
        .iter()
        .flat_map(|request| request.tools.iter().map(|tool| tool.name.to_string()))
        .collect();
    assert_eq!(offered, ["message", "read", "wait"]);
}

#[test]
fn spawns_beyond_max_concurrent_wait_for_a_slot() {
    let policy = SubagentsConfig {
        max_concurrent: Some(1),
        ..SubagentsConfig::default()
    };
    let both = vec![
        MockStreamEvent::tool_call(
            "t1",
            TASK,
            json!({ "description": "t1", "prompt": "First." }),
        ),
        MockStreamEvent::tool_call(
            "t2",
            TASK,
            json!({ "description": "t2", "prompt": "Second." }),
        ),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let parent = MockCompletionModel::from_stream_turns(
        std::iter::once(both).chain((0..4).map(|_| reply("Noted."))),
    );
    let subagents =
        MockCompletionModel::from_stream_turns([reply("First done."), reply("Second done.")]);
    let mut session = Session::new(policy, &parent, &subagents);
    // The parent's status line said one task was queued meanwhile.
    let queued = Cell::new(0);
    session.run(|world| {
        let counts = world.iter_entities().filter_map(|e| e.get::<QueuedTasks>());
        queued.set(queued.get().max(counts.map(|q| q.0).max().unwrap_or(0)));
        let steps = world.resource::<Steps>().0.len();
        // Both reports reached the parent: the second may come a request
        // after the one the first came in.
        let read = format!("{:?}", parent.requests());
        steps == 4 && read.contains("First done.") && read.contains("Second done.")
    });

    assert_eq!(queued.get(), 1);
    // The second started only once the first finished.
    let steps = &session.app.world().resource::<Steps>().0;
    let order: Vec<&str> = steps.iter().map(|(step, _)| *step).collect();
    assert_eq!(order, ["started", "finished", "started", "finished"]);
    let agents: Vec<Entity> = steps.iter().map(|(_, agent)| *agent).collect();
    assert_eq!(agents.first(), agents.get(1));
    assert_ne!(agents.first(), agents.get(2));
    let read = format!("{:?}", parent.requests());
    assert!(read.contains("First done.") && read.contains("Second done."));
}

#[test]
fn a_subagent_over_a_cap_is_stopped_and_reported_with_its_partial_answer() {
    let over_calls = SubagentsConfig {
        max_turn_calls: Some(1),
        ..SubagentsConfig::default()
    };
    let over_cost = SubagentsConfig {
        max_cost: Some(0.5),
        ..SubagentsConfig::default()
    };
    // Its first reply costs $1, as its provider says.
    let usage = Usage::new().input_tokens(1_000).cost(Cost::from_total(1.0));
    for (policy, cap) in [(over_calls, "max_turn_calls"), (over_cost, "max_cost")] {
        let parent = MockCompletionModel::from_stream_turns([
            task("t1", "x"),
            reply("Waiting."),
            reply("Done."),
        ]);
        let read = vec![
            MockStreamEvent::text("Half way."),
            MockStreamEvent::tool_call("r1", "read", json!({})),
            MockStreamEvent::final_response(usage),
        ];
        let subagent = MockCompletionModel::from_stream_turns([read, reply("Never sent.")]);
        let mut session = Session::new(policy, &parent, &subagent);
        session.run(|_| parent.request_count() >= 3);

        let report = read_by(&parent, 2);
        assert!(
            report.contains(cap) && report.contains("Half way."),
            "{report}"
        );
        assert!(!report.contains("Never sent."));
    }
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SubagentPolicyPlugin);
}
