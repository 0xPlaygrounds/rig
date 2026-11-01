use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_basics::subagents::SubagentsPlugin;
use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use schemars::JsonSchema;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, log_ended, reply, run_until, start_call,
};

/// The arguments of the test's stand-in tools.
#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// The kernel, the journal, the subagents and the profiles plugin on
/// `store`, with `RIG_HOME` at the fixtures (whose `agents` hold
/// `reviewer.md`) and stand-in `read`, `search` and `write` tools, after
/// the first frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/agent_profiles");
    app.insert_resource(RigHome(Home::at(&fixtures)))
        .insert_resource(ProjectRoot(fixtures))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            SubagentsPlugin,
            AgentProfilesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    for tool in ["read", "search", "write"] {
        let answer = |called: On<ToolCalled<NoArgs>>,
                      mut calls: Query<&mut Decision, ToolCalls>| {
            if let Ok(mut call) = calls.get_mut(called.call) {
                call.decide(Decider::plugin("test"), ToolResult::success("ok".into()));
            }
        };
        app.add_open_tool(tool, "A stand-in.", ToolOptions::default(), answer);
    }
    app.finish();
    app.update();
    (app, wakes)
}

/// The app's agent nothing spawned, answered by `model`.
fn answered_by(app: &mut App, model: &MockCompletionModel) -> Entity {
    let world = app.world_mut();
    let found = world
        .query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>()
        .iter(world)
        .next();
    let agent = found.unwrap_or_else(|| world.spawn(Agent).id());
    if let Some(connection) = connect(model) {
        let choice = ModelChoice(connection.spec.reference());
        world.entity_mut(agent).insert((connection, choice));
    }
    agent
}

/// Runs frames until `turns` turns of the session ended, then one more,
/// in which the journal writes.
fn until_ended(app: &mut App, wakes: &Receiver<()>, turns: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
}

/// Runs `/line` for `agent`, then sends it a message and waits for its turn.
fn command_then_ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
    let ended = app.world().resource::<Ended>().0.len();
    let ask = Deliver::new(agent, "Look at the code.", DeliveryMode::Queue);
    app.world_mut().trigger(ask);
    until_ended(app, wakes, ended + 1);
}

/// What `agent`'s `task` call with `args` answers, run by passes of the
/// stages until it is answered.
fn refusal_of(app: &mut App, agent: Entity, args: serde_json::Value) -> String {
    let call = start_call(app.world_mut(), agent, ("task", "t0"), args);
    for _ in 0..8 {
        if let Some(outcome) = call.and_then(|call| app.world().get::<Outcome>(call)) {
            return outcome.tool_result().output().render();
        }
        app.world_mut().run_schedule(AgentPass);
    }
    String::new()
}

/// The system prompt and the tools offered of `request`.
fn sent(request: Option<&CompletionRequest>) -> (String, Vec<String>) {
    let Some(request) = request else {
        return (String::new(), Vec::new());
    };
    let prompt = format!("{:?}", request.chat_history.first());
    let tools = request
        .tools
        .iter()
        .map(|tool| tool.name.as_str().to_owned());
    (prompt, tools.collect())
}

/// `/profile reviewer` gives the agent the profile's prompt and only its
/// tools, a `write` call is refused, the profile is back after a restart,
/// and `/profile none` restores the agent as it was.
#[test]
fn a_profile_applies_survives_a_restart_and_comes_off() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model =
        MockCompletionModel::from_stream_turns([calls("w1", "write", json!({})), reply("Looked.")]);
    let agent = answered_by(&mut app, &model);
    command_then_ask(&mut app, &wakes, agent, "profile reviewer");

    let requests = model.requests();
    let (prompt, tools) = sent(requests.first());
    assert!(prompt.contains("Be terse."), "{prompt}");
    assert_eq!(tools, ["read", "search"]);
    let conversation = app.world().get::<Conversation>(agent);
    let history = format!(
        "{:?}",
        conversation.map(|conversation| conversation.messages())
    );
    assert!(history.contains("no tool named `write`"), "{history}");
    drop(app);

    // A restart brings the profile back, with its prompt section.
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([reply("Looked again.")]);
    let agent = answered_by(&mut app, &model);
    let applied = app.world().get::<AppliedProfile>(agent);
    let applied = applied.map(|applied| applied.name.as_str());
    assert_eq!(applied, Some("reviewer"));

    command_then_ask(&mut app, &wakes, agent, "profile none");
    let requests = model.requests();
    let (prompt, tools) = sent(requests.first());
    assert!(!prompt.contains("Be terse."), "{prompt}");
    assert!(tools.iter().any(|tool| tool == "write"), "{tools:?}");
}

/// A `task` call with `profile: "reviewer"` starts a subagent with the
/// profile's prompt and tools; an agent offered `task` sees the profiles
/// listed, and an unknown profile is refused with the known ones.
#[test]
fn task_starts_a_subagent_as_a_profile() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([reply("Reviewed."), reply("Thanks.")]);
    let parent = answered_by(&mut app, &model);

    let unknown = json!({ "description": "Review", "prompt": "Review it.", "profile": "poet" });
    let refusal = refusal_of(&mut app, parent, unknown);
    assert!(refusal.contains("Known profiles: reviewer"), "{refusal}");

    let args = json!({ "description": "Review", "prompt": "Review it.", "profile": "reviewer" });
    start_call(app.world_mut(), parent, ("task", "t1"), args);
    // The subagent's turn, then the parent's on its report.
    until_ended(&mut app, &wakes, 2);

    let requests = model.requests();
    let (prompt, tools) = sent(requests.first());
    assert!(prompt.contains("Be terse."), "{prompt}");
    assert_eq!(tools, ["message", "read", "search", "wait"]);
    let (prompt, tools) = sent(requests.get(1));
    assert!(tools.iter().any(|tool| tool == "task"), "{tools:?}");
    assert!(!prompt.contains("Be terse."), "{prompt}");
    let mut offered = requests
        .get(1)
        .into_iter()
        .flat_map(|request| &request.tools);
    let task = offered.find(|tool| tool.name.as_str() == "task");
    let description = task.map(|task| task.description.as_str());
    assert!(
        description.is_some_and(|text| text.contains("- reviewer: Reads and reviews")),
        "{description:?}"
    );
    let properties = task.and_then(|task| task.parameters.get("properties"));
    assert!(properties.is_some_and(|declared| declared.get("profile").is_some()));
}

#[test]
fn a_profile_in_a_limited_run_keeps_its_tools_for_later_runs() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    app.insert_resource(RunTools::new(["task", "read", "message", "wait"]));
    let model = MockCompletionModel::from_stream_turns([reply("Reviewed."), reply("Thanks.")]);
    let parent = answered_by(&mut app, &model);
    let args = json!({ "description": "Review", "prompt": "Review it.", "profile": "reviewer" });
    start_call(app.world_mut(), parent, ("task", "t1"), args);
    until_ended(&mut app, &wakes, 2);
    let (_, tools) = sent(model.requests().first());
    assert_eq!(tools, ["message", "read", "wait"]);
    let world = app.world();
    let child = world
        .get::<Spawned>(parent)
        .and_then(|spawned| spawned.iter().next());
    let access = child.and_then(|child| world.get::<ToolAccess>(child));
    // Out of this run only: a later run gives it the profile's `search`.
    assert!(access.is_some_and(|access| access.keeps("search") && !access.allows("search")));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AgentProfilesPlugin);
}
