use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use schemars::JsonSchema;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The arguments of the test's stand-in tools.
#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

/// The kernel, the journal and the profiles plugin on `store`, reading the
/// fixture profiles, with stand-in `read`, `search` and `write` tools, after
/// the first frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/agents");
    app.insert_resource(ProfileDirs(vec![fixtures]))
        .add_plugins((AgentPlugin, JournalPlugin, AgentProfilesPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    for tool in ["read", "search", "write"] {
        let answer = |called: On<ToolCalled<NoArgs>>, mut commands: Commands| {
            let done = ToolOutput(ToolResult::success("ok".into()));
            commands.entity(called.call).insert_if_new(done);
        };
        app.add_open_tool(tool, "A stand-in.", ToolOptions::default(), answer);
    }
    app.finish();
    app.update();
    (app, wakes)
}

/// The app's agent, answered by `model`.
fn answered_by(app: &mut App, model: &MockCompletionModel) -> Entity {
    let world = app.world_mut();
    let found = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let agent = found.unwrap_or_else(|| world.spawn(Agent).id());
    if let Some(connection) = connect(model) {
        let choice = ModelChoice(connection.spec.reference());
        world.entity_mut(agent).insert((connection, choice));
    }
    agent
}

/// Runs `/line` for `agent`, then sends it a message and waits for the turn.
fn command_then_ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
    app.update();
    let ended = app.world().resource::<Ended>().0;
    let ask = Deliver::new(agent, "Look at the code.", DeliveryMode::Queue);
    app.world_mut().trigger(ask);
    run_until(app, wakes, |world| world.resource::<Ended>().0 > ended);
    // The journal writes at the end of the frame.
    app.update();
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
    let history = format!("{:?}", conversation.map(Conversation::messages));
    assert!(history.contains("no tool named `write`"), "{history}");
    drop(app);

    // A restart brings the profile back, with its prompt section.
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([reply("Looked again.")]);
    let agent = answered_by(&mut app, &model);
    let applied = app.world().get::<AppliedProfile>(agent);
    assert_eq!(
        applied.map(|applied| applied.name.as_str()),
        Some("reviewer")
    );
    let world = app.world_mut();
    let mut sections = world.query::<(&PromptSection, &SectionOf)>();
    let profile_prompt = sections
        .iter(world)
        .find(|(section, of)| of.0 == agent && section.tag == "agent_profile");
    let text = profile_prompt.map(|(section, _)| section.text.as_str());
    assert_eq!(text, Some("Be terse."));

    command_then_ask(&mut app, &wakes, agent, "profile none");
    let requests = model.requests();
    let (prompt, tools) = sent(requests.first());
    assert!(!prompt.contains("Be terse."), "{prompt}");
    assert_eq!(tools, ["read", "search", "write"]);
}
