use std::sync::mpsc::Receiver;

use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_on, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until,
};

use super::*;

/// The arguments of the dummy tools.
#[derive(Deserialize, JsonSchema)]
struct Anything {}

/// The arguments of `jira_create_issue`.
#[derive(Deserialize, JsonSchema)]
struct Issue {
    /// The issue's title.
    title: String,
}

/// The issues `jira_create_issue` created.
#[derive(Resource, Default)]
struct Created(Vec<String>);

fn dummy(called: On<ToolCalled<Anything>>, mut commands: Commands) {
    let result = ToolResult::success(ToolOutput::text("ok"));
    commands
        .entity(called.call)
        .insert_if_new(Outcome::from(result));
}

fn create_issue(
    called: On<ToolCalled<Issue>>,
    mut created: ResMut<Created>,
    mut commands: Commands,
) {
    created.0.push(called.args.title.clone());
    let result = ToolResult::success(ToolOutput::text("Created PROJ-1."));
    commands
        .entity(called.call)
        .insert_if_new(Outcome::from(result));
}

/// The kernel, the journal and the plugin on `store`, with 49 deferred
/// dummy tools and `jira_create_issue`, after the first frame restored the
/// session, and the loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let settings = "[tool_search]\ntools = [\"dummy_*\", \"jira_*\"]";
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, ToolSearchPlugin))
        .init_resource::<Created>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    for index in 0..49 {
        let name = format!("dummy_{index}");
        let description = format!("Dummy tool number {index}.");
        app.add_open_tool(&name, &description, ToolOptions::default(), dummy);
    }
    app.add_open_tool(
        "jira_create_issue",
        "Create an issue in a Jira project.\nMore detail here.",
        ToolOptions::default(),
        create_issue,
    );
    app.finish();
    app.update();
    (app, wakes)
}

/// The names of the tools a request offered.
fn offered(request: &CompletionRequest) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

/// What `/tools` says about the app's agent.
fn count_tools(app: &mut App) -> String {
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(RunCommand::new(agent, "tools"));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .last()
        .map(|notice| notice.text.clone())
        .unwrap_or_default()
}

#[test]
fn a_deferred_tool_is_offered_once_tool_search_found_or_the_model_called_it() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            TOOL_SEARCH,
            serde_json::json!({ "query": "jira issue" }),
        ),
        calls(
            "2",
            "jira_create_issue",
            serde_json::json!({ "title": "Login fails" }),
        ),
        // Never searched for: it runs anyway, and is active from then on.
        calls("3", "dummy_7", serde_json::json!({})),
        reply("Filed PROJ-1."),
    ]);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    app.world_mut().trigger(Deliver::new(
        agent,
        "File a Jira issue: login fails.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let requests = model.requests();
    assert_eq!(requests.len(), 4, "{requests:?}");
    let [first, second, third, fourth] = requests.as_slice() else {
        return;
    };
    assert_eq!(offered(first), [TOOL_SEARCH], "only the core set first");
    let prompt = serde_json::to_string(first).unwrap_or_default();
    assert!(prompt.contains("50 more tools are deferred"), "{prompt}");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let found = outputs.get("1").cloned().unwrap_or_default();
    assert!(
        found.contains("- jira_create_issue: Create an issue in a Jira project."),
        "the search names the tool: {found}"
    );
    assert_eq!(offered(second), ["jira_create_issue", TOOL_SEARCH]);
    assert_eq!(app.world().resource::<Created>().0, ["Login fails"]);
    assert_eq!(offered(third), ["jira_create_issue", TOOL_SEARCH]);
    let guessed = outputs.get("3").cloned().unwrap_or_default();
    assert!(
        guessed.contains("`dummy_7` was deferred and not offered yet"),
        "the guessed call's result says it is active now: {guessed}"
    );
    assert_eq!(
        offered(fourth),
        ["dummy_7", "jira_create_issue", TOOL_SEARCH]
    );
    let counted = "Tools: 3 active, 48 deferred. Activated: jira_create_issue, dummy_7.";
    assert_eq!(count_tools(&mut app), counted);
    drop(app);

    // A restart on the same session brings the activated tools back.
    let (mut app, _) = start(&store);
    assert_eq!(count_tools(&mut app), counted, "saved with the session");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ToolSearchPlugin);
}
