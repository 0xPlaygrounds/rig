use rig_cassette::journal::MemoryStore;
use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

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

/// The names of the tools a request offered.
fn offered(request: &CompletionRequest) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

/// The request's conversation, as JSON.
fn history(request: &CompletionRequest) -> String {
    serde_json::to_string(&request.chat_history).unwrap_or_default()
}

#[test]
fn a_deferred_tool_is_offered_once_tool_search_found_or_the_model_called_it() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let settings = "[tool_search]\ntools = [\"dummy_*\", \"jira_*\"]";
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Settings::from_toml(settings))
        .add_plugins((AgentPlugin, JournalPlugin, ToolSearchPlugin))
        .init_resource::<Created>()
        .init_resource::<Ended>()
        .add_observer(log_ended);
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
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and the catalog's model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(
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
    assert!(
        history(second).contains("- jira_create_issue: Create an issue in a Jira project."),
        "the search result names the tool: {}",
        history(second)
    );
    assert_eq!(offered(second), ["jira_create_issue", TOOL_SEARCH]);
    assert_eq!(app.world().resource::<Created>().0, ["Login fails"]);
    assert_eq!(offered(third), ["jira_create_issue", TOOL_SEARCH]);
    assert!(
        history(fourth).contains("`dummy_7` was deferred and not offered yet"),
        "the guessed call's result says it is active now: {}",
        history(fourth)
    );
    assert_eq!(
        offered(fourth),
        ["dummy_7", "jira_create_issue", TOOL_SEARCH]
    );
    let activated = app
        .world()
        .get::<ActivatedTools>(agent)
        .map(|a| a.0.clone());
    assert_eq!(
        activated,
        Some(vec!["jira_create_issue".to_owned(), "dummy_7".to_owned()])
    );
}
