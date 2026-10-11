use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

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

/// Whether a turn ended.
#[derive(Resource, Default)]
struct Ended(bool);

fn dummy(called: On<ToolCalled<Anything>>, mut commands: Commands) {
    commands
        .entity(called.call)
        .insert(ToolOutput(ToolResult::success("ok".into())));
}

fn create_issue(
    called: On<ToolCalled<Issue>>,
    mut created: ResMut<Created>,
    mut commands: Commands,
) {
    created.0.push(called.args.title.clone());
    commands
        .entity(called.call)
        .insert(ToolOutput(ToolResult::success("Created PROJ-1.".into())));
}

/// The names of the tools a request offered.
fn offered(request: &rig_core::completion::CompletionRequest) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[test]
fn a_deferred_tool_is_offered_once_tool_search_found_it() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(DeferredTools::matching(["dummy_*", "jira_*"]))
        .add_plugins((AgentPlugin, JournalPlugin, ToolSearchPlugin))
        .init_resource::<Created>()
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 = true);
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
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0);

    let requests = model.requests();
    assert_eq!(requests.len(), 3, "{requests:?}");
    let [first, second, third] = requests.as_slice() else {
        return;
    };
    assert_eq!(offered(first), [TOOL_SEARCH], "only the core set first");
    let prompt = serde_json::to_string(first).unwrap_or_default();
    assert!(prompt.contains("50 more tools are deferred"), "{prompt}");
    // The search result, in the second request, names the tool.
    let history = serde_json::to_string(&second.chat_history).unwrap_or_default();
    assert!(
        history.contains("- jira_create_issue: Create an issue in a Jira project."),
        "{history}"
    );
    assert_eq!(offered(second), ["jira_create_issue", TOOL_SEARCH]);
    assert_eq!(offered(third), ["jira_create_issue", TOOL_SEARCH]);
    assert_eq!(app.world().resource::<Created>().0, ["Login fails"]);
    let activated = app
        .world()
        .get::<ActivatedTools>(agent)
        .map(|a| a.0.clone());
    assert_eq!(activated, Some(vec!["jira_create_issue".to_owned()]));
}
