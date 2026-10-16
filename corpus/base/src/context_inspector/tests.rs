use corpus_shared::few_shot_examples::FewShotExamplesPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::load;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};

use super::*;

const EXAMPLE: &str = "---
when: commit
---
## user
Commit the parser fix.
## assistant
fix(parser): accept a trailing comma
";

const RULES: &str = "Answer in one word.";

/// A plugin adding the "House rules" section and a tool.
#[derive(Default)]
struct HouseRulesPlugin;

impl Plugin for HouseRulesPlugin {
    fn build(&self, app: &mut App) {
        let section = PromptSection::new(PromptSection::ORDER_PROJECT, "house_rules", RULES);
        app.world_mut().spawn((Name::new("House rules"), section));
        app.add_open_tool(
            "lookup",
            "Look a word up.",
            ToolOptions::default(),
            |_: On<ToolCalled<serde_json::Value>>| {},
        );
    }
}

/// Types `line` to `agent`, as the user, and returns the notices it wrote.
fn command(app: &mut App, agent: Entity, line: &str) -> String {
    app.world_mut()
        .resource_mut::<Collected<Notice>>()
        .0
        .clear();
    let typed = Deliver::new(agent, line, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let texts: Vec<&str> = notices.iter().map(|notice| notice.text.as_str()).collect();
    texts.join("\n")
}

#[test]
fn why_shows_sections_examples_and_tools_and_full_writes_the_request() {
    let home = scratch_home("context-inspector");
    let root = home.root().join("project");
    let examples = root.join(".rig").join("examples");
    std::fs::create_dir_all(&examples).ok();
    std::fs::write(examples.join("commit.md"), EXAMPLE).ok();
    let session = SessionId::generate();
    let (mut app, wakes) = app_in(&home, &session);
    app.insert_resource(ProjectRoot(root)).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        WatchPlugin,
    ));
    load::<HouseRulesPlugin>(&mut app, "house-rules", "test");
    load::<FewShotExamplesPlugin>(&mut app, "corpus-few-shot-examples", "test");
    load::<ContextInspectorPlugin>(&mut app, "corpus-context-inspector", "test");
    collect::<Notice>(&mut app)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<FileChanges>(&mut app);
    app.finish();
    app.update();
    // The few-shot plugin reads its examples on its watch's first listing.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Collected<FileChanges>>().0.is_empty()
    });
    let model = MockCompletionModel::from_stream_turns([reply("Committed.")]);
    let world = app.world_mut();
    let (agent, connection) = (first_agent(world), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "one agent, and the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    let said = Deliver::new(agent, "commit this", DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(said);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let sent = model.requests().len();
    assert_eq!(sent, 1, "the turn's one request reached the model");

    let why = command(&mut app, agent, "/why");
    for expected in [
        "<house_rules> \"House rules\"",
        "from HouseRulesPlugin",
        "Tools offered: 1",
        "lookup",
        "+ 2 messages added before the conversation (user, assistant;",
        "tokens), by plugin few-shot-examples",
    ] {
        assert!(why.contains(expected), "`{expected}` in:\n{why}");
    }

    let full = command(&mut app, agent, "/context --last full");
    let id = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
    let path = home
        .session(&session)
        .path()
        .join("requests")
        .join(format!("{}-last-request.json", id.unwrap_or_default()));
    assert!(full.contains(&path.display().to_string()), "{full}");
    let json = std::fs::read_to_string(&path).unwrap_or_default();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
    assert!(parsed.is_ok(), "{json}");
    assert!(
        json.contains(RULES) && json.contains("Commit the parser fix."),
        "{json}"
    );
    assert_eq!(model.requests().len(), sent, "/why calls no model");
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ContextInspectorPlugin);
}
