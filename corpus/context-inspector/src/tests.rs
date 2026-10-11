use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use corpus_few_shot_examples::{ExamplesProject, FewShotExamplesPlugin};
use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::load;
use rig_harness_test_support::{app_on, connect, reply, run_until};

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

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The notices' texts.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

fn see(mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>) {
    seen.0
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// A project of its own beside the test binary, holding `commit.md`.
fn project() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe
        .parent()?
        .join(format!("context-inspector-{}", std::process::id()));
    let examples = root.join(".rig").join("examples");
    std::fs::create_dir_all(&examples).ok()?;
    std::fs::write(examples.join("commit.md"), EXAMPLE).ok()?;
    Some(root)
}

/// Runs `/<line>` for `agent` and returns the notice it wrote.
fn command(app: &mut App, agent: Entity, line: &str) -> String {
    app.world_mut().resource_mut::<Seen>().0.clear();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: line.to_owned(),
    });
    app.update();
    app.world().resource::<Seen>().0.join("\n")
}

#[test]
fn why_shows_sections_examples_and_tools_and_full_writes_the_request() {
    let root = project();
    assert!(root.is_some(), "the project was written");
    let Some(root) = root else {
        return;
    };
    let (mut app, wakes): (App, Receiver<()>) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, JournalPlugin));
    load::<HouseRulesPlugin>(&mut app, "house-rules", "test");
    load::<FewShotExamplesPlugin>(&mut app, "corpus-few-shot-examples", "test");
    load::<ContextInspectorPlugin>(&mut app, "corpus-context-inspector", "test");
    app.insert_resource(ExamplesProject(root.clone()))
        .insert_resource(RequestDumps(root.join("requests")))
        .init_resource::<Ended>()
        .init_resource::<Seen>()
        .add_systems(Last, see)
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("Committed.")]);
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let (agent, connection) = (agent.ok(), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "one agent, and the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);

    let said = Deliver::new(agent, "commit this", DeliveryMode::Queue).with_origin(Origin::user());
    app.world_mut().trigger(said);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0 >= 1 && world.get::<LastRequest>(agent).is_some()
    });
    let sent = model.requests().len();
    assert_eq!(sent, 1, "the turn's one request reached the model");

    let why = command(&mut app, agent, "why");
    for expected in [
        "<house_rules> \"House rules\"",
        "from HouseRulesPlugin",
        "Tools offered: 1",
        "lookup",
        "+ 2 messages added before the conversation (user, assistant;",
        "by FewShotExamplesPlugin",
    ] {
        assert!(why.contains(expected), "`{expected}` in:\n{why}");
    }

    let full = command(&mut app, agent, "why full");
    let id = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
    let path = root
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
    std::fs::remove_dir_all(root).ok();
}
