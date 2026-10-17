use std::path::Path;

use rig_basics::coding_tools::ReadTool;
use rig_basics::compaction::{Compact, CompactionPlugin};
use rig_basics::subagents::SubagentsPlugin;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// The db rule's body.
const DB: &str = "Never use SELECT *.";

/// The migrations rule's body.
const DOWN: &str = "Every migration needs a down section.";

/// A project in `root` with its files and its `.rig/rules`: one always-on,
/// two for globs, one to look up, and one whose front matter is broken.
fn project(root: &Path) -> Option<()> {
    for (file, text) in [
        ("src/db/q.rs", "fn q() {}\n"),
        ("src/db/x.rs", "fn x() {}\n"),
        ("src/main.rs", "fn main() {}\n"),
        ("migrations/1.sql", "create table t (id int);\n"),
        (
            ".rig/rules/always.md",
            "---\nalways: true\n---\nUse tabs.\n",
        ),
        (
            ".rig/rules/db.md",
            &format!("---\nglobs: [\"src/db/**\", \"*.sql\"]\n---\n{DB}\n"),
        ),
        (
            ".rig/rules/migrations.md",
            &format!("---\nglobs: [\"migrations/*.sql\"]\n---\n{DOWN}\n"),
        ),
        (
            ".rig/rules/style.md",
            "---\ndescription: Naming conventions\n---\nSnake case.\n",
        ),
        (
            ".rig/rules/broken.md",
            "---\nglobs: [unclosed\n---\nOops.\n",
        ),
    ] {
        let path = root.join(file);
        std::fs::create_dir_all(path.parent()?).ok()?;
        std::fs::write(path, text).ok()?;
    }
    Some(())
}

/// The rules section of the system prompt.
fn section(world: &World) -> String {
    let sections = world.try_query::<&PromptSection>();
    let mut texts = sections.iter().flat_map(|query| query.iter_manual(world));
    texts
        .find(|section| section.tag == "rules")
        .map(|section| section.text.clone())
        .unwrap_or_default()
}

/// The output of the tool call `id`, as the model read it.
fn output(world: &World, id: &str) -> String {
    let outputs = &world.resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

/// Each request `model` was sent, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let json = requests.iter().map(serde_json::to_string);
    json.map(Result::unwrap_or_default).collect()
}

#[test]
fn rules_reach_the_agent_by_prompt_by_touch_and_on_request() {
    let home = scratch_home("corpus-sn-glob-rules");
    let root = home.root().join("project");
    assert!(project(&root).is_some(), "the project is written");
    let path = |file: &str| json!({ "path": file });
    let main = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("r1", "read", path("src/db/q.rs")),
            MockStreamEvent::tool_call("r2", "read", path("src/main.rs")),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        calls("l1", "list", path("migrations")),
        calls("r3", "read", path("migrations/1.sql")),
        calls("r4", RULE, json!({ "name": "style" })),
        reply("Done."),
        // The summary `/compact` asks for.
        reply("## Goal\nLook at the queries."),
        calls("r5", "read", path("src/db/q.rs")),
        calls(
            "t1",
            "task",
            json!({ "description": "Read x", "prompt": "Read src/db/x.rs.", "model": "test/sub" }),
        ),
        reply("Started."),
        reply("Thanks."),
        reply("Thanks again."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([
        calls("s1", "read", path("src/db/x.rs")),
        reply("It defines x."),
    ]);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(scripted_models([("test/main", &main), ("test/sub", &sub)]))
        .insert_resource(ProjectRoot(root.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, WatchPlugin, ReadTool))
        .add_tool_per_call(
            rig_harness::rig_tools::List::default(),
            ToolOptions::default(),
            |place| rig_harness::rig_tools::List {
                workspace: place.workspace,
                base: Some(place.base),
                confine: place.confine,
            },
        )
        .add_plugins((SubagentsPlugin, CompactionPlugin, GlobRulesPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
    run_until(&mut app, &wakes, |world| section(world).contains("style"));

    // Turn 1: the reads, the look-up.
    let ask = Deliver::new(agent, "Look at the queries.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let requests = sent(&main);
    let first = requests.first().map(String::as_str).unwrap_or_default();
    assert!(first.contains("Use tabs."), "{first}");
    assert!(first.contains("- style: Naming conventions"), "{first}");
    assert!(!first.contains("Snake case."), "{first}");
    assert!(
        !first.contains("Oops."),
        "a broken rule is skipped: {first}"
    );
    let q = output(app.world(), "r1");
    assert!(q.contains("fn q()"), "{q}");
    assert!(q.contains("Rules for `src/db/**` (db.md)"), "{q}");
    assert!(q.trim_end().ends_with(DB), "{q}");
    let main_rs = output(app.world(), "r2");
    assert!(
        main_rs.contains("fn main()") && !main_rs.contains("Rules for"),
        "{main_rs}"
    );
    // A listing touches what it lists: the migrations rule comes with it,
    // the db rule held already does not; a read of a listed file repeats
    // neither.
    let listing = output(app.world(), "l1");
    assert!(listing.contains("1.sql"), "{listing}");
    assert!(listing.trim_end().ends_with(DOWN), "{listing}");
    assert!(!listing.contains(DB), "{listing}");
    let sql = output(app.world(), "r3");
    assert!(sql.contains("create table"), "{sql}");
    assert!(!sql.contains(DB) && !sql.contains(DOWN), "{sql}");
    assert_eq!(output(app.world(), "r4"), "Snake case.");
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned = notices
        .iter()
        .any(|notice| notice.text.contains("broken.md skipped"));
    assert!(warned, "{notices:?}");

    // Condensed, the conversation lost the rule: the next touch brings it
    // again; a subagent gets it by its own touch.
    let compact = Compact {
        entity: agent,
        focus: String::new(),
    };
    app.world_mut().trigger(compact);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 1
    });
    let again = Deliver::new(agent, "Once more, and ask a subagent.", DeliveryMode::Steer);
    app.world_mut().trigger(again);
    run_until(&mut app, &wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key("s1") && main.requests().len() >= 9
    });
    let q = output(app.world(), "r5");
    assert!(q.trim_end().ends_with(DB), "{q}");
    let x = output(app.world(), "s1");
    assert!(x.contains("fn x()") && x.trim_end().ends_with(DB), "{x}");
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(GlobRulesPlugin);
}
