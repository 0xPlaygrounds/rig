use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ReadTool, ShellTool};
use rig_basics::compaction::{Compact, CompactionPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;
use rig_basics::compaction::summary_author;

/// The spec's pattern: a command starting with `echo test` tests.
const SETTINGS: &str = "[working_set]\ntest_pattern = '^echo test'\n";

/// The agent with the tools, compaction and the plugin, on `store`, working
/// in `project`, after the first frame restored the session.
fn start(store: &MemoryStore, project: &Path) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(SETTINGS))
        .insert_resource(ProjectRoot(project.to_owned()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            EditTool,
            ShellTool,
            CompactionPlugin,
            WorkingSetPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

/// Runs frames until `count` turns ended.
fn until_ended(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= count
    });
}

#[test]
fn the_working_set_follows_a_compaction_and_a_resume() {
    let project = scratch_home("working-set-project").root().to_owned();
    std::fs::create_dir_all(&project).ok();
    let (a, b) = (project.join("a.rs"), project.join("b.rs"));
    std::fs::write(&a, "fn a() {}\n").ok();
    std::fs::write(&b, "fn b() {\n    old();\n}\n").ok();
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": a.display().to_string() })),
        calls(
            "e1",
            "edit",
            json!({
                "path": b.display().to_string(),
                "edits": [{ "old_text": "    old();\n", "new_text": "    new();\n    more();\n" }],
            }),
        ),
        calls("s1", "shell", json!({ "command": "echo test; exit 1" })),
        reply("The test fails."),
        reply("## Goal\nMake the test pass."),
        reply("On it."),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store, &project);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    let fix = Deliver::new(agent, "Fix b.rs and run the test.", DeliveryMode::Steer);
    app.world_mut().trigger(fix);
    until_ended(&mut app, &wakes, 1);
    // `/compact` does this.
    let focus = String::new();
    app.world_mut().trigger(Compact {
        entity: agent,
        focus,
    });
    until_ended(&mut app, &wakes, 2);
    let summary = summary_author();
    let compacted = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| Some(conversation.author(0)? == &summary));
    assert_eq!(compacted, Some(true), "compacted");
    let next = Deliver::new(agent, "Go on.", DeliveryMode::Steer);
    app.world_mut().trigger(next);
    until_ended(&mut app, &wakes, 3);

    let requests = model.requests();
    let last = requests
        .last()
        .map(|request| serde_json::to_string(request).unwrap_or_default());
    let last = last.unwrap_or_default();
    for expected in [
        "Working set",
        "b.rs: 4 lines now (+2 -1 by edit)",
        "- a.rs",
        "`echo test; exit 1` failed (exit 1)",
    ] {
        assert!(last.contains(expected), "{expected} in {last}");
    }
    let tracked = app
        .world()
        .get::<WorkingSet>(agent)
        .cloned()
        .unwrap_or_default();
    assert!(tracked.read.is_empty(), "the read list starts over");
    assert_eq!(tracked.changed.len(), 1, "the changed list stays");

    // A new app on the same session has the block, and the tracking.
    drop(app);
    let (mut resumed, _) = start(&store, &project);
    let world = resumed.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let conversation = world.get::<Conversation>(agent);
    let messages = conversation.map(|conversation| conversation.messages().to_vec());
    let saved = serde_json::to_string(&messages.unwrap_or_default()).unwrap_or_default();
    assert_eq!(saved.matches("Working set").count(), 1, "once: {saved}");
    let tracked = world.get::<WorkingSet>(agent).cloned().unwrap_or_default();
    assert_eq!(tracked.changed.len(), 1, "restored");
    assert_eq!(tracked.runs.len(), 1, "restored");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(WorkingSetPlugin);
}
