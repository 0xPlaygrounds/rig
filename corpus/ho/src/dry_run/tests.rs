use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ReadTool, ShellTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until,
};
use serde_json::json;

use super::*;

/// The kernel, the journal, the coding tools and dry run on `store`, after
/// the first frame restored the session, its agent and its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, DryRunPlugin))
        .add_plugins((ReadTool, WriteTool, EditTool, ShellTool))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// A fresh directory for the test's files, beside the test's binary.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("dry-run-{name}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

/// What the model read as call `id`'s result.
fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

#[test]
fn dry_run_previews_changes_keeps_them_with_the_session_and_applies_them() {
    let dir = scratch("preview");
    let file = dir.join("a.txt");
    std::fs::write(&file, "one").ok();
    let path = file.display().to_string();
    let edit = json!({ "path": "a.txt", "edits": [{ "old_text": "one", "new_text": "two" }] });
    let model = MockCompletionModel::from_stream_turns([
        calls("e1", "edit", edit),
        calls("r1", "read", json!({ "path": "a.txt" })),
        calls("s1", "shell", json!({ "command": "touch b" })),
        calls("s2", "shell", json!({ "command": "ls" })),
        calls(
            "w1",
            "write",
            json!({ "path": "../out.txt", "content": "x" }),
        ),
        reply("Previewed."),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store);
    app.world_mut()
        .entity_mut(agent)
        .insert(WorkDir(dir.clone()));
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    app.world_mut()
        .trigger(RunCommand::new(agent, "dry-run on"));
    app.world_mut()
        .trigger(Deliver::new(agent, "Change a.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.update();

    assert_eq!(std::fs::read_to_string(&file).unwrap_or_default(), "one");
    let edited = output(&app, "e1");
    assert!(
        edited.contains("-one") && edited.contains("+two"),
        "{edited}"
    );
    assert!(edited.contains(NOT_APPLIED), "{edited}");
    let read = output(&app, "r1");
    assert!(read.contains("two") && !read.contains("one"), "{read}");
    let touched = output(&app, "s1");
    assert!(touched.contains(NOT_EXECUTED), "{touched}");
    assert!(!dir.join("b").exists());
    let listed = output(&app, "s2");
    assert!(listed.contains("a.txt"), "{listed}");
    // Confined to the agent's directory, as the real `write` is.
    let outside = output(&app, "w1");
    assert!(outside.contains("outside"), "{outside}");
    drop(app);

    // A new app on the same session still previews the change, in dry run.
    let (mut app, _wakes, agent) = start(&store);
    let overlay = app.world().resource::<Overlay>();
    assert_eq!(overlay.content(&path), Some("two"), "{overlay:?}");
    assert!(app.world().get::<DryRun>(agent).is_some());
    assert_eq!(std::fs::read_to_string(&file).unwrap_or_default(), "one");

    app.world_mut()
        .trigger(RunCommand::new(agent, "dry-run apply"));
    app.update();
    assert_eq!(std::fs::read_to_string(&file).unwrap_or_default(), "two");
    assert!(app.world().resource::<Overlay>().0.is_empty());
    assert!(!dir.with_file_name("out.txt").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DryRunPlugin);
}
