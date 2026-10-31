use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, WriteTool};
use rig_basics::subagents::SubagentsPlugin;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, first_agent, log_ended, log_tool_outputs, reply, run_until,
    scripted_models,
};
use rig_tui::AppStatus;
use serde_json::json;

use super::*;

const PARENT: &str = "scope-fence/parent";
const SUB: &str = "scope-fence/sub";
const OUTSIDE: &str = "Outside the current scope";

/// A project beside the test binary with `src/parser/a.rs` and `docs/`.
fn project() -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("sn-scope-fence-project");
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(dir.join("src/parser")).is_ok());
    assert!(std::fs::create_dir_all(dir.join("docs")).is_ok());
    assert!(std::fs::write(dir.join("src/parser/a.rs"), "let x = 1;\n").is_ok());
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

/// An app on `store` for the project `dir`, its agent on `parent`, with
/// the file tools, subagents and the fence.
fn start(
    store: &MemoryStore,
    dir: &Path,
    parent: &MockCompletionModel,
    sub: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([(PARENT, parent), (SUB, sub)]))
        .insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EditTool,
            WriteTool,
            SubagentsPlugin,
        ))
        .add_plugins(ScopeFencePlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = PARENT.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

fn command(app: &mut App, agent: Entity, line: &str) {
    app.world_mut()
        .trigger(RunCommand::new(agent, line.to_owned()));
    app.update();
}

fn status(app: &App) -> String {
    let items = &app.world().resource::<AppStatus>().0;
    items.iter().map(|item| item.text.clone()).collect()
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

fn ended(app: &App) -> usize {
    app.world().resource::<Ended>().0.len()
}

fn write(id: &str, path: &str) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, "write", json!({ "path": path, "content": "x" }))
}

fn turn(events: impl IntoIterator<Item = MockStreamEvent>) -> Vec<MockStreamEvent> {
    let mut events: Vec<MockStreamEvent> = events.into_iter().collect();
    events.push(MockStreamEvent::final_response_with_default_usage());
    events
}

#[test]
fn file_changes_outside_the_scope_are_refused_until_it_is_lifted() {
    let dir = project();
    let edit =
        json!({ "path": "src/parser/a.rs", "edits": [{ "old_text": "1", "new_text": "2" }] });
    let task = json!({ "description": "docs", "prompt": "Write docs/d.md.", "model": SUB });
    let parent = MockCompletionModel::from_stream_turns([
        turn([
            MockStreamEvent::tool_call("e", "edit", edit),
            write("w1", "docs/b.md"),
            write("w2", "src/parser/../../docs/c.md"),
        ]),
        calls("t", "task", task),
        reply("started"),
        reply("the subagent was refused"),
        calls(
            "w3",
            "write",
            json!({ "path": "docs/b.md", "content": "x" }),
        ),
        reply("wrote it"),
    ]);
    let sub =
        MockCompletionModel::from_stream_turns([turn([write("s", "docs/d.md")]), reply("refused")]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &dir, &parent, &sub);
    command(&mut app, agent, "scope src/parser");
    assert_eq!(app.world().resource::<ScopeFence>().line(), "src/parser/");
    assert_eq!(status(&app), "scope: src/parser/");

    app.world_mut()
        .trigger(Deliver::new(agent, "Fix the parser.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 3
    });
    let text = |path: &str| std::fs::read_to_string(dir.join(path)).ok();
    assert_eq!(text("src/parser/a.rs").as_deref(), Some("let x = 2;\n"));
    assert!(text("docs/b.md").is_none() && text("docs/c.md").is_none());
    let requests = parent.requests();
    let second = requests
        .get(1)
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .unwrap_or_default();
    assert_eq!(second.matches(OUTSIDE).count(), 2, "{second}");
    assert!(output(&app, "s").contains(OUTSIDE), "{}", output(&app, "s"));
    assert!(text("docs/d.md").is_none());

    command(&mut app, agent, "scope off");
    assert_eq!(status(&app), "");
    let before = ended(&app);
    app.world_mut()
        .trigger(Deliver::new(agent, "Write the docs.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
    assert_eq!(text("docs/b.md").as_deref(), Some("x"));

    // The fence is saved with the session and comes back on resume.
    command(&mut app, agent, "scope src/parser");
    drop(app);
    let parent = MockCompletionModel::from_stream_turns([
        calls("r", "write", json!({ "path": "docs/e.md", "content": "x" })),
        reply("refused again"),
    ]);
    let (mut app, wakes, agent) = start(&store, &dir, &parent, &sub);
    assert_eq!(status(&app), "scope: src/parser/");
    app.world_mut()
        .trigger(Deliver::new(agent, "Write e.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert!(output(&app, "r").contains(OUTSIDE), "{}", output(&app, "r"));
    assert!(text("docs/e.md").is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ScopeFencePlugin);
}
