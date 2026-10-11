use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ReadTool;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, connect, reply, run_until};
use rig_subagents::SubagentsPlugin;
use serde_json::{Value, json};

use super::*;

/// A directory of this test's own under the build directory.
fn scratch() -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"),
        PathBuf::from,
    );
    target.join(format!("corpus-export-test-{}", std::process::id()))
}

#[test]
fn exports_the_saved_session_with_the_subagent_under_its_task_call() {
    let dir = scratch();
    std::fs::create_dir_all(&dir).ok();
    let notes = dir.join("notes.txt");
    std::fs::write(&notes, "The answer is in the subagent.").ok();

    // The parent reads a file and starts a subagent in one reply, waits,
    // and answers once the subagent reported; the subagent answers by a
    // model of its own.
    let read = json!({ "path": notes.display().to_string() });
    let task = json!({ "description": "Find the answer", "prompt": "Answer briefly." });
    let parent_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("r1", "read", read),
            MockStreamEvent::tool_call("t1", "task", task),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Waiting for the subagent."),
        reply("Done."),
    ]);
    let child_model = MockCompletionModel::from_stream_turns([reply("sub-answer")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let child_connection = connect(&child_model);
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ReadTool,
            SubagentsPlugin,
            ExportPlugin,
        ))
        .add_observer(move |spawned: On<Add<SpawnedBy>>, mut commands: Commands| {
            if let Some(connection) = child_connection.clone() {
                commands.entity(spawned.entity).insert(connection);
            }
        });
    let connection = connect(&parent_model);
    assert!(connection.is_some(), "the scripted model does not connect");
    let Some(connection) = connection else {
        return;
    };
    let choice = ModelChoice(connection.spec.reference());
    let parent = app.world_mut().spawn((Agent, connection, choice)).id();
    app.finish();
    app.update();

    let asked = "Read the notes and ask a subagent.";
    app.world_mut()
        .trigger(Deliver::new(parent, asked, DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        let idle = world
            .try_query::<&TurnOf>()
            .is_some_and(|mut turns| turns.iter(world).next().is_none());
        parent_model.request_count() == 3 && idle
    });
    assert_eq!(parent_model.request_count(), 3);

    let export = |app: &mut App, line: String| {
        app.world_mut()
            .trigger(Deliver::new(parent, line, DeliveryMode::Queue));
        app.update();
    };
    let markdown = dir.join("out.md");
    export(&mut app, format!("/export md {}", markdown.display()));
    let text = std::fs::read_to_string(&markdown).unwrap_or_default();
    assert!(text.contains(asked), "{text}");
    assert!(text.contains("Tool call `read`"), "{text}");
    assert!(text.contains("Done."), "{text}");
    let after_task = text.split("Tool call `task`").nth(1).unwrap_or_default();
    let subagent = after_task.split("### Subagent").nth(1).unwrap_or_default();
    assert!(subagent.contains("sub-answer"), "{text}");

    let json_path = dir.join("out.json");
    export(&mut app, format!("/export json {}", json_path.display()));
    let exported = std::fs::read_to_string(&json_path).unwrap_or_default();
    let exported: Value = serde_json::from_str(&exported).unwrap_or_default();
    let agents = exported
        .get("agents")
        .and_then(Value::as_array)
        .map(Vec::len);
    assert_eq!(agents, Some(2), "{exported}");

    std::fs::remove_dir_all(&dir).ok();
}
