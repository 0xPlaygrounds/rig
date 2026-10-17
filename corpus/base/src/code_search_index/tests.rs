use std::path::PathBuf;
use std::process::Command;

use rig_basics::coding_tools::EditTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until,
};
use serde_json::json;

use super::*;

/// A git project beside the test's binary with networking code in
/// `net.rs` and drawing code in `ui.rs`.
fn project() -> std::io::Result<PathBuf> {
    let dir = std::env::current_exe()?.with_file_name("code-search-project");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir)?;
    let dir = std::fs::canonicalize(dir)?;
    let git = Command::new("git")
        .args(["init", "-q"])
        .current_dir(&dir)
        .status()?;
    if !git.success() {
        return Err(std::io::Error::other("git init failed"));
    }
    std::fs::write(
        dir.join("net.rs"),
        "/// Sends the request again when the server is busy.\n\
         pub fn retry_request(client: &Client, request: Request) -> Response {\n    \
         loop {\n        if let Ok(response) = client.send(request.clone()) {\n            \
         return response;\n        }\n    }\n}\n",
    )?;
    std::fs::write(
        dir.join("ui.rs"),
        "pub fn draw_frame(canvas: &mut Canvas) {\n    canvas.clear();\n    \
         canvas.draw_border();\n}\n",
    )?;
    Ok(dir)
}

/// The file of a `code_search` result's first hit.
fn first_hit(result: &str) -> Option<&str> {
    let line = result.lines().find(|line| line.contains(".rs:"))?;
    line.split(':').next()
}

/// Runs frames until `count` turns ended.
fn turns_ended(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= count
    });
}

#[test]
fn search_finds_code_by_meaning_sees_an_edit_and_adds_context() {
    let project = project();
    assert!(project.is_ok(), "{project:?}");
    let Ok(project) = project else {
        return;
    };
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Settings::from_toml("[code_search]\ncontext_chunks = 3"))
        .insert_resource(ProjectRoot(project.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, EditTool, CodeSearchPlugin))
        .init_resource::<ToolOutputs>()
        .init_resource::<Ended>()
        .add_observer(log_tool_outputs)
        .add_observer(log_ended);
    app.finish();
    app.update();
    run_until(&mut app, &wakes, |world| {
        let index = world.resource::<CodeIndex>();
        index.is_built() && index.size().0 == 2
    });
    assert_eq!(app.world().resource::<CodeIndex>().size().0, 2);

    let ui = project.join("ui.rs");
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "code_search", json!({ "query": "retry request" })),
        calls(
            "2",
            "edit",
            json!({ "path": ui.to_string_lossy(), "edits": [{
                "old_text": "pub fn draw_frame",
                "new_text": "pub fn backoff_delay(attempt: u32) -> u64 {\n    100 << attempt\n}\n\npub fn draw_frame",
            }] }),
        ),
        reply("done"),
        calls("3", "code_search", json!({ "query": "backoff delay" })),
        reply("found"),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world);
    let connection = connect(&model);
    assert!(agent.is_some() && connection.is_some());
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(
        agent,
        "Find the retry code.",
        DeliveryMode::Queue,
    ));
    turns_ended(&mut app, &wakes, 1);
    // The edited file is indexed again from the call's `FilesChanged`,
    // before any search looks at its time.
    run_until(&mut app, &wakes, |world| {
        world
            .resource::<CodeIndex>()
            .contains("ui.rs", "fn backoff_delay")
    });
    let reindexed = app
        .world()
        .resource::<CodeIndex>()
        .contains("ui.rs", "fn backoff_delay");
    app.world_mut().trigger(Deliver::new(
        agent,
        "Where is the delay?",
        DeliveryMode::Queue,
    ));
    turns_ended(&mut app, &wakes, 2);
    let saved = project.join(INDEX_FILE).exists();
    std::fs::remove_dir_all(&project).ok();

    assert!(reindexed, "the edit re-indexed ui.rs");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(outputs.len(), 3, "{outputs:?}");
    let output = |id: &str| outputs.get(id).map(String::as_str).unwrap_or_default();
    let (retry, backoff) = (output("1"), output("3"));
    assert_eq!(first_hit(retry), Some("net.rs"), "{retry}");
    assert_eq!(first_hit(backoff), Some("ui.rs"), "{backoff}");
    assert!(backoff.contains("fn backoff_delay"), "{backoff}");
    assert!(saved, "the index is kept in .rig/index/");
    // Every request carries the chunks found for its user message; the
    // conversation does not keep them.
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 5, "{sent:?}");
    assert!(
        sent.iter()
            .all(|request| request.contains("<code-context>"))
    );
    assert!(
        sent.iter()
            .take(3)
            .all(|request| request.contains("net.rs:"))
    );
    let world = app.world_mut();
    let kept = world
        .query::<&Conversation>()
        .iter(world)
        .next()
        .map(|conversation| serde_json::to_string(conversation.messages()).unwrap_or_default());
    assert!(kept.is_some_and(|kept| !kept.contains("<code-context>")));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CodeSearchPlugin);
}
