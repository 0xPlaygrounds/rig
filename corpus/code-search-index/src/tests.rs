use std::path::PathBuf;
use std::process::Command;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::EditTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};
use serde_json::json;

use super::*;

/// Each tool call's name and result text, in the order they settled.
#[derive(Resource, Default)]
struct Seen(Vec<(String, String)>);

fn seen(settled: On<Add<Settled>>, calls: Query<(&ToolRequest, &Outcome)>, mut seen: ResMut<Seen>) {
    if let Ok((request, outcome)) = calls.get(settled.entity) {
        let result = outcome.tool_result();
        let text = result.output().as_text().unwrap_or_default().to_owned();
        seen.0
            .push((request.call.function.name.as_str().to_owned(), text));
    }
}

/// A git project in the system's temporary directory with networking code
/// in `net.rs` and drawing code in `ui.rs`.
fn project() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("corpus-code-search-{}", std::process::id()));
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
        .add_plugins((AgentPlugin, JournalPlugin, EditTool, CodeSearchPlugin))
        .init_resource::<Seen>()
        .init_resource::<Ended>()
        .add_observer(seen)
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
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
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
    let outputs = &app.world().resource::<Seen>().0;
    assert_eq!(outputs.len(), 3, "{outputs:?}");
    let [(first, retry), (edit, _), (second, backoff)] = outputs.as_slice() else {
        return;
    };
    assert_eq!(
        (first.as_str(), edit.as_str(), second.as_str()),
        ("code_search", "edit", "code_search")
    );
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
