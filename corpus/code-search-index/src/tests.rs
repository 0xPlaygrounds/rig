use std::path::PathBuf;
use std::process::Command;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::EditTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::json;

use super::*;

/// Each tool call's name and result text, in the order they ended, and
/// whether the turn ended.
#[derive(Resource, Default)]
struct Seen {
    outputs: Vec<(String, String)>,
    ended: bool,
}

const UI: &str = "pub fn draw_frame(canvas: &mut Canvas) {\n    canvas.clear();\n    \
                  canvas.draw_border();\n}\n";

/// A git project in the system's temporary directory with networking code
/// in `net.rs` and drawing code in `ui.rs`.
fn project() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("corpus-code-search-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let dir = std::fs::canonicalize(dir)?;
    let git = Command::new("git")
        .arg("init")
        .arg("-q")
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
    std::fs::write(dir.join("ui.rs"), UI)?;
    Ok(dir)
}

/// The file of a `code_search` result's first hit.
fn first_hit(result: &str) -> Option<&str> {
    let line = result.lines().find(|line| line.contains(".rs:"))?;
    line.split(':').next()
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
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(CodeSearchConfig {
            root: project.clone(),
            embedder: Embedder::Hashing,
            context_chunks: 3,
        })
        .add_plugins((AgentPlugin, JournalPlugin, EditTool, CodeSearchPlugin))
        .init_resource::<Seen>()
        .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.ended = true)
        .add_observer(
            |add: On<Add<ToolOutput>>,
             outputs: Query<(&ToolCallRun, &ToolOutput)>,
             mut seen: ResMut<Seen>| {
                if let Ok((run, ToolOutput(result))) = outputs.get(add.entity) {
                    let text = result.output().as_text().unwrap_or_default().to_owned();
                    let name = run.call.function.name.as_str().to_owned();
                    seen.outputs.push((name, text));
                }
            },
        );
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
        calls("3", "code_search", json!({ "query": "backoff delay" })),
        reply("done"),
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
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| world.resource::<Seen>().ended);
    // The index was saved once built.
    let saved = project.join(INDEX_FILE).exists();
    std::fs::remove_dir_all(&project).ok();

    let outputs = &app.world().resource::<Seen>().outputs;
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
    // Every request of the turn carries the chunks found for the user's
    // message; the conversation does not keep them.
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 4, "{sent:?}");
    assert!(
        sent.iter()
            .all(|request| request.contains("<code-context>"))
    );
    assert!(sent.iter().all(|request| request.contains("net.rs:")));
    let world = app.world_mut();
    let kept = world
        .query::<&Conversation>()
        .iter(world)
        .next()
        .map(|conversation| serde_json::to_string(conversation.messages()).unwrap_or_default());
    assert!(kept.is_some_and(|kept| !kept.contains("<code-context>")));
}
