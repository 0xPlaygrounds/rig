use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_tools::Read;
use serde_json::json;

use super::*;

/// A directory beside the test binary, not in the system's temp.
fn scratch() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.parent().map(PathBuf::from).unwrap_or_default();
    dir.join(format!("format-on-write-{}", std::process::id()))
}

/// `sh -c script`, the file's path as `$1`.
fn script(script: &str) -> Vec<String> {
    ["sh", "-c", script, "sh"].map(str::to_owned).to_vec()
}

/// The text of each tool result in the request `index` the model got.
fn results_of(model: &MockCompletionModel, index: usize) -> String {
    let requests = model.requests();
    let history = requests
        .get(index)
        .map(|request| request.chat_history.clone());
    let mut text = String::new();
    if let Some(message::Message::User { content }) = history.unwrap_or_default().last() {
        for item in content.iter() {
            if let UserContent::ToolResult(result) = item {
                for part in &result.content {
                    text.push_str(part.as_text().unwrap_or_default());
                }
            }
        }
    }
    text
}

#[test]
fn a_write_is_formatted_and_linted_before_the_next_read() {
    let dir = scratch();
    std::fs::create_dir_all(&dir).ok();
    let file = dir.join("a.txt").display().to_string();
    let config = FormatOnWrite {
        formatters: HashMap::from([(
            "txt".to_owned(),
            script(r#"tr a-z A-Z < "$1" > "$1.up" && mv "$1.up" "$1""#),
        )]),
        linters: HashMap::from([(
            "txt".to_owned(),
            script(r#"awk 'length > 10 { print "W: line too long" }' "$1""#),
        )]),
        timeout: Duration::from_secs(10),
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(config)
        .add_plugins((AgentPlugin, JournalPlugin, FormatOnWritePlugin))
        .add_tool_with(Write, ToolOptions::default())
        .add_tool_with(Read, ToolOptions::default());
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "w1",
            "write",
            json!({ "path": file, "content": "hello world wide" }),
        ),
        calls("r1", "read", json!({ "path": file })),
        reply("Done."),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(agent, "Write a.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| model.request_count() >= 3);

    let written = results_of(&model, 1);
    let read = results_of(&model, 2);
    std::fs::remove_dir_all(&dir).ok();
    assert!(written.contains("(formatted with sh)"), "{written}");
    assert!(written.contains("1\tHELLO WORLD WIDE"), "{written}");
    assert!(written.contains("W: line too long"), "{written}");
    assert!(read.contains("\tHELLO WORLD WIDE"), "{read}");
}
