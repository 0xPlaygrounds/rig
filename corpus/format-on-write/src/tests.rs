use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::json;

use super::*;

/// A directory beside the test binary, not in the system's temp, gone when
/// dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let exe = std::env::current_exe().unwrap_or_default();
        let dir = exe.parent().map(PathBuf::from).unwrap_or_default();
        let dir = dir.join(format!("format-on-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The spec's settings: `.txt` files are uppercased, and a line longer
/// than 10 characters is too long.
const SETTINGS: &str = r#"
[format_on_write.formatters]
txt = '''sh -c 'tr a-z A-Z < "$1" > "$1.up" && mv "$1.up" "$1"' sh'''

[format_on_write.linters]
txt = '''awk 'length > 10 { print "W: line too long" }' '''
"#;

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
    let scratch = Scratch::new();
    let file = scratch.0.join("a.txt").display().to_string();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(SETTINGS))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ReadTool,
            WriteTool,
            FormatOnWritePlugin,
        ));
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
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    world.trigger(Deliver::new(agent, "Write a.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| model.request_count() >= 3);

    let written = results_of(&model, 1);
    let read = results_of(&model, 2);
    assert!(written.contains("(formatted with sh)"), "{written}");
    assert!(written.contains("1\tHELLO WORLD WIDE"), "{written}");
    assert!(written.contains("W: line too long"), "{written}");
    assert!(read.contains("\tHELLO WORLD WIDE"), "{read}");
}
