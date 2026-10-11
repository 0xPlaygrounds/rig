use std::fs;
use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::{Value, json};

use super::*;

/// A `shell` that returns all of its command's output. The agent's own
/// `shell` (rig-tools) keeps only its last 2000 lines, so the first
/// lines of `seq 1 5000` would never reach this plugin through it.
struct WholeShell;

impl PortableTool for WholeShell {
    const NAME: &'static str = "shell";
    type Args = Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run a command with `sh -c`.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"],
        })
    }

    async fn call(&self, args: Value) -> Result<String, ToolExecutionError> {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(ToolExecutionError::from_error)?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// A spill directory beside the test binary, not in the system's temp.
fn spill_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.parent().map(PathBuf::from).unwrap_or_default();
    dir.join(format!("output-truncate-spill-{}", std::process::id()))
}

/// The text of every tool result in `messages`.
fn tool_results(messages: &[message::Message]) -> String {
    let mut text = String::new();
    for message in messages {
        if let message::Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::ToolResult(result) = item {
                    for part in &result.content {
                        text.push_str(part.as_text().unwrap_or_default());
                    }
                }
            }
        }
    }
    text
}

#[test]
fn a_long_shell_output_reaches_the_model_cut_and_is_kept_whole() {
    let dir = spill_dir();
    let config = TruncateConfig {
        spill_dir: dir.clone(),
        ..TruncateConfig::default()
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(config.clone())
        .add_plugins((AgentPlugin, JournalPlugin, OutputTruncatePlugin))
        .add_tool(WholeShell);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "shell", json!({ "command": "seq 1 5000" })),
        reply("Counted."),
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
    world.trigger(Deliver::new(agent, "Count.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| model.request_count() >= 2);

    // The second request carries the cut: head, marker, tail.
    let requests = model.requests();
    let sent = tool_results(
        &requests
            .get(1)
            .map(|r| r.chat_history.clone())
            .unwrap_or_default(),
    );
    let lines: Vec<&str> = sent.lines().collect();
    assert_eq!(lines.first(), Some(&"1"), "{sent}");
    assert!(lines.contains(&"150") && !lines.contains(&"151"), "{sent}");
    assert!(
        lines.contains(&"5000") && !lines.contains(&"2500"),
        "{sent}"
    );
    assert!(
        sent.contains("[… 4,750 lines omitted; full output in `"),
        "{sent}"
    );
    assert!(sent.contains("c1.txt` …]"), "{sent}");
    // The conversation, which the transcript shows, holds the same cut.
    let conversation = app.world().get::<Conversation>(agent);
    let kept = conversation
        .map(|c| tool_results(c.messages()))
        .unwrap_or_default();
    assert_eq!(kept, sent);
    // The spill file has every line.
    let spilled = fs::read_to_string(dir.join("c1.txt")).unwrap_or_default();
    assert_eq!(spilled.lines().count(), 5000);

    // An error is cut the same way and stays an error.
    let long: String = (1..=1000).map(|n| format!("error {n}\n")).collect();
    let failed = ToolResult::failed(ToolExecutionError::other(long));
    let cut = shorten(&failed, "c2", &config);
    assert!(cut.as_ref().is_some_and(ToolResult::is_error), "{cut:?}");
    let shown = cut.map(|cut| cut.output().render()).unwrap_or_default();
    assert!(
        shown.contains("lines omitted") && shown.contains("error 1000"),
        "{shown}"
    );

    fs::remove_dir_all(&dir).ok();
}
