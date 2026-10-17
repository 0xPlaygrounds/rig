use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use rig_basics::coding_tools::ShellTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_core::message::{self, UserContent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::json;

use super::*;

/// A project directory beside the test binary, not in the system's temp.
fn project() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.parent().map(PathBuf::from).unwrap_or_default();
    dir.join(format!("output-truncate-{}", std::process::id()))
}

/// The text of every tool result in `messages`.
fn tool_results(messages: &[impl std::borrow::Borrow<message::Message>]) -> String {
    let mut text = String::new();
    for message in messages
        .iter()
        .map(std::borrow::Borrow::<message::Message>::borrow)
    {
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
    let root = project();
    let spills = root.join(".rig/spill");
    // A spill file of last week, which startup sweeps away.
    fs::create_dir_all(&spills).ok();
    let stale = spills.join("old.txt");
    let file = fs::File::create(&stale);
    let eight_days = SystemTime::now() - Duration::from_secs(8 * 24 * 60 * 60);
    assert!(file.and_then(|file| file.set_modified(eight_days)).is_ok());

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(root.clone())).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ShellTool,
        OutputTruncatePlugin,
    ));
    app.finish();
    app.update();
    assert!(!stale.exists());

    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "shell", json!({ "command": "seq 1 5000" })),
        reply("Counted."),
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
    world.trigger(Deliver::new(agent, "Count.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| model.request_count() >= 2);

    // The second request carries the cut: head, marker, tail.
    let requests = model.requests();
    let history = requests.get(1).map(|r| r.chat_history.clone());
    let sent = tool_results(&history.unwrap_or_default());
    let lines: Vec<&str> = sent.lines().collect();
    assert_eq!(lines.first(), Some(&"1"), "{sent}");
    assert!(lines.contains(&"150") && !lines.contains(&"151"), "{sent}");
    assert!(
        lines.contains(&"5000") && !lines.contains(&"2500"),
        "{sent}"
    );
    assert!(
        sent.contains("[… 4,751 lines omitted; full output in `.rig/spill/c1.txt` …]"),
        "{sent}"
    );
    // The conversation, which the transcript shows, holds the same cut.
    let conversation = app.world().get::<Conversation>(agent);
    let kept = conversation.map(|c| tool_results(c.messages()));
    assert_eq!(kept.unwrap_or_default(), sent);
    // The spill file has every line, then how the command ended.
    let spilled = fs::read_to_string(spills.join("c1.txt")).unwrap_or_default();
    let numbers: Vec<&str> = spilled.lines().take(5000).collect();
    let expected: Vec<String> = (1..=5000).map(|n| n.to_string()).collect();
    assert_eq!(numbers, expected);

    // An error is cut the same way and stays an error.
    let long: String = (1..=1000).map(|n| format!("error {n}\n")).collect();
    let failed = ToolResult::failed(ToolExecutionError::other(long));
    let cut = shorten(&failed, "c2", &Truncate::default(), &root);
    assert!(cut.as_ref().is_some_and(ToolResult::is_error), "{cut:?}");
    let shown = cut.map(|cut| cut.output().render()).unwrap_or_default();
    assert!(
        shown.contains("lines omitted") && shown.contains("error 1000"),
        "{shown}"
    );

    fs::remove_dir_all(&root).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(OutputTruncatePlugin);
}
