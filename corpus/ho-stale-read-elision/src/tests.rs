use std::collections::HashMap;
use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ReadTool};
use rig_core::completion::CompletionRequest;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until,
};
use serde_json::json;

use super::*;

/// Each tool result of `messages` as text, by its call's wire id.
fn results(messages: &[Message]) -> HashMap<String, String> {
    let content = messages.iter().filter_map(|message| match message {
        Message::User { content } => Some(content),
        _ => None,
    });
    let results = content.flatten().filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(result),
        _ => None,
    });
    results
        .map(|result| {
            let text = result.content.iter().filter_map(ToolResultContent::as_text);
            (result.call.wire().into_owned(), text.collect())
        })
        .collect()
}

/// Each tool call's wire id in `messages`.
fn call_ids(messages: &[Message]) -> Vec<String> {
    let replies = messages.iter().filter_map(|message| match message {
        Message::Assistant(reply) => Some(&reply.content),
        _ => None,
    });
    let calls = replies.flatten().filter_map(|block| match block {
        AssistantContent::ToolCall(call) => Some(call.id.wire().into_owned()),
        _ => None,
    });
    calls.collect()
}

/// Turn 1 reads `a.rs` twice and `b.rs` once, then edits `b.rs`; turn 2
/// reads `a.rs` again. Turn 2's requests stub the superseded reads of turn
/// 1 and nothing else, while the conversation keeps every result whole.
#[test]
fn superseded_reads_go_out_as_stubs_and_stay_whole_in_the_conversation() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-project");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let (a, b) = (dir.join("a.rs"), dir.join("b.rs"));
    let body = "// filler line\n".repeat(50);
    std::fs::write(&a, format!("fn a() {{}} // A-BODY\n{body}")).ok();
    std::fs::write(&b, format!("fn b() {{}} // B-BODY\n{body}")).ok();
    let (a, b) = (a.to_string_lossy(), b.to_string_lossy());
    let edit = json!({ "path": b, "edits": [{ "old_text": "B-BODY", "new_text": "B-NEW" }] });
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": a })),
        calls("r2", "read", json!({ "path": a })),
        calls("r3", "read", json!({ "path": b })),
        calls("e1", "edit", edit),
        reply("Done."),
        calls("r4", "read", json!({ "path": a })),
        reply("Done again."),
    ]);

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        ReadTool,
        EditTool,
        StaleReadElisionPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    for (message, turns) in [("Look at a and b.", 1), ("Check a again.", 2)] {
        let message = Deliver::new(agent, message, DeliveryMode::Queue);
        app.world_mut().trigger(message);
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == turns
        });
    }
    std::fs::remove_dir_all(&dir).ok();

    let requests = model.requests();
    assert_eq!(requests.len(), 7, "five requests in turn 1, two in turn 2");
    let history = |at: usize| {
        let request = requests.get(at).map(|request: &CompletionRequest| request);
        request
            .map(|request| request.chat_history.clone())
            .unwrap_or_default()
    };
    let first = history(5);
    let sent = results(&first);
    let result = |id: &str| sent.get(id).cloned().unwrap_or_default();
    assert_eq!(
        result("r1"),
        format!("[superseded: {a} was read again later]")
    );
    assert!(result("r2").contains("A-BODY"), "{}", result("r2"));
    assert_eq!(
        result("r3"),
        format!("[superseded: {b} was changed after this read]")
    );
    assert!(!result("e1").starts_with("[superseded"), "{}", result("e1"));
    for id in call_ids(&first) {
        assert!(sent.contains_key(&id), "the call {id} keeps its result");
    }

    // The conversation still holds every result whole.
    let kept = app.world().get::<Conversation>(agent);
    let kept = results(
        kept.map(|conversation| conversation.messages())
            .unwrap_or_default(),
    );
    let kept = |id: &str| kept.get(id).cloned().unwrap_or_default();
    assert!(kept("r1").contains("A-BODY"), "{}", kept("r1"));
    assert!(kept("r3").contains("B-BODY"), "{}", kept("r3"));

    // Turn 2's own read stays whole; it supersedes turn 1's second read.
    let next = results(&history(6));
    let next = |id: &str| next.get(id).cloned().unwrap_or_default();
    assert!(next("r4").contains("A-BODY"), "{}", next("r4"));
    assert_eq!(
        next("r2"),
        format!("[superseded: {a} was read again later]")
    );
    let elided = app
        .world()
        .get::<Elided>(agent)
        .copied()
        .unwrap_or_default();
    assert_eq!(elided.results, 3);
    assert!(elided.tokens > 0);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(StaleReadElisionPlugin);
}
