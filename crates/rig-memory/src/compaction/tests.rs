use super::*;
use rig_core::message::{AssistantMessage, CallId, ToolCall, ToolFunction, ToolName, ToolResult};

fn call(id: &str, tool: &str, path: &str) -> Message {
    Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(
                ToolName::new(tool).expect("tool name"),
                rig_core::serde_json::json!({ "path": path }),
            ),
        ),
    )]))
}

fn result(id: &str, output: &str) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: CallId::from_wire(id),
            name: ToolName::new("read").expect("tool name"),
            content: vec![ToolResultContent::text(output)],
        })],
    }
}

#[test]
fn a_later_set_overrides_an_earlier_one_in_the_summary() {
    let mut state = SummaryState {
        summary: "did things".to_owned(),
        tracked: Vec::new(),
    };
    let messages = [
        call("1", "read", "a.rs"),
        call("2", "read", "b.rs"),
        call("3", "edit", "b.rs"),
    ];
    state.track(
        &messages,
        &[
            TrackArgument {
                tool: "read",
                argument: "path",
                set: "read-files",
            },
            TrackArgument {
                tool: "edit",
                argument: "path",
                set: "modified-files",
            },
        ],
    );
    let text = state.message().expect("a summary");
    assert!(text.contains("<read-files>\na.rs\n</read-files>"), "{text}");
    assert!(
        text.contains("<modified-files>\nb.rs\n</modified-files>"),
        "{text}"
    );
}

#[test]
fn a_cut_never_separates_a_call_from_its_result() {
    let messages = [
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", "x"),
        Message::assistant("done"),
    ];
    let counter = HeuristicTokenCounter::default();
    assert_eq!(cut_at(&messages, 0, 0, false, &counter), Some(3));
    assert_eq!(cut_at(&messages, 0, usize::MAX, false, &counter), None);
    assert_eq!(cut_at(&messages, 0, usize::MAX, true, &counter), Some(3));
}

fn spec(window: u32) -> Option<ModelSpec> {
    rig_core::providers::registry::ProviderId::catalog("ollama")
        .map(|vendor| ModelSpec::new(vendor, "test").with_context_window(window))
}

/// Where `reason` cuts `messages` for a model with `window` tokens.
fn cut(messages: &[Message], window: u32, reason: &CompactReason) -> Option<usize> {
    let spec = spec(window);
    assert!(spec.is_some());
    CompactionPolicy::default().cut(messages, 0, reason, spec.as_ref())
}

/// A conversation of turns, each a question and a long answer.
fn turns(count: usize) -> Vec<Message> {
    let answer = "word ".repeat(2_000);
    (0..count)
        .flat_map(|turn| {
            [
                Message::user(format!("question {turn}")),
                Message::assistant(answer.clone()),
            ]
        })
        .collect()
}

#[test]
fn a_compaction_keeps_the_recent_work_unless_asked() {
    let (short, long) = (turns(2), turns(20));
    let asked = CompactReason::Asked {
        focus: String::new(),
    };
    // Asked, or refused as too long, a short conversation is summarized
    // all but its newest reply; near the window, it is left alone.
    assert_eq!(cut(&short, 200_000, &asked), Some(short.len() - 1));
    let overflow = CompactReason::Overflow;
    assert_eq!(cut(&short, 200_000, &overflow), Some(short.len() - 1));
    assert_eq!(cut(&short, 200_000, &CompactReason::Threshold), None);
    // Otherwise the newest 20k tokens stay, a quarter of the window at most.
    let at = cut(&long, 200_000, &overflow);
    assert!(at.is_some_and(|at| at < long.len() - 2), "{at:?}");
    let policy = CompactionPolicy::default();
    let keep = |window| policy.keep(&CompactReason::Threshold, spec(window).as_ref());
    assert_eq!((keep(200_000), keep(40_000)), (20_000, 10_000));
}

#[test]
fn clearing_keeps_the_newest_outputs_and_the_last_message() {
    let big = "x".repeat(4_000);
    let mut messages = vec![
        call("1", "read", "a.rs"),
        result("1", &big),
        call("2", "read", "b.rs"),
        result("2", &big),
        result("3", &big),
    ];
    let policy = ClearToolOutputs::new(1_500);
    let cleared = policy.clear(&mut messages);
    assert_eq!(cleared.results, 1);
    assert_eq!(cleared.tokens, 1_000);
    assert!(policy.is_cleared(match &messages[1] {
        Message::User { content } => match &content[0] {
            UserContent::ToolResult(result) => &result.content,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }));
    let again = policy.apply(messages).expect("clearing never fails");
    assert_eq!(policy.clear(&mut again.clone()), Cleared::default());
}
