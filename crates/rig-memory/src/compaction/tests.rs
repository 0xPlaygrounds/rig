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
        upto: 2,
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
fn the_summary_leads_the_first_live_user_message() {
    let state = SummaryState {
        upto: 1,
        summary: "s".to_owned(),
        tracked: Vec::new(),
    };
    let request = state.request(&[Message::user("old"), Message::user("new")]);
    let [Message::User { content }] = request.as_slice() else {
        panic!("one user message: {request:?}");
    };
    assert_eq!(content.len(), 2);
}

#[test]
fn a_cut_never_separates_a_call_from_its_result() {
    let state = SummaryState::default();
    let messages = [
        Message::user("start"),
        call("1", "read", "a.rs"),
        result("1", "x"),
        Message::assistant("done"),
    ];
    let counter = HeuristicTokenCounter::default();
    assert_eq!(state.cut(&messages, 0, false, &counter), Some(3));
    assert_eq!(state.cut(&messages, usize::MAX, false, &counter), None);
    assert_eq!(state.cut(&messages, usize::MAX, true, &counter), Some(3));
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
