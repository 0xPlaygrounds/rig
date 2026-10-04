use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: crate::message::CallId::from_wire(id),
        function: ToolFunction::new(
            crate::message::ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
        native: None,
    })
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        is_error: false,
        call: crate::message::CallId::from_wire(id),
        name: crate::message::ToolName::new("add").expect("tool name"),
        content: vec![ToolResultContent::text("3")],
    })
}
fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(content))
}

#[test]
fn unanswered_and_orphan_results_are_rejected() {
    let unanswered = vec![assistant(vec![call("c1")]), Message::user("no result")];
    assert!(matches!(
        validate_canonical(&unanswered),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
    let orphan = vec![
        Message::user("hi"),
        Message::User {
            content: vec![result("ghost")],
        },
    ];
    assert!(matches!(
        validate_canonical(&orphan),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
    let trailing = vec![assistant(vec![call("c1")])];
    assert!(matches!(
        validate_canonical(&trailing),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
}
