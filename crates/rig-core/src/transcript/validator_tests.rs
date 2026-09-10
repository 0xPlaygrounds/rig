use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: ToolCallId::new_or_mint(id),
        provider: None,
        function: ToolFunction {
            name: "add".into(),
            arguments: serde_json::json!({}),
        },
        additional_params: None,
        signature: None,
    })
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: ToolCallId::new_or_mint(id),
        provider: None,
        name: "add".into(),
        content: vec![ToolResultContent::text("3")],
    })
}
fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant { id: None, content }
}

#[test]
fn canonical_transcripts_pass() {
    let history = vec![
        Message::user("hi"),
        assistant(vec![call("c1")]),
        Message::User {
            content: vec![result("c1")],
        },
        assistant(vec![AssistantContent::text("done")]),
        Message::user("thanks"),
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    assert!(validate_canonical(&[]).is_ok());
}

#[test]
fn consecutive_assistant_is_rejected() {
    let history = vec![
        assistant(vec![AssistantContent::text("a")]),
        assistant(vec![AssistantContent::text("b")]),
    ];
    assert_eq!(
        validate_canonical(&history),
        Err(TranscriptError::ConsecutiveAssistant { index: 1 })
    );
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
