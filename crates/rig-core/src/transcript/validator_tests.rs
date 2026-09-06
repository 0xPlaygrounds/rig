use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: ToolCallId::new_or_minted(id, 0),
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
        call: ToolCallId::new_or_minted(id, 0),
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

/// Equal-looking IDs from separate namespaces answer only their own calls.
#[test]
fn typed_identity_transcripts_preserve_namespaces_and_completion_scope() {
    let generated = ToolCallId::minted(0);
    let explicit = ToolCallId::new("tool-0").expect("explicit ID");
    let call_for = |id: ToolCallId| {
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new("add".into(), serde_json::json!({})),
        ))
    };
    let result_for = |id: ToolCallId| {
        UserContent::ToolResult(ToolResult {
            call: id,
            provider: None,
            name: "add".into(),
            content: vec![ToolResultContent::text("3")],
        })
    };
    let history = vec![
        assistant(vec![
            call_for(generated.clone()),
            call_for(explicit.clone()),
        ]),
        Message::User {
            content: vec![result_for(explicit.clone()), result_for(generated.clone())],
        },
        assistant(vec![call_for(generated.clone())]),
        Message::User {
            content: vec![result_for(generated.clone())],
        },
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    let mismatched = vec![
        assistant(vec![call_for(generated)]),
        Message::User {
            content: vec![result_for(explicit)],
        },
    ];
    assert!(matches!(
        validate_canonical(&mismatched),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
}
