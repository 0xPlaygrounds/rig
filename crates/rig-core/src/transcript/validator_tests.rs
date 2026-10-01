use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};
use crate::tool::ToolOutput;

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: crate::message::CallId::from_wire(id),
        function: ToolFunction {
            name: crate::message::ToolName::new("add").expect("tool name"),
            arguments: serde_json::json!({}),
        },
        additional_params: None,
        signature: None,
    })
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: crate::message::CallId::from_wire(id),
        name: crate::message::ToolName::new("add").expect("tool name"),
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

/// A result answers only the call whose id it carries.
#[test]
fn a_result_answers_only_its_own_call() {
    let (first, second) = (
        crate::message::CallId::from_wire(""),
        crate::message::CallId::from_wire(""),
    );
    let tool = crate::message::ToolName::new("add").expect("tool name");
    let call_for = |id: crate::message::CallId| {
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(tool.clone(), serde_json::json!({})),
        ))
    };
    let result_for = |id: crate::message::CallId| {
        UserContent::tool_result(id, tool.clone(), vec![ToolResultContent::text("3")])
    };
    let history = vec![
        assistant(vec![call_for(first.clone()), call_for(second.clone())]),
        Message::User {
            content: vec![result_for(second.clone()), result_for(first.clone())],
        },
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    let mismatched = vec![
        assistant(vec![call_for(first)]),
        Message::User {
            content: vec![result_for(second)],
        },
    ];
    assert!(matches!(
        validate_canonical(&mismatched),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
}

fn user(content: Vec<UserContent>) -> Message {
    Message::User { content }
}
fn placeholder(id: &str) -> UserContent {
    tool_result_output(
        crate::message::CallId::from_wire(id),
        crate::message::ToolName::new("add").expect("tool name"),
        ToolOutput::text("interrupted"),
    )
}
fn interrupted(_: &ToolCall) -> ToolOutput {
    ToolOutput::text("interrupted")
}
/// Repairs `history` and checks that the result is canonical and that a
/// second pass changes nothing.
fn repair(mut history: Vec<Message>) -> (Vec<Message>, usize) {
    let added = answer_unanswered(&mut history, interrupted);
    assert_eq!(validate_canonical(&history), Ok(()));
    let repaired = history.clone();
    assert_eq!(answer_unanswered(&mut history, interrupted), 0);
    assert_eq!(history, repaired);
    (history, added)
}

#[test]
fn a_fully_unanswered_turn_gets_a_new_result_message() {
    let mut seen = Vec::new();
    let mut history = vec![
        Message::user("hi"),
        assistant(vec![
            AssistantContent::text("checking"),
            call("c1"),
            call("c2"),
        ]),
    ];
    let added = answer_unanswered(&mut history, |call| {
        seen.push(call.id.clone());
        ToolOutput::text("interrupted")
    });
    assert_eq!(added, 2);
    assert_eq!(
        seen,
        [
            crate::message::CallId::from_wire("c1"),
            crate::message::CallId::from_wire("c2"),
        ]
    );
    assert_eq!(
        history[2..],
        [user(vec![placeholder("c1"), placeholder("c2")])]
    );
    assert_eq!(repair(history).1, 0);
}

#[test]
fn existing_results_survive_and_missing_ones_follow_call_order() {
    let history = vec![
        assistant(vec![call("c1"), call("c2"), call("c3")]),
        user(vec![result("c2")]),
    ];
    let (history, added) = repair(history);
    assert_eq!(added, 2);
    assert_eq!(
        history[1],
        user(vec![placeholder("c1"), result("c2"), placeholder("c3")])
    );
}

#[test]
fn placeholders_go_before_text() {
    let text = UserContent::text("keep going");
    let history = vec![
        assistant(vec![call("c1"), call("c2")]),
        user(vec![result("c1"), text.clone()]),
    ];
    let (history, added) = repair(history);
    assert_eq!(added, 1);
    assert_eq!(
        history[1],
        user(vec![result("c1"), placeholder("c2"), text.clone()])
    );

    let (history, _) = repair(vec![assistant(vec![call("c1")]), user(vec![text.clone()])]);
    assert_eq!(history[1], user(vec![placeholder("c1"), text]));
}

#[test]
fn unanswered_calls_in_earlier_turns_are_answered() {
    let history = vec![
        Message::user("hi"),
        assistant(vec![call("c1")]),
        Message::user("never mind"),
        assistant(vec![call("c2")]),
        Message::system("context"),
        assistant(vec![call("c3")]),
        user(vec![result("c3")]),
        assistant(vec![AssistantContent::text("done")]),
    ];
    let (history, added) = repair(history);
    assert_eq!(added, 2);
    assert_eq!(
        history,
        vec![
            Message::user("hi"),
            assistant(vec![call("c1")]),
            user(vec![placeholder("c1"), UserContent::text("never mind")]),
            assistant(vec![call("c2")]),
            user(vec![placeholder("c2")]),
            Message::system("context"),
            assistant(vec![call("c3")]),
            user(vec![result("c3")]),
            assistant(vec![AssistantContent::text("done")]),
        ]
    );
}

/// System messages between a call and its results do not hide the results.
#[test]
fn results_after_a_system_message_are_kept() {
    let history = vec![
        assistant(vec![call("c1"), call("c2")]),
        Message::system("context"),
        user(vec![result("c1")]),
    ];
    let (history, added) = repair(history);
    assert_eq!(added, 1);
    assert_eq!(history.len(), 3);
    assert_eq!(history[2], user(vec![result("c1"), placeholder("c2")]));
}

#[test]
fn a_canonical_history_is_unchanged() {
    let history = vec![
        Message::user("hi"),
        assistant(vec![call("c1"), call("c1")]),
        user(vec![result("c1")]),
        assistant(vec![AssistantContent::text("done")]),
    ];
    let mut repaired = history.clone();
    let added = answer_unanswered(&mut repaired, |_| {
        panic!("a canonical history has no unanswered calls")
    });
    assert_eq!(added, 0);
    assert_eq!(repaired, history);
    assert_eq!(repair(Vec::new()), (Vec::new(), 0));
}

/// Other violations are left for the caller.
#[test]
fn orphan_results_are_not_repaired() {
    let mut history = vec![assistant(vec![call("c1")]), user(vec![result("ghost")])];
    assert_eq!(answer_unanswered(&mut history, interrupted), 1);
    assert_eq!(history[1], user(vec![result("ghost"), placeholder("c1")]));
    assert!(matches!(
        validate_canonical(&history),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
}
