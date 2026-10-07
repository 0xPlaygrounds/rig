use super::*;
use rig_core::message::{ToolFunction, ToolName, ToolResultContent, UserContent};
use serde_json::json;

fn call(id: &str) -> ToolCall {
    ToolCall::from_wire(
        id,
        ToolFunction::new(
            ToolName::new("add").expect("tool name"),
            json!({"patched": true}),
        ),
    )
}

fn result(id: &str) -> UserContent {
    UserContent::tool_result(
        CallId::from_wire(id),
        ToolName::new("add").expect("tool name"),
        vec![ToolResultContent::text("3")],
    )
}

fn results(ids: &[&str]) -> Vec<Message> {
    vec![Message::User {
        content: ids.iter().map(|id| result(id)).collect(),
    }]
}

fn kinds(items: ProjectedItems) -> Vec<String> {
    items
        .into_iter()
        .map(|item| match item {
            MultiTurnStreamItem::ToolExecutionCommitted { tool_call } => {
                format!("ran:{}", tool_call.id)
            }
            MultiTurnStreamItem::ToolResult { tool_result } => {
                format!("result:{}", tool_result.call)
            }
            MultiTurnStreamItem::ToolCall { tool_call } => format!("call:{}", tool_call.id),
            other => panic!("unexpected item {other:?}"),
        })
        .collect()
}

#[test]
fn an_execution_commit_precedes_the_result_of_the_call_whose_body_ran() {
    let items = committed_stream_items(&results(&["a", "b"]), &[call("b")]).expect("answered");
    assert_eq!(kinds(items), ["result:a", "ran:b", "result:b"]);
}

#[test]
fn an_execution_commit_pairs_with_its_result_by_id_not_position() {
    // Committed in the opposite order to the batch: the ran call still tags
    // its own result.
    let items = committed_stream_items(&results(&["b", "a"]), &[call("b")]).expect("answered");
    assert_eq!(kinds(items), ["ran:b", "result:b", "result:a"]);
}

#[test]
fn a_ran_call_no_committed_result_answers_is_refused() {
    assert_eq!(
        committed_stream_items(&results(&["a", "b"]), &[call("c")]).err(),
        Some(CallId::from_wire("c")),
        "a ran call must not tag some other call's result"
    );
}
