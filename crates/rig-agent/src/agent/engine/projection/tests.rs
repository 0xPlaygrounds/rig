use super::*;
use rig_core::message::{CallId, ToolFunction, ToolName, ToolResultContent, UserContent};
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
fn an_execution_commit_precedes_the_result_of_the_slot_whose_body_ran() {
    let items =
        committed_stream_items(&results(&["a", "b"]), &[None, Some(call("b"))]).expect("answered");
    assert_eq!(kinds(items), ["result:a", "ran:b", "result:b"]);
}

#[test]
fn a_repeated_id_tags_the_result_of_the_call_that_ran_not_its_skipped_twin() {
    // One batch answers provider id `x` twice: the first `x` was skipped
    // (hook or malformed arguments), the second ran. Only the second result
    // carries the execution commit.
    let items =
        committed_stream_items(&results(&["x", "x"]), &[None, Some(call("x"))]).expect("answered");
    assert_eq!(kinds(items), ["result:x", "ran:x", "result:x"]);
}

#[test]
fn a_projection_outside_a_tool_batch_tags_nothing() {
    let items = committed_stream_items(&results(&["a"]), &[]).expect("projected");
    assert_eq!(kinds(items), ["result:a"]);
}

#[test]
fn a_batch_that_does_not_line_up_with_the_committed_results_is_refused() {
    assert!(
        committed_stream_items(&results(&["a", "b"]), &[Some(call("c")), None]).is_err(),
        "a ran call must not tag some other call's result"
    );
    assert!(
        committed_stream_items(&results(&["a"]), &[None, Some(call("b"))]).is_err(),
        "a slot with no committed result is refused"
    );
    assert!(
        committed_stream_items(&results(&["a", "b"]), &[None]).is_err(),
        "a committed result with no slot is refused"
    );
}
