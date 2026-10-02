//! rig#2269: items a stateless Responses client must send back unchanged.
//!
//! - `compaction` items reach history and go back verbatim;
//! - an output message's `phase` goes back on the item it came on, and each
//!   message item keeps its own id.

use super::*;
use crate::completion;
use serde_json::json;

/// The assistant turn a unary reply with `output` folds into.
fn history_of(output: serde_json::Value) -> completion::Message {
    let mut body = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5-mini",
        "tools": [],
    });
    body["output"] = output;
    crate::test_utils::decode_reply(
        &wire(),
        &crate::completion::CompletionRequest::new("hi"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(body.to_string())],
        serde_json::Value::Null,
    )
    .expect("the body decodes")
    .message()
    .expect("the reply has content")
}

fn wire() -> wire::Responses {
    crate::providers::openai::OpenAIConfig::new("key").responses("gpt-5-mini")
}

/// The `input` the OpenAI wire for the turn's own model sends for `turn`.
fn replayed(turn: completion::Message) -> Vec<serde_json::Value> {
    use crate::wire::{Operation, Wire};
    let wire = wire();
    let history = vec![completion::Message::user("hi"), turn];
    let request = crate::operation::Completion::prepare(
        crate::completion::CompletionRequest::from(history),
        &wire.describe(),
    )
    .expect("the history is valid");
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
        .expect("the request encodes");
    let mut input = crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    input.remove(0);
    input
}

fn message_item(id: &str, phase: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message", "id": id, "role": "assistant", "status": "completed", "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

#[test]
fn compaction_output_item_decodes_without_loss() {
    let wire = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes",
        "status": "completed",
        "future_field": {"nested": [1, 2, 3]}
    });
    let output: Output = serde_json::from_value(wire.clone()).expect("compaction decodes");
    assert_eq!(output, Output::Unknown(wire.clone()));
    assert_eq!(serde_json::to_value(&output).expect("it serializes"), wire);
}

/// The exact window `/responses/compact` returns — regular items around an
/// opaque compaction item — reaches history and goes back as it came.
#[test]
fn a_compacted_window_replays_every_item_verbatim() {
    let window = json!([
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "..."},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
         "arguments": "{}", "status": "completed"}
    ]);
    let turn = history_of(window.clone());
    let mut replayed = replayed(turn);
    // The call's synthetic answer follows the turn.
    let answer = replayed.pop().expect("the unanswered call gets a result");
    assert_eq!(answer["type"], "function_call_output");
    assert_eq!(json!(replayed), window);
}

#[test]
fn output_message_phase_decodes_and_is_absent_by_default() {
    let with: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [], "phase": "final_answer"
    }))
    .expect("decodes");
    assert_eq!(with.phase.as_deref(), Some("final_answer"));
    let without: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed", "content": []
    }))
    .expect("decodes");
    assert_eq!(without.phase, None);
    let back = serde_json::to_value(&without).expect("serializes");
    assert!(
        back.get("phase").is_none(),
        "absent phase must not serialize as null"
    );
}

/// A reply with a commentary message, a call and a final answer replays as
/// its items, each under its own id with its own `phase`, in the order the
/// reply stated them.
#[test]
fn each_message_item_replays_with_its_own_phase_and_id() {
    let output = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e"},
        message_item("msg_a", "commentary", "Checking."),
        message_item("msg_b", "final_answer", "Done."),
    ]);
    assert_eq!(json!(replayed(history_of(output.clone()))), output);
}
