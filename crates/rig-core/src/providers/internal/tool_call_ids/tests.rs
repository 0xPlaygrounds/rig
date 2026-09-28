//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover use of this sidecar.

use super::*;
use crate::message::{CallId, ToolFunction, ToolResult};

/// The provider id the shared adapter history's second call carries.
const SHARED: &str = "call_shared";

fn call(id: CallId) -> Message {
    Message::Assistant {
        id: None,
        content: crate::NonEmpty::new(AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction {
                name: crate::message::ToolName::new("test").expect("tool name"),
                arguments: serde_json::json!({}),
            },
        ))),
    }
}

fn result(id: CallId) -> Message {
    Message::User {
        content: crate::NonEmpty::new(UserContent::ToolResult(ToolResult {
            call: id,
            name: crate::message::ToolName::new("possibly_repaired").expect("tool name"),
            content: crate::NonEmpty::new(crate::message::ToolResultContent::text("")),
        })),
    }
}

fn provider(id: &str) -> CallId {
    CallId::from_wire(id)
}

fn local() -> CallId {
    CallId::from_wire("")
}

/// A history whose calls are answered out of order across text: a
/// rig-issued call, a provider call, and the rig-issued id reused a turn
/// later.
pub(crate) fn adapter_request() -> crate::completion::CompletionRequest {
    let id = local();
    crate::completion::CompletionRequest {
        model: None,
        chat_history: crate::NonEmpty::with_rest(
            Message::system("system"),
            [
                call(id.clone()),
                call(provider(SHARED)),
                result(provider(SHARED)),
                Message::assistant("intervening text"),
                result(id.clone()),
                call(id.clone()),
                result(id),
            ],
        ),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(128),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

pub(crate) fn assert_adapter_pairs(wire: serde_json::Value) {
    fn visit(value: &serde_json::Value, calls: &mut Vec<String>, results: &mut Vec<String>) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    visit(value, calls, results);
                }
            }
            serde_json::Value::Object(object) => {
                let kind = object.get("type").and_then(serde_json::Value::as_str);
                let reference = match kind {
                    Some("function_call") => object
                        .get("call_id")
                        .or_else(|| object.get("id"))
                        .map(|id| (true, id)),
                    Some("tool_use") => object.get("id").map(|id| (true, id)),
                    Some("function_call_output" | "function_result") => {
                        object.get("call_id").map(|id| (false, id))
                    }
                    Some("tool_result") => object.get("tool_use_id").map(|id| (false, id)),
                    _ => None,
                };
                if let Some((is_call, id)) = reference {
                    let id = id.as_str().expect("wire identity is a string").to_owned();
                    if is_call {
                        calls.push(id);
                    } else {
                        results.push(id);
                    }
                }
                if object.get("role").and_then(serde_json::Value::as_str) == Some("tool") {
                    results.push(object["tool_call_id"].as_str().unwrap().to_owned());
                }
                if let Some(array) = object
                    .get("tool_calls")
                    .and_then(serde_json::Value::as_array)
                {
                    calls.extend(
                        array
                            .iter()
                            .map(|call| call["id"].as_str().unwrap().to_owned()),
                    );
                }
                for value in object.values() {
                    visit(value, calls, results);
                }
            }
            _ => {}
        }
    }
    let mut calls = Vec::new();
    let mut results = Vec::new();
    visit(&wire, &mut calls, &mut results);
    assert_eq!(calls.len(), 3, "{wire}");
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()],
        "{wire}"
    );
    assert_eq!(calls[1], SHARED);
    assert_eq!(calls.iter().collect::<HashSet<_>>().len(), 3, "{wire}");
}

pub(crate) fn adapter_requests() -> Vec<crate::completion::CompletionRequest> {
    vec![adapter_request()]
}

#[test]
fn wire_slot_assignment_is_atomic_and_uses_original_content_order() {
    let history = vec![Message::Assistant {
        id: None,
        content: crate::NonEmpty::with_rest(
            AssistantContent::text("before"),
            [
                AssistantContent::ToolCall(ToolCall::from_wire(
                    "first",
                    ToolFunction {
                        name: crate::message::ToolName::new("test").expect("tool name"),
                        arguments: serde_json::json!({}),
                    },
                )),
                AssistantContent::text("between"),
                AssistantContent::ToolCall(ToolCall::from_wire(
                    "second",
                    ToolFunction {
                        name: crate::message::ToolName::new("test").expect("tool name"),
                        arguments: serde_json::json!({}),
                    },
                )),
            ],
        ),
    }];
    let ids = ToolCallIds::new(&history).unwrap();
    for count in [1, 3] {
        let mut slots = vec!["unchanged".to_owned(); count];
        assert!(matches!(
            ids.apply(0, slots.iter_mut()),
            Err(ToolCallIdError::WireSlotCount { .. })
        ));
        assert!(slots.iter().all(|slot| slot == "unchanged"));
    }
    let mut slots = [String::new(), String::new()];
    ids.apply(0, slots.iter_mut()).unwrap();
    assert_eq!(slots, ["first", "second"]);
}

#[test]
fn a_reused_rig_issued_id_is_distinguished_across_turns() {
    let generated = local();
    let history = vec![
        call(generated.clone()),
        result(generated.clone()),
        call(generated.clone()),
        result(generated),
    ];
    let ids = ToolCallIds::new(&history).unwrap();
    assert_eq!(ids.get(0, 0), ids.get(1, 0));
    assert_eq!(ids.get(2, 0), ids.get(3, 0));
    assert_ne!(ids.get(0, 0), ids.get(2, 0));
}

#[test]
fn a_provider_result_without_its_call_keeps_the_providers_id() {
    let history = vec![result(provider("real"))];
    let ids = ToolCallIds::new(&history).unwrap();
    assert_eq!(ids.get(0, 0), Some("real"));
    assert!(matches!(
        ToolCallIds::new(&[result(local())]),
        Err(ToolCallIdError::OrphanResult { .. })
    ));
}

#[test]
fn allows_sequential_real_id_reuse_but_rejects_duplicate_answers() {
    let mut history = vec![
        call(provider("real")),
        result(provider("real")),
        call(provider("real")),
        result(provider("real")),
    ];
    let ids = ToolCallIds::new(&history).unwrap();
    assert!((0..4).all(|message| ids.get(message, 0) == Some("real")));
    history.push(result(provider("real")));
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::DuplicateResult { .. })
    ));
}

#[test]
fn overlapping_calls_with_one_provider_id_are_ambiguous() {
    let history = vec![call(provider("a")), call(provider("a"))];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::Ambiguous { .. })
    ));
}
