//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover their use.

use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

/// The provider id the shared adapter history's second call carries.
const SHARED: &str = "call_shared";

fn call(id: CallId) -> Message {
    Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction {
                name: crate::message::ToolName::new("test").expect("tool name"),
                arguments: serde_json::json!({}),
            },
        ))],
    }
}

fn result(id: CallId) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: id,
            name: crate::message::ToolName::new("possibly_repaired").expect("tool name"),
            content: vec![crate::message::ToolResultContent::text("")],
        })],
    }
}

fn provider(id: &str) -> CallId {
    CallId::from_wire(id)
}

fn local() -> CallId {
    CallId::from_wire("")
}

/// A history whose calls are answered out of order across text: a
/// rig-issued call, a provider call, and another rig-issued call a turn
/// later.
pub(crate) fn adapter_request() -> crate::completion::CompletionRequest {
    let id = local();
    let later = local();
    crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("system"),
            call(id.clone()),
            call(provider(SHARED)),
            result(provider(SHARED)),
            Message::assistant("intervening text"),
            result(id),
            call(later.clone()),
            result(later),
        ],
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
    assert_eq!(
        calls.iter().collect::<std::collections::HashSet<_>>().len(),
        3,
        "{wire}"
    );
}

pub(crate) fn adapter_requests() -> Vec<crate::completion::CompletionRequest> {
    vec![adapter_request()]
}

#[test]
fn wire_slot_assignment_is_atomic_and_uses_original_content_order() {
    let function = || {
        ToolFunction::new(
            crate::message::ToolName::new("test").expect("tool name"),
            serde_json::json!({}),
        )
    };
    let history = vec![Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::text("before"),
            AssistantContent::ToolCall(ToolCall::new(provider("first"), function())),
            AssistantContent::text("between"),
            AssistantContent::ToolCall(ToolCall::new(provider("second"), function())),
        ],
    }];
    let ids = WireIds::new(&history);
    for count in [1, 3] {
        let mut slots = vec!["unchanged".to_owned(); count];
        assert!(ids.apply(0, slots.iter_mut()).is_err());
        assert!(slots.iter().all(|slot| slot == "unchanged"));
    }
    let mut slots = [String::new(), String::new()];
    ids.apply(0, slots.iter_mut()).expect("two slots");
    assert_eq!(slots, ["first", "second"]);
}

#[test]
fn a_rig_issued_id_is_one_alias_that_no_provider_id_takes() {
    let issued = local();
    let history = vec![
        call(provider("tool-0")),
        call(issued.clone()),
        result(issued),
        call(local()),
    ];
    let ids = WireIds::new(&history);
    assert_eq!(ids.get(0, 0), Some("tool-0"));
    assert_eq!(ids.get(1, 0), Some("tool-1"));
    assert_eq!(ids.get(1, 0), ids.get(2, 0), "a result answers its call");
    assert_eq!(ids.get(3, 0), Some("tool-2"));
}

#[test]
fn a_provider_result_without_its_call_keeps_the_providers_id() {
    let history = vec![result(provider("real"))];
    assert_eq!(WireIds::new(&history).get(0, 0), Some("real"));
}
