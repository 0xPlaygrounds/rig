//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover use of this sidecar.

use super::*;
use crate::message::{ProviderCallId, ToolCallId, ToolFunction, ToolResult};

fn call(id: ToolCallId, provider: Option<&str>) -> Message {
    Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall {
            id,
            provider: provider.and_then(ProviderCallId::new),
            function: ToolFunction {
                name: "test".into(),
                arguments: serde_json::json!({}),
            },
            signature: None,
            additional_params: None,
        })],
    }
}

fn result(id: ToolCallId, provider: Option<&str>) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: id,
            provider: provider.and_then(ProviderCallId::new),
            name: "possibly_repaired".into(),
            content: vec![],
        })],
    }
}

fn explicit(id: &str) -> ToolCallId {
    ToolCallId::new(id).unwrap()
}

pub(crate) fn adapter_request() -> crate::completion::CompletionRequest {
    let id = ToolCallId::minted(0);
    let real = id.wire_hint().into_owned();
    crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("system"),
            call(id.clone(), None),
            call(explicit(&real), Some(&real)),
            result(explicit(&real), Some(&real)),
            Message::assistant("intervening text"),
            result(id.clone(), None),
            call(id.clone(), None),
            result(id, None),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(128),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
        observation: None,
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
    assert_eq!(calls[1], ToolCallId::minted(0).wire_hint());
    assert_eq!(calls.iter().collect::<HashSet<_>>().len(), 3, "{wire}");
}

pub(crate) fn adapter_requests() -> Vec<crate::completion::CompletionRequest> {
    let full = adapter_request();
    let mut call_only = full.clone();
    let Message::User { content } = &mut call_only.chat_history[3] else {
        panic!("result message");
    };
    let UserContent::ToolResult(result) = &mut content[0] else {
        panic!("result content");
    };
    result.provider = None;
    let mut result_only = full.clone();
    let Message::Assistant { content, .. } = &mut result_only.chat_history[2] else {
        panic!("call message");
    };
    let AssistantContent::ToolCall(call) = &mut content[0] else {
        panic!("call content");
    };
    call.provider = None;
    vec![full, call_only, result_only]
}

#[test]
fn wire_slot_assignment_is_atomic_and_uses_original_content_order() {
    let history = vec![Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::text("before"),
            AssistantContent::ToolCall(ToolCall::from_wire(
                "first",
                ToolFunction {
                    name: "test".into(),
                    arguments: serde_json::json!({}),
                },
            )),
            AssistantContent::text("between"),
            AssistantContent::ToolCall(ToolCall::from_wire(
                "second",
                ToolFunction {
                    name: "test".into(),
                    arguments: serde_json::json!({}),
                },
            )),
        ],
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
fn reserves_future_genuine_ids_and_distinguishes_repeated_turns() {
    let generated = ToolCallId::minted(0);
    let hint = generated.wire_hint().into_owned();
    let history = vec![
        call(generated.clone(), None),
        result(generated.clone(), None),
        call(generated.clone(), None),
        result(generated, None),
        call(explicit(&hint), Some(&hint)),
        result(explicit(&hint), Some(&hint)),
    ];
    let before = serde_json::to_value(&history).unwrap();
    let ids = ToolCallIds::new(&history).unwrap();
    assert_eq!(ids.get(0, 0), ids.get(1, 0));
    assert_eq!(ids.get(2, 0), ids.get(3, 0));
    assert_ne!(ids.get(0, 0), ids.get(2, 0));
    assert_ne!(ids.get(0, 0), ids.get(4, 0));
    assert_ne!(ids.get(2, 0), ids.get(4, 0));
    assert_eq!(ids.get(4, 0), Some(hint.as_str()));
    assert_eq!(ids.get(4, 0), ids.get(5, 0));
    assert_eq!(before, serde_json::to_value(&history).unwrap());
    assert_eq!(ids.ids, ToolCallIds::new(&history).unwrap().ids);
}

#[test]
fn merges_one_leg_metadata_and_handles_reversed_results_across_text() {
    for metadata_on_result in [false, true] {
        let history = vec![
            call(
                ToolCallId::minted(0),
                (!metadata_on_result).then_some("real"),
            ),
            call(ToolCallId::minted(1), None),
            Message::assistant("working"),
            result(ToolCallId::minted(1), None),
            result(ToolCallId::minted(0), metadata_on_result.then_some("real")),
        ];
        let ids = ToolCallIds::new(&history).unwrap();
        assert_eq!(ids.get(0, 0), Some("real"));
        assert_eq!(ids.get(4, 0), Some("real"));
        assert_eq!(ids.get(1, 0), ids.get(3, 0));
        assert_eq!(ids.get(2, 0), None);
    }
}

#[test]
fn rejects_contradictory_references_and_overlapping_local_ids() {
    let history = vec![
        call(explicit("a"), Some("real")),
        result(explicit("a"), Some("other")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::Conflict { .. })
    ));
    let history = vec![
        call(explicit("a"), None),
        call(explicit("b"), Some("real")),
        result(explicit("a"), Some("real")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::Conflict { .. })
    ));
    let history = vec![
        call(explicit("a"), Some("one")),
        call(explicit("a"), Some("two")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::Ambiguous { .. })
    ));
}

#[test]
fn partial_history_requires_provider_identity_and_never_matches_future_calls() {
    let history = vec![
        result(explicit("orphan"), Some("real")),
        call(explicit("orphan"), None),
    ];
    let ids = ToolCallIds::new(&history).unwrap();
    assert_eq!(ids.get(0, 0), Some("real"));
    assert_ne!(ids.get(0, 0), ids.get(1, 0));
    assert!(matches!(
        ToolCallIds::new(&[result(explicit("orphan"), None)]),
        Err(ToolCallIdError::OrphanResult { .. })
    ));
    let history = vec![
        result(explicit("a"), Some("real")),
        result(explicit("b"), Some("real")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::DuplicateResult { .. })
    ));
}

#[test]
fn allows_sequential_real_id_reuse_but_rejects_duplicate_answers() {
    let mut history = vec![
        call(explicit("a"), Some("real")),
        result(explicit("a"), Some("real")),
        call(explicit("a"), Some("real")),
        result(explicit("a"), Some("real")),
    ];
    let ids = ToolCallIds::new(&history).unwrap();
    assert!((0..4).all(|message| ids.get(message, 0) == Some("real")));
    history.push(result(explicit("a"), Some("real")));
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::DuplicateResult { .. })
    ));
}

#[test]
fn matches_exact_provider_call_id_when_local_handles_differ() {
    let history = vec![
        call(explicit("a"), Some("real")),
        result(explicit("b"), Some("real")),
    ];
    let ids = ToolCallIds::new(&history).unwrap();
    assert_eq!(ids.get(0, 0), Some("real"));
    assert_eq!(ids.get(1, 0), Some("real"));
}

#[test]
fn rejects_provider_identity_learned_after_overlapping_calls() {
    for reverse in [false, true] {
        let mut results = [
            result(explicit("a"), Some("real")),
            result(explicit("b"), Some("real")),
        ];
        if reverse {
            results.reverse();
        }
        let mut history = vec![call(explicit("a"), None), call(explicit("b"), None)];
        history.extend(results);
        assert!(matches!(
            ToolCallIds::new(&history),
            Err(ToolCallIdError::Ambiguous { .. })
        ));
    }
}

#[test]
fn stale_local_result_cannot_consume_later_provider_id_reuse() {
    let history = vec![
        call(explicit("a"), Some("real")),
        result(explicit("a"), Some("real")),
        call(explicit("b"), Some("real")),
        result(explicit("a"), Some("real")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::DuplicateResult { .. })
    ));
}

#[test]
fn stale_result_alias_cannot_consume_later_provider_id_reuse() {
    let history = vec![
        call(explicit("a"), Some("real")),
        result(explicit("alias"), Some("real")),
        call(explicit("b"), Some("real")),
        result(explicit("alias"), Some("real")),
    ];
    assert!(matches!(
        ToolCallIds::new(&history),
        Err(ToolCallIdError::DuplicateResult { .. })
    ));
}
