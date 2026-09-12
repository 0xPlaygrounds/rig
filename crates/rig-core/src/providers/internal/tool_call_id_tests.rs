//! Synthetic wire edge cases complement cassette replay: missing and colliding IDs
//! cannot be reliably requested from a live provider.
use crate::{
    completion::{CompletionResponse, NormalizeCompletionResponse},
    message::AssistantContent,
};
use serde_json::{Value, json};

fn assert_normalization(convert: impl Fn() -> CompletionResponse) {
    let first = convert();
    assert_eq!(
        first.choice,
        convert().choice,
        "repeat conversion is deterministic"
    );
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls
            .iter()
            .map(|call| &call.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert_eq!(calls[1].id.explicit(), Some("tool-0"));
    assert_eq!(calls[1].provider.as_ref().unwrap().call_id, "tool-0");
    for index in [0, 2] {
        assert!(calls[index].provider.is_none());
    }
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(call.function.arguments, json!({"n":index}));
    }
}

fn chat_wire() -> Value {
    json!({"id":"response", "object":"chat.completion", "created":0, "model":"test",
    "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"prefix", "tool_calls":
        (0..3).map(|i|json!({"id":if i==1 {"tool-0"} else {""},"type":"function","function":{"name":"same","arguments":json!({"n":i}).to_string()}})).collect::<Vec<_>>()
    }}]})
}

#[test]
fn openai_chat_missing_ids_and_later_explicit_collision() {
    assert_normalization(|| {
        serde_json::from_value::<crate::providers::openai::completion::CompletionResponse>(
            chat_wire(),
        )
        .unwrap()
        .normalize("test")
        .unwrap()
    });
}

#[test]
fn openrouter_missing_ids_and_later_explicit_collision() {
    assert_normalization(|| {
        serde_json::from_value::<crate::providers::openrouter::completion::CompletionResponse>(
            chat_wire(),
        )
        .unwrap()
        .normalize("test")
        .unwrap()
    });
}

#[test]
fn anthropic_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","model":"test","role":"assistant","stop_reason":"tool_use",
        "usage":{"input_tokens":1,"output_tokens":1},
        "content":(0..3).map(|i|json!({"type":"tool_use","id":if i==1 {"tool-0"} else {""},"name":"same","input":{"n":i}})).collect::<Vec<_>>()});
    assert_normalization(|| {
        serde_json::from_value::<crate::providers::anthropic::completion::CompletionResponse>(
            wire.clone(),
        )
        .unwrap()
        .normalize("test")
        .unwrap()
    });
}

#[test]
fn cohere_missing_ids_and_later_explicit_collision() {
    let mut message = chat_wire()["choices"][0]["message"].clone();
    message["content"] = json!([{"type":"text","text":"prefix"}]);
    let wire = json!({"id":"response","finish_reason":"TOOL_CALL","message":message});
    assert_normalization(|| {
        serde_json::from_value::<crate::providers::cohere::completion::CompletionResponse>(
            wire.clone(),
        )
        .unwrap()
        .try_into()
        .unwrap()
    });
}

#[test]
fn gemini_rest_missing_ids_and_later_explicit_collision() {
    let wire = json!({"candidates":[{"content":{"role":"model","parts":
        (0..3).map(|i|json!({"functionCall":{"id":if i==1 {"tool-0"} else {""},"name":"same","args":{"n":i}}})).collect::<Vec<_>>()
    },"finishReason":"STOP"}]});
    assert_normalization(|| {
        serde_json::from_value::<
            crate::providers::gemini::completion::gemini_api_types::GenerateContentResponse,
        >(wire.clone())
        .unwrap()
        .try_into()
        .unwrap()
    });
}

#[test]
fn gemini_interactions_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","status":"completed","steps":
        (0..3).map(|i|json!({"type":"function_call","id":if i==1 {"tool-0"} else {""},"name":"same","arguments":{"n":i}})).collect::<Vec<_>>()
    });
    assert_normalization(|| {
        serde_json::from_value::<
            crate::providers::gemini::interactions_api::interactions_api_types::Interaction,
        >(wire.clone())
        .unwrap()
        .try_into()
        .unwrap()
    });
}

#[test]
fn openai_responses_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","object":"response","created_at":0,"status":"completed","model":"test","tools":[],"output":
        (0..3).map(|i|json!({"type":"function_call","id":format!("item-{i}"),"call_id":if i==1 {"tool-0"} else {""},"name":"same","arguments":json!({"n":i}).to_string(),"status":"completed"})).collect::<Vec<_>>()
    });
    assert_normalization(|| {
        serde_json::from_value::<crate::providers::openai::responses_api::CompletionResponse>(
            wire.clone(),
        )
        .unwrap()
        .normalize("test")
        .unwrap()
    });
}
