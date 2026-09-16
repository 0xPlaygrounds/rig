//! Synthetic wire edge cases complement cassette replay: missing and colliding IDs
//! cannot be reliably requested from a live provider.
use crate::operation::Completion;
use crate::wire::{Fold, Operation, Reply, Wire, WireFrame};
use crate::{completion::CompletionResponse, message::AssistantContent};
use serde_json::{Value, json};

/// Fold one whole reply document through a wire's own decoder — the single
/// path a unary reply takes in production, minus the transport.
fn fold_document<W: Wire<Op = Completion>>(wire: &W, body: &Value) -> CompletionResponse {
    let body = body.to_string();
    let mut driver =
        crate::driver::WireDriver::<Completion, _>::new(wire.decoder(crate::wire::Mode::Unary));
    driver.push(WireFrame::Text(body.clone()));
    driver.finish();
    let mut fold = <Completion as Operation>::fold(&crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("probe")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(32),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    });
    for item in driver.drain() {
        fold.absorb(item.expect("the reply decodes without an in-band error"))
            .expect("the fold accepts every event");
    }
    Fold::<Completion>::finish(
        fold,
        Reply {
            provider: wire.name().to_owned(),
            raw: serde_json::from_str(&body).unwrap_or(Value::Null),
            provider_request_id: None,
        },
    )
    .expect("the fold produces a response")
}

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
    let wire = crate::providers::openai::wire::OpenAI::with_key(
        &crate::providers::openai::wire::OPENAI,
        "test-key",
    )
    .chat("test");
    assert_normalization(|| fold_document(&wire, &chat_wire()));
}

#[test]
fn openrouter_missing_ids_and_later_explicit_collision() {
    let wire = crate::providers::openai::wire::OpenAI::with_key(
        &crate::providers::openai::wire::OPENROUTER,
        "test-key",
    )
    .chat("test");
    assert_normalization(|| fold_document(&wire, &chat_wire()));
}

#[test]
fn anthropic_missing_ids_and_later_explicit_collision() {
    let wire = json!({"type":"message","id":"response","model":"test","role":"assistant","stop_reason":"tool_use",
        "usage":{"input_tokens":1,"output_tokens":1},
        "content":(0..3).map(|i|json!({"type":"tool_use","id":if i==1 {"tool-0"} else {""},"name":"same","input":{"n":i}})).collect::<Vec<_>>()});
    let messages = crate::providers::anthropic::wire::Anthropic::new("test-key").messages("test");
    assert_normalization(|| fold_document(&messages, &wire));
}

#[test]
fn cohere_missing_ids_and_later_explicit_collision() {
    let mut message = chat_wire()["choices"][0]["message"].clone();
    message["content"] = json!([{"type":"text","text":"prefix"}]);
    let wire = json!({"id":"response","finish_reason":"TOOL_CALL","message":message});
    let chat = crate::providers::cohere::Cohere::new("test-key").chat("test");
    assert_normalization(|| fold_document(&chat, &wire));
}

#[test]
fn gemini_rest_missing_ids_and_later_explicit_collision() {
    let wire = json!({"candidates":[{"content":{"role":"model","parts":
        (0..3).map(|i|json!({"functionCall":{"id":if i==1 {"tool-0"} else {""},"name":"same","args":{"n":i}}})).collect::<Vec<_>>()
    },"finishReason":"STOP"}]});
    let generate = crate::providers::gemini::Gemini::new("test-key").generate_content("test");
    assert_normalization(|| fold_document(&generate, &wire));
}

#[test]
fn gemini_interactions_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","status":"completed","steps":
        (0..3).map(|i|json!({"type":"function_call","id":if i==1 {"tool-0"} else {""},"name":"same","arguments":{"n":i}})).collect::<Vec<_>>()
    });
    let interactions = crate::providers::gemini::Gemini::new("test-key").interactions("test");
    assert_normalization(|| fold_document(&interactions, &wire));
}

#[test]
fn openai_responses_missing_ids_and_later_explicit_collision() {
    let wire = json!({"id":"response","object":"response","created_at":0,"status":"completed","model":"test","tools":[],"output":
        (0..3).map(|i|json!({"type":"function_call","id":format!("item-{i}"),"call_id":if i==1 {"tool-0"} else {""},"name":"same","arguments":json!({"n":i}).to_string(),"status":"completed"})).collect::<Vec<_>>()
    });
    assert_normalization(|| {
        crate::providers::openai::responses_api::wire::fold_body(
            "test",
            serde_json::from_value::<crate::providers::openai::responses_api::CompletionResponse>(
                wire.clone(),
            )
            .unwrap(),
        )
        .unwrap()
    });
}
