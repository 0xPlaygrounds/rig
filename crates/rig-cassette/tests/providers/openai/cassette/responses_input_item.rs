use rig::completion::Message as CompletionMessage;
use rig::error::ProviderError;
use rig::message::AssistantContent;
use rig::wire::{Mode, Wire};
use serde_json::Value;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The body the OpenAI Responses wire for `model` sends for `request`.
fn body_of(
    model: &str,
    request: rig::completion::CompletionRequest,
) -> Result<Value, rig::error::EncodeError> {
    let wire = rig::providers::openai::responses_api::wire::Responses::new(
        rig::providers::openai::OpenAIConfig::new("unused"),
        model,
    );
    let encoded = wire.encode(request, Mode::Unary)?;
    match encoded.request.body() {
        rig::wire::Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        _ => Err(rig::error::EncodeError::request("a Responses body is JSON")),
    }
}

/// `message` as the input items of a request to OpenAI.
fn input_of(message: CompletionMessage) -> Result<Vec<Value>, rig::error::EncodeError> {
    let body = body_of(
        "gpt-5",
        rig::completion::CompletionRequest::from(vec![message]),
    )?;
    Ok(body["input"].as_array().cloned().unwrap_or_default())
}

#[test]
fn test_input_item_serialization_avoids_duplicate_role() {
    let items =
        input_of(CompletionMessage::user("hello")).expect("user text converts to one input item");
    let json = serde_json::to_string(&items).expect("serialize the input");
    let role_count = json.matches("\"role\"").count();

    assert_eq!(
        role_count, 1,
        "an input item should serialize a single role field, got {role_count}: {json}"
    );
}

#[test]
fn openai_responses_request_auto_adds_reasoning_encrypted_include() {
    let mut core_request =
        rig::completion::CompletionRequest::from(vec![CompletionMessage::user("hello")]);
    core_request.additional_params = Some(serde_json::json!({
        "reasoning": { "effort": "low" }
    }));

    let request = body_of("gpt-test", core_request).expect("convert request");
    assert_eq!(
        request["include"],
        serde_json::json!(["reasoning.encrypted_content"]),
        "include should be auto-populated when reasoning is configured"
    );
}

/// The reasoning block a whole Responses body holding `item` decodes to.
fn decoded_reasoning(item: serde_json::Value) -> rig::message::Reasoning {
    let wire = rig::providers::openai::responses_api::wire::Responses::new(
        rig::providers::openai::OpenAIConfig::new("unused"),
        "gpt-5",
    );
    let body = serde_json::json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [item],
    });
    let response = rig::test_utils::history::decode(
        &wire,
        rig::wire::Mode::Unary,
        [rig::wire::WireFrame::Text(body.to_string())],
    )
    .expect("the body decodes");
    response
        .choice
        .into_iter()
        .find_map(|block| match block {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .expect("the reasoning item decodes to a reasoning block")
}

#[test]
fn openai_responses_reasoning_output_preserves_encrypted_content() {
    let item = serde_json::json!({
        "type": "reasoning",
        "id": "rs_out_1",
        "summary": [
            { "type": "summary_text", "text": "summary text" }
        ],
        "encrypted_content": "cipher_blob",
        "status": "completed"
    });
    let reasoning = decoded_reasoning(item.clone());
    assert_eq!(reasoning.text, "summary text");
    let native = reasoning.native.expect("the item is the block's native");
    assert_eq!(native.item, item);
    assert_eq!(native.item["encrypted_content"], "cipher_blob");
}

#[test]
fn openai_responses_reasoning_output_preserves_reasoning_text_content() {
    let reasoning = decoded_reasoning(serde_json::json!({
        "type": "reasoning",
        "id": "rs_text_1",
        "summary": [],
        "content": [
            { "type": "reasoning_text", "text": "visible reasoning" }
        ],
        "status": "completed"
    }));
    assert_eq!(reasoning.text, "visible reasoning");
    assert_eq!(
        reasoning.native.expect("the item is kept").item["id"],
        "rs_text_1"
    );
}

#[test]
fn openai_responses_reasoning_output_without_summary_is_not_dropped() {
    let reasoning = decoded_reasoning(serde_json::json!({
        "type": "reasoning",
        "id": "rs_empty",
        "summary": []
    }));
    assert!(reasoning.text.is_empty());
    let native = reasoning
        .native
        .expect("a contentless reasoning item must still decode as one");
    assert_eq!(native.item["id"], "rs_empty");
    assert!(native.item.get("encrypted_content").is_none());
}

#[test]
fn assistant_tool_call_without_provider_id_serializes_the_minted_call_id() {
    // An empty wire id records no provider id and mints rig's correlation
    // handle; the Responses wire requires a `call_id`, so the minted id is
    // sent instead of the old "`call_id` is required" request error.
    let message = CompletionMessage::Assistant(rig_core::message::AssistantMessage::new(vec![
        AssistantContent::tool_call(
            "",
            rig_core::message::ToolName::new("my_tool").expect("tool name"),
            serde_json::json!({"arg":"value"}),
        ),
    ]));

    let items =
        input_of(message).expect("id-less tool call should serialize with the minted call_id");
    let item_json = items[0].clone();
    let call_id = item_json
        .get("call_id")
        .and_then(|value| value.as_str())
        .expect("function_call should carry a call_id");
    assert!(
        !call_id.is_empty(),
        "minted call_id must be non-empty: {item_json}"
    );
    assert!(
        item_json.get("id").is_none(),
        "no provider-native `fc_` item id exists to round-trip: {item_json}"
    );
}

#[test]
fn user_tool_result_without_provider_id_serializes_the_minted_call_id() {
    // An empty provider call id mints rig's correlation handle; the
    // Responses wire requires a `call_id`, so the minted id is sent instead
    // of the old "`call_id` is required" request error.
    let message = CompletionMessage::tool_result(
        rig_core::message::CallId::from_wire(""),
        rig_core::message::ToolName::new("my_tool").expect("tool name"),
        "result payload",
    );

    let items =
        input_of(message).expect("id-less tool result should serialize with the minted call_id");
    let item_json = items[0].clone();
    assert_eq!(
        item_json.get("type").and_then(|value| value.as_str()),
        Some("function_call_output"),
        "tool result should serialize as a function_call_output: {item_json}"
    );
    let call_id = item_json
        .get("call_id")
        .and_then(|value| value.as_str())
        .expect("function_call_output should carry a call_id");
    assert!(
        !call_id.is_empty(),
        "minted call_id must be non-empty: {item_json}"
    );
}

#[test]
fn openai_responses_invalid_additional_params_returns_error_without_panicking() {
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let mut request =
            rig::completion::CompletionRequest::from(vec![CompletionMessage::user("hello")]);
        request.additional_params = Some(serde_json::json!("not_a_valid_object"));
        body_of("gpt-test", request)
    }));

    let conversion = panic_result.expect("request conversion should not panic");
    assert!(matches!(
        conversion.map_err(ProviderError::from),
        Err(ProviderError::Request(error))
            if error
                .to_string()
                .contains("Invalid OpenAI Responses additional_params payload")
    ));
}

#[test]
fn openai_responses_request_preserves_prompt_cache_parameters() {
    let mut request =
        rig::completion::CompletionRequest::from(vec![CompletionMessage::user("hello")]);
    request.additional_params = Some(serde_json::json!({
        "prompt_cache_key": "tenant-agent-scaffold",
        "prompt_cache_retention": "24h"
    }));

    let request_json = body_of("gpt-test", request).expect("convert request");

    assert_eq!(
        request_json
            .get("prompt_cache_key")
            .and_then(|value| value.as_str()),
        Some("tenant-agent-scaffold")
    );
    assert_eq!(
        request_json
            .get("prompt_cache_retention")
            .and_then(|value| value.as_str()),
        Some("24h")
    );
}
