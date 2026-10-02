use rig::completion::Message as CompletionMessage;
use rig::error::ProviderError;
use rig::message::AssistantContent;
use rig::providers::openai::responses_api::{
    CompletionRequest as OpenAIResponsesRequest, Include, InputItem, Output, ReasoningSummary,
    ResponsesRequestParams,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// `message` as the input items of a request to OpenAI.
fn input_of(message: CompletionMessage) -> Result<Vec<InputItem>, rig::error::EncodeError> {
    OpenAIResponsesRequest::try_from(ResponsesRequestParams {
        model: "gpt-5".to_owned(),
        request: rig::completion::CompletionRequest::from(vec![message]),
        system_instructions_placement: Default::default(),
    })
    .map(|request| request.input)
}

#[test]
fn test_input_item_serialization_avoids_duplicate_role() {
    let items: Vec<InputItem> =
        input_of(CompletionMessage::user("hello")).expect("user text converts to one input item");
    let json = serde_json::to_string(&items).expect("serialize InputItem");
    let role_count = json.matches("\"role\"").count();

    assert_eq!(
        role_count, 1,
        "InputItem should serialize a single role field, got {role_count}: {json}"
    );
}

#[test]
fn openai_responses_request_auto_adds_reasoning_encrypted_include() {
    let core_request = rig::completion::CompletionRequest {
        chat_history: vec![CompletionMessage::user("hello")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "reasoning": { "effort": "low" }
        })),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let request = OpenAIResponsesRequest::try_from(("gpt-test".to_string(), core_request))
        .expect("convert request");
    let include = request
        .additional_parameters
        .include
        .expect("include should be auto-populated when reasoning is configured");
    assert!(
        include
            .iter()
            .any(|item| matches!(item, Include::ReasoningEncryptedContent))
    );
}

#[test]
fn openai_responses_reasoning_output_preserves_encrypted_content() {
    let output: Output = serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": "rs_out_1",
        "summary": [
            { "type": "summary_text", "text": "summary text" }
        ],
        "encrypted_content": "cipher_blob",
        "status": "completed"
    }))
    .expect("deserialize reasoning output");

    let Output::Reasoning {
        id,
        summary,
        encrypted_content,
        ..
    } = &output
    else {
        panic!("expected a reasoning output item");
    };
    assert_eq!(id, "rs_out_1");
    assert_eq!(encrypted_content.as_deref(), Some("cipher_blob"));
    assert_eq!(
        summary
            .iter()
            .map(ReasoningSummary::text)
            .collect::<Vec<_>>(),
        ["summary text"]
    );
}

#[test]
fn openai_responses_reasoning_output_preserves_reasoning_text_content() {
    let output: Output = serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": "rs_text_1",
        "summary": [],
        "content": [
            { "type": "reasoning_text", "text": "visible reasoning" }
        ],
        "status": "completed"
    }))
    .expect("deserialize reasoning output");

    let Output::Reasoning {
        id,
        summary,
        content,
        ..
    } = &output
    else {
        panic!("expected a reasoning output item");
    };
    assert_eq!(id, "rs_text_1");
    assert!(summary.is_empty());
    assert_eq!(content, &["visible reasoning".to_string()]);
}

#[test]
fn openai_responses_reasoning_output_without_summary_is_not_dropped() {
    let output: Output = serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": "rs_empty",
        "summary": []
    }))
    .expect("deserialize reasoning output");

    let Output::Reasoning {
        id,
        summary,
        content,
        encrypted_content,
        ..
    } = &output
    else {
        panic!("a contentless reasoning item must still decode as one");
    };
    assert_eq!(id, "rs_empty");
    assert!(summary.is_empty());
    assert!(content.is_empty());
    assert_eq!(encrypted_content.as_deref(), None);
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

    let items: Vec<InputItem> =
        input_of(message).expect("id-less tool call should serialize with the minted call_id");
    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
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

    let items: Vec<InputItem> =
        input_of(message).expect("id-less tool result should serialize with the minted call_id");
    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
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
        let request = rig::completion::CompletionRequest {
            chat_history: vec![CompletionMessage::user("hello")],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(serde_json::json!("not_a_valid_object")),
            model: None,
            output_schema: None,
            record_telemetry_content: false,
        };
        OpenAIResponsesRequest::try_from(("gpt-test".to_string(), request))
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
    let request = rig::completion::CompletionRequest {
        chat_history: vec![CompletionMessage::user("hello")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "prompt_cache_key": "tenant-agent-scaffold",
            "prompt_cache_retention": "24h"
        })),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let request = OpenAIResponsesRequest::try_from(("gpt-test".to_string(), request))
        .expect("convert request");
    let request_json = serde_json::to_value(request).expect("serialize request");

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
