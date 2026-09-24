use rig::completion::Message as CompletionMessage;
use rig::error::ProviderError;
use rig::message::{AssistantContent, Reasoning, ReasoningContent};
use rig::providers::openai::responses_api::{
    CompletionRequest as OpenAIResponsesRequest, Include, InputItem, Output, ReasoningSummary,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn test_input_item_serialization_avoids_duplicate_role() {
    let items: Vec<InputItem> = CompletionMessage::user("hello")
        .try_into()
        .expect("user text converts to one input item");
    let json = serde_json::to_string(&items).expect("serialize InputItem");
    let role_count = json.matches("\"role\"").count();

    assert_eq!(
        role_count, 1,
        "InputItem should serialize a single role field, got {role_count}: {json}"
    );
}

#[test]
fn assistant_reasoning_without_id_is_omitted() {
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::Reasoning(Reasoning::new("thought"))],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("idless reasoning should be omitted without an error");
    assert!(items.is_empty());
}

#[test]
fn assistant_reasoning_encrypted_only_serializes_encrypted_content() {
    let reasoning = Reasoning::encrypted("encrypted_blob").with_id("rs_1".to_string());
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    let item_type = item_json
        .get("type")
        .and_then(|value| value.as_str())
        .expect("reasoning item should include type");
    assert_eq!(item_type, "reasoning");
    assert_eq!(
        item_json.get("id").and_then(|value| value.as_str()),
        Some("rs_1")
    );
    assert_eq!(
        item_json
            .get("encrypted_content")
            .and_then(|value| value.as_str()),
        Some("encrypted_blob")
    );
    assert_eq!(
        item_json
            .get("summary")
            .and_then(|value| value.as_array())
            .map(Vec::len),
        Some(0)
    );
}

#[test]
fn assistant_reasoning_mixed_content_serializes_text_content_and_summaries() {
    let mut reasoning =
        Reasoning::new_with_signature("step-1", Some("sig-1".to_string())).with_id("rs_2".into());
    reasoning
        .content
        .push(ReasoningContent::Summary("summary-2".to_string()));
    reasoning
        .content
        .push(ReasoningContent::Encrypted("ciphertext".to_string()));
    reasoning.content.push(ReasoningContent::Redacted {
        data: "redacted".to_string(),
    });

    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    let content = item_json
        .get("content")
        .and_then(|value| value.as_array())
        .expect("reasoning item should include reasoning content array");
    let content_texts: Vec<&str> = content
        .iter()
        .filter(|entry| entry.get("type").and_then(|kind| kind.as_str()) == Some("reasoning_text"))
        .filter_map(|entry| entry.get("text").and_then(|text| text.as_str()))
        .collect();
    let summary = item_json
        .get("summary")
        .and_then(|value| value.as_array())
        .expect("reasoning item should include summary array");
    let summary_texts: Vec<&str> = summary
        .iter()
        .filter_map(|entry| entry.get("text").and_then(|text| text.as_str()))
        .collect();

    assert_eq!(content_texts, vec!["step-1"]);
    assert_eq!(summary_texts, vec!["summary-2"]);
    assert_eq!(
        item_json
            .get("encrypted_content")
            .and_then(|value| value.as_str()),
        Some("ciphertext")
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

// The mapping from a Responses `output[]` item to rig's reasoning blocks now
// lives in the wire's one decoder, which no public entry point exposes
// outside a recorded exchange; `From<Output> for Vec<AssistantContent>` was
// the client layer's copy of it. What these cells can still pin without a
// fixture is the half that is theirs: the typed `Output::Reasoning` item
// preserves everything the provider sent, which is the precondition for the
// decoder to map it. The mapping half is covered by
// `raw_capture_matrix::responses_reasoning_raw_round_trips_typed`, which
// asserts the normalized reasoning block carries the recorded item's
// encrypted content.

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
fn openai_empty_reasoning_content_roundtrips_to_request_item() {
    // The request side is unchanged, so the cell keeps its subject: a
    // reasoning block with no content still converts to an input item rather
    // than being dropped or erroring.
    let reasoning = Reasoning {
        provider: None,
        id: Some("rs_roundtrip_empty".to_string()),
        content: Vec::new(),
    };

    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };
    let items: Vec<InputItem> = message
        .try_into()
        .expect("empty reasoning content should still convert");

    assert_eq!(items.len(), 1);
    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    assert_eq!(
        item_json.get("id").and_then(|value| value.as_str()),
        Some("rs_roundtrip_empty")
    );
    assert_eq!(
        item_json
            .get("summary")
            .and_then(|value| value.as_array())
            .map(Vec::len),
        Some(0)
    );
    assert!(
        item_json
            .get("encrypted_content")
            .is_none_or(serde_json::Value::is_null)
    );
}

#[test]
fn assistant_reasoning_redacted_only_serializes_as_encrypted_content() {
    let reasoning = Reasoning::redacted("opaque-redacted").with_id("rs_redacted".to_string());
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    assert_eq!(
        item_json
            .get("encrypted_content")
            .and_then(|value| value.as_str()),
        Some("opaque-redacted")
    );
    assert_eq!(
        item_json
            .get("summary")
            .and_then(|value| value.as_array())
            .map(Vec::len),
        Some(0)
    );
}

#[test]
fn openai_responses_request_reasoning_without_id_is_omitted_without_panicking() {
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let request = rig::completion::CompletionRequest {
            chat_history: vec![CompletionMessage::Assistant {
                id: Some("assistant_message_id".to_string()),
                content: vec![AssistantContent::Reasoning(Reasoning::new("thought"))],
            }],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
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
                .contains("OpenAI Responses request input must contain at least one item")
    ));
}

#[test]
fn assistant_tool_call_with_local_id_omits_function_call_item_id() {
    let message = CompletionMessage::Assistant {
        id: None,
        content: vec![AssistantContent::tool_call_with_call_id(
            "history_tool_1",
            "call_local_1".to_string(),
            "my_tool",
            serde_json::json!({}),
        )],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("tool call with call_id should convert");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    assert_eq!(
        item_json.get("type").and_then(|value| value.as_str()),
        Some("function_call")
    );
    assert!(
        item_json.get("id").is_none(),
        "non-`fc_` tool-call IDs must be omitted so the API pairs by call_id: {item_json}"
    );
    assert_eq!(
        item_json.get("call_id").and_then(|value| value.as_str()),
        Some("call_local_1")
    );
}

#[test]
fn assistant_tool_call_with_local_fc_prefix_without_separator_omits_function_call_item_id() {
    let message = CompletionMessage::Assistant {
        id: None,
        content: vec![AssistantContent::tool_call_with_call_id(
            "fclocal_1",
            "call_local_1".to_string(),
            "my_tool",
            serde_json::json!({}),
        )],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("tool call with call_id should convert");
    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    assert!(
        item_json.get("id").is_none(),
        "only provider-native `fc_` item IDs should round-trip: {item_json}"
    );
}

#[test]
fn assistant_tool_call_with_provider_item_id_keeps_it() {
    let message = CompletionMessage::Assistant {
        id: None,
        content: vec![AssistantContent::tool_call_with_call_id(
            "fc_native_1",
            "call_native_1".to_string(),
            "my_tool",
            serde_json::json!({}),
        )],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("tool call with call_id should convert");
    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    assert_eq!(
        item_json.get("id").and_then(|value| value.as_str()),
        Some("fc_native_1"),
        "provider-native fc item IDs must round-trip"
    );
}

#[test]
fn assistant_tool_call_without_provider_id_serializes_the_minted_call_id() {
    // An empty wire id records no provider id and mints rig's correlation
    // handle; the Responses wire requires a `call_id`, so the minted id is
    // sent instead of the old "`call_id` is required" request error.
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: vec![AssistantContent::tool_call(
            "",
            "my_tool",
            serde_json::json!({"arg":"value"}),
        )],
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("id-less tool call should serialize with the minted call_id");
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
    let message = CompletionMessage::tool_result("", "my_tool", "result payload");

    let items: Vec<InputItem> = message
        .try_into()
        .expect("id-less tool result should serialize with the minted call_id");
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
