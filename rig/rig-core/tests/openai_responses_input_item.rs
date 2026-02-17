use rig::OneOrMany;
use rig::completion::{CompletionError, Message as CompletionMessage};
use rig::message::{AssistantContent, Reasoning, ReasoningContent};
use rig::providers::openai::responses_api::{
    CompletionRequest as OpenAIResponsesRequest, Include, InputItem, Message, Output, UserContent,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn test_input_item_serialization_avoids_duplicate_role() {
    let message = Message::User {
        content: OneOrMany::one(UserContent::InputText {
            text: "hello".to_string(),
        }),
        name: None,
    };
    let item: InputItem = message.into();
    let json = serde_json::to_string(&item).expect("serialize InputItem");
    let role_count = json.matches("\"role\"").count();

    assert_eq!(
        role_count, 1,
        "InputItem should serialize a single role field, got {role_count}: {json}"
    );
}

#[test]
fn assistant_reasoning_without_id_returns_error() {
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(Reasoning::new("thought"))),
    };

    let items: Result<Vec<InputItem>, CompletionError> = message.try_into();
    assert!(matches!(
        items,
        Err(CompletionError::ProviderError(message))
            if message.contains("OpenAI-generated ID is required")
    ));
}

#[test]
fn assistant_reasoning_encrypted_only_serializes_encrypted_content() {
    let reasoning = Reasoning::encrypted("encrypted_blob").with_id("rs_1".to_string());
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
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
fn assistant_reasoning_mixed_content_serializes_only_text_like_summaries() {
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
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
    };

    let items: Vec<InputItem> = message
        .try_into()
        .expect("assistant reasoning should convert to InputItem");
    assert_eq!(items.len(), 1);

    let item_json = serde_json::to_value(&items[0]).expect("serialize InputItem");
    let summary = item_json
        .get("summary")
        .and_then(|value| value.as_array())
        .expect("reasoning item should include summary array");
    let summary_texts: Vec<&str> = summary
        .iter()
        .filter_map(|entry| entry.get("text").and_then(|text| text.as_str()))
        .collect();

    assert_eq!(summary_texts, vec!["step-1", "summary-2"]);
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
        preamble: None,
        chat_history: OneOrMany::one(CompletionMessage::user("hello")),
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

    let content: Vec<AssistantContent> = output.into();
    assert_eq!(content.len(), 1);
    let Some(AssistantContent::Reasoning(reasoning)) = content.first() else {
        panic!("expected reasoning output content");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_out_1"));
    assert!(matches!(
        reasoning.content.first(),
        Some(ReasoningContent::Summary(summary)) if summary == "summary text"
    ));
    assert!(matches!(
        reasoning.content.get(1),
        Some(ReasoningContent::Encrypted(value)) if value == "cipher_blob"
    ));
}

#[test]
fn openai_responses_reasoning_output_without_summary_is_not_dropped() {
    let output: Output = serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": "rs_empty",
        "summary": []
    }))
    .expect("deserialize reasoning output");

    let content: Vec<AssistantContent> = output.into();
    assert_eq!(content.len(), 1);
    let Some(AssistantContent::Reasoning(reasoning)) = content.first() else {
        panic!("expected reasoning output content");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_empty"));
    assert!(reasoning.content.is_empty());
}

#[test]
fn openai_empty_reasoning_content_roundtrips_to_request_item() {
    let output: Output = serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": "rs_roundtrip_empty",
        "summary": []
    }))
    .expect("deserialize reasoning output");
    let content: Vec<AssistantContent> = output.into();
    let Some(AssistantContent::Reasoning(reasoning)) = content.first().cloned() else {
        panic!("expected reasoning output content");
    };

    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
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
        content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
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
fn openai_responses_request_reasoning_without_id_returns_error_without_panicking() {
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let request = rig::completion::CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(CompletionMessage::Assistant {
                id: Some("assistant_message_id".to_string()),
                content: OneOrMany::one(AssistantContent::Reasoning(Reasoning::new("thought"))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            model: None,
            output_schema: None,
        };
        OpenAIResponsesRequest::try_from(("gpt-test".to_string(), request))
    }));

    let conversion = panic_result.expect("request conversion should not panic");
    assert!(matches!(
        conversion,
        Err(CompletionError::ProviderError(message))
            if message.contains("OpenAI-generated ID is required")
    ));
}

#[test]
fn assistant_tool_call_without_call_id_returns_request_error() {
    let message = CompletionMessage::Assistant {
        id: Some("assistant_message_id".to_string()),
        content: OneOrMany::one(AssistantContent::tool_call(
            "tool_1",
            "my_tool",
            serde_json::json!({"arg":"value"}),
        )),
    };

    let items: Result<Vec<InputItem>, CompletionError> = message.try_into();
    assert!(matches!(
        items,
        Err(CompletionError::RequestError(error))
            if error
                .to_string()
                .contains("Assistant tool call `call_id` is required")
    ));
}

#[test]
fn user_tool_result_without_call_id_returns_request_error() {
    let message = CompletionMessage::tool_result("tool_1", "result payload");

    let items: Result<Vec<InputItem>, CompletionError> = message.try_into();
    assert!(matches!(
        items,
        Err(CompletionError::RequestError(error))
            if error
                .to_string()
                .contains("Tool result `call_id` is required")
    ));
}

#[test]
fn openai_responses_invalid_additional_params_returns_error_without_panicking() {
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let request = rig::completion::CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(CompletionMessage::user("hello")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(serde_json::json!(true)),
            model: None,
            output_schema: None,
        };
        OpenAIResponsesRequest::try_from(("gpt-test".to_string(), request))
    }));

    let conversion = panic_result.expect("request conversion should not panic");
    assert!(matches!(
        conversion,
        Err(CompletionError::RequestError(error))
            if error
                .to_string()
                .contains("Invalid OpenAI Responses additional_params payload")
    ));
}
