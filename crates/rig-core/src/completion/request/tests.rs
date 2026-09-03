use super::{CompletionResponse, FinishReason, ProviderCapabilities, Usage};
use crate::message::AssistantContent;

mod message_content_validation {
    use super::super::CompletionRequest;
    use crate::message::{AssistantContent, Message, UserContent};

    fn request(chat_history: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history,
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn a_populated_history_passes() {
        let request = request(vec![Message::user("hello")]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn an_empty_history_is_rejected() {
        // `chat_history` was non-empty by construction until the container
        // was removed; the field is public, so this is now constructible.
        let error = request(Vec::new())
            .validate_message_content()
            .expect_err("an empty history must not reach a provider");
        assert!(
            error.to_string().contains("empty chat history"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn an_empty_user_message_is_rejected_by_index_and_role() {
        let error = request(vec![
            Message::user("hello"),
            Message::User {
                content: Vec::new(),
            },
        ])
        .validate_message_content()
        .expect_err("an empty user message must not reach a provider");
        let message = error.to_string();
        assert!(message.contains("user message at index 1"), "{message}");
    }

    #[test]
    fn an_empty_assistant_message_is_rejected_by_index_and_role() {
        let error = request(vec![Message::Assistant {
            id: None,
            content: Vec::new(),
        }])
        .validate_message_content()
        .expect_err("an empty assistant message must not reach a provider");
        let message = error.to_string();
        assert!(
            message.contains("assistant message at index 0"),
            "{message}"
        );
    }

    #[test]
    fn an_empty_system_message_is_not_rejected() {
        // System content is a `String` and always has been, so the removed
        // container never constrained it. Rejecting an empty one would be a
        // new restriction, and would break a history carrying a
        // conditionally built preamble that resolved to `""`.
        let request = request(vec![
            Message::System {
                content: String::new(),
            },
            Message::user("hello"),
        ]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn a_block_less_tool_result_is_rejected_naming_the_tool() {
        use crate::message::{ToolCallId, ToolResult, ToolResultContent};
        // `ToolResult::content` was non-empty by construction until the
        // container was removed; the message around it has one item, so the
        // message-level check alone would let this reach the wire as
        // `"content": []`.
        let error = request(vec![
            Message::user("hello"),
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    call: ToolCallId::new_or_minted("call_1", 0),
                    provider: None,
                    name: "lookup".to_owned(),
                    content: Vec::<ToolResultContent>::new(),
                })],
            },
        ])
        .validate_message_content()
        .expect_err("a block-less tool result must not reach a provider");
        let message = error.to_string();
        assert!(message.contains("`lookup`"), "{message}");
        assert!(message.contains("index 0"), "{message}");
        assert!(message.contains("user message at index 1"), "{message}");
    }

    #[test]
    fn a_tool_result_with_one_empty_string_block_is_accepted() {
        use crate::message::{ToolCallId, ToolResult, ToolResultContent};
        // The guard is on the cardinality of the block list, not on the
        // blocks' content. A tool that legitimately returned an empty
        // string produces one block and must still send — this pins the
        // "no blocks" / "no content" distinction so a future tightening
        // pass cannot collapse it.
        let request = request(vec![Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: ToolCallId::new_or_minted("call_1", 0),
                provider: None,
                name: "lookup".to_owned(),
                content: vec![ToolResultContent::text("")],
            })],
        }]);
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn the_legacy_fabricated_sentinel_still_passes() {
        // Histories persisted before message content became a `Vec` encode a
        // content-less assistant turn as a single empty text part. That is a
        // one-element list, so it validates — the rule is block count, not
        // block content — and, being caller-supplied history, it is never
        // filtered: it goes to the wire as-is (where some providers reject
        // it). Callers migrating pre-`Vec` histories drop such turns
        // themselves; see MIGRATING.
        let request = request(vec![
            Message::user("hello"),
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::text("")],
            },
            Message::User {
                content: vec![UserContent::text("and again")],
            },
        ]);
        assert!(request.validate_message_content().is_ok());
    }
}

fn tool_call_choice() -> Vec<AssistantContent> {
    vec![AssistantContent::tool_call(
        "call_1",
        "lookup",
        serde_json::json!({"query": "rig"}),
    )]
}

#[test]
fn normalized_response_round_trips_through_serde() {
    let response = CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage {
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
            cached_input_tokens: 1,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 1,
        },
        "example",
    )
    .with_message_id("msg_123")
    .with_finish_reason(FinishReason::Stop)
    .with_model("provider-model-v2");

    let encoded = serde_json::to_value(&response).expect("serialize response");
    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded.clone()).expect("deserialize");

    assert_eq!(
        serde_json::to_value(decoded).expect("re-serialize"),
        encoded
    );
}

/// Serde must not be a back door around `reconcile_with_output`: a
/// persisted `"stop"` next to a tool-call choice deserializes as
/// `ToolCalls`, exactly as if it had gone through the setter.
#[test]
fn deserializing_stop_with_a_tool_call_reconciles_to_tool_calls() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        tool_call_choice(),
        Usage::new(),
        "example",
    ))
    .expect("serialize response");
    encoded["finish_reason"] = serde_json::json!("stop");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Serde must not be a back door around the empty-string filtering either:
/// a persisted `""` identifier deserializes as `None`.
#[test]
fn deserializing_empty_identifiers_yields_none() {
    let mut encoded = serde_json::to_value(CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage::new(),
        "example",
    ))
    .expect("serialize response");
    encoded["message_id"] = serde_json::json!("");
    encoded["response_id"] = serde_json::json!("");
    encoded["model"] = serde_json::json!("");

    let decoded =
        serde_json::from_value::<CompletionResponse>(encoded).expect("deserialize response");

    assert_eq!(decoded.message_id, None);
    assert_eq!(decoded.response_id, None);
    assert_eq!(decoded.model, None);
}

#[test]
fn unknown_finish_reason_survives_a_serde_round_trip_verbatim() {
    let reason = FinishReason::Other("provider_specific_stop".to_owned());
    let encoded = serde_json::to_string(&reason).expect("serialize");
    let decoded = serde_json::from_str::<FinishReason>(&encoded).expect("deserialize");

    assert_eq!(decoded, reason);
}

#[test]
fn stop_with_a_tool_call_reconciles_to_tool_calls() {
    let response = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
        .with_finish_reason(FinishReason::Stop);

    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

/// The `Option` setter is what provider conversions actually reach for, so
/// it must reconcile identically — a provider holding an `Option` must not
/// have to choose between ergonomics and correctness.
#[test]
fn optional_setter_reconciles_exactly_like_the_plain_setter() {
    let via_option = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
        .with_optional_finish_reason(Some(FinishReason::Stop));
    let via_plain = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
        .with_finish_reason(FinishReason::Stop);

    assert_eq!(via_option.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(via_option.finish_reason, via_plain.finish_reason);
}

#[test]
fn reconciliation_only_upgrades_a_natural_stop() {
    // A truncated tool call is still a truncation; a filtered one is still
    // filtered. Overriding either would lose why the turn actually ended.
    for reason in [
        FinishReason::Length,
        FinishReason::ContentFilter,
        FinishReason::Other("provider_specific".to_owned()),
    ] {
        let response = CompletionResponse::new(tool_call_choice(), Usage::new(), "example")
            .with_finish_reason(reason.clone());

        assert_eq!(response.finish_reason, Some(reason));
    }
}

#[test]
fn reconciliation_leaves_a_stop_without_tool_calls_alone() {
    let response = CompletionResponse::new(
        vec![AssistantContent::text("done")],
        Usage::new(),
        "example",
    )
    .with_finish_reason(FinishReason::Stop);

    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
}

#[test]
fn provider_capabilities_are_externally_configurable_from_default() {
    let capabilities = ProviderCapabilities::default().with_native_output_tool_composition(true);

    assert!(capabilities.composes_native_output_with_tools);
    assert!(!ProviderCapabilities::new().composes_native_output_with_tools);
    assert_eq!(ProviderCapabilities::new(), ProviderCapabilities::default());
}

#[test]
fn usage_has_values_reflects_the_zero_sentinel() {
    use super::Usage;

    assert!(!Usage::new().has_values());

    let mut usage = Usage::new();
    usage.reasoning_tokens = 1;
    assert!(usage.has_values());
}

use super::*;
use crate::test_utils::MockCompletionModel;

#[test]
fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
    let default_request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), "completion prompt").build();
    assert!(!default_request.record_telemetry_content);

    let default_json = serde_json::to_value(&default_request).expect("serialize request");
    assert!(
        default_json.get("record_telemetry_content").is_none(),
        "safe default should not serialize the telemetry opt-in field"
    );
    let default_roundtrip: CompletionRequest =
        serde_json::from_value(default_json).expect("deserialize default request");
    assert!(!default_roundtrip.record_telemetry_content);

    let opt_in_request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), "completion prompt")
            .record_content_telemetry(true)
            .build();
    assert!(opt_in_request.record_telemetry_content);

    let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
    assert!(
        opt_in_json.get("record_telemetry_content").is_none(),
        "local telemetry policy must not be serialized into provider requests"
    );
    let legacy_roundtrip: CompletionRequest =
        serde_json::from_value(opt_in_json).expect("deserialize legacy request");
    assert!(
        !legacy_roundtrip.record_telemetry_content,
        "missing field should deserialize to the safe default"
    );
}

/// The deserialization mirror carries `raw`: a response with a captured
/// payload survives serialize → deserialize with the payload intact, a
/// response serialized before the field existed still loads with `raw`
/// unset, and an unset `raw` is not written.
#[test]
fn normalized_response_raw_round_trips_through_serde_mirror() {
    let payload = serde_json::json!({
        "id": "chatcmpl-1",
        "system_fingerprint": "fp_abc",
        "choices": [{"finish_reason": "stop"}]
    });
    let response = CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage::new(),
        "example",
    )
    .with_response_id("chatcmpl-1")
    .with_raw(payload.clone());

    let encoded = serde_json::to_value(&response).expect("serialize response");
    assert_eq!(encoded["raw"], payload);
    let decoded: CompletionResponse =
        serde_json::from_value(encoded.clone()).expect("deserialize response");
    assert_eq!(decoded.raw, payload);
    assert_eq!(decoded.response_id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );

    let legacy = serde_json::json!({
        "choice": [{"type": "text", "text": "hello"}],
        "usage": serde_json::to_value(Usage::new()).unwrap(),
        "provider": "example"
    });
    let decoded: CompletionResponse = serde_json::from_value(legacy).expect("legacy loads");
    assert!(decoded.raw.is_null());

    let bare = serde_json::to_value(CompletionResponse::new(
        vec![AssistantContent::text("hello")],
        Usage::new(),
        "example",
    ))
    .unwrap();
    assert!(bare.get("raw").is_none());
}

fn test_document(id: &str, text: &str) -> Document {
    Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

#[test]
fn message_telemetry_includes_normalized_documents() {
    let builder = CompletionRequestBuilder::new(MockCompletionModel::default(), "prompt")
        .preamble("system".to_string())
        .message(Message::user("history"))
        .document(test_document("doc1", "static context secret"));

    let messages = builder.messages_for_telemetry();
    assert_eq!(messages.len(), 4);
    assert!(matches!(messages[0], Message::System { .. }));
    assert!(is_document_message(&messages[1], "doc1"));
    assert!(matches!(
        &messages[2],
        Message::User { content }
            if matches!(content.first(), Some(UserContent::Text(text)) if text.text == "history")
    ));
    assert!(matches!(
        &messages[3],
        Message::User { content }
            if matches!(content.first(), Some(UserContent::Text(text)) if text.text == "prompt")
    ));

    let request = builder.build();
    assert_eq!(messages, request.chat_history_with_documents());
}

fn is_document_message(message: &Message, expected_id: &str) -> bool {
    let Message::User { content } = message else {
        return false;
    };

    content.iter().any(|content| {
        matches!(
            content,
            UserContent::Document(document)
                if document.data.to_string().contains(&format!("<file id: {expected_id}>"))
        )
    })
}

#[test]
fn test_document_display_without_metadata() {
    let doc = Document {
        id: "123".to_string(),
        text: "This is a test document.".to_string(),
        additional_props: HashMap::new(),
    };

    let expected = "<file id: 123>\nThis is a test document.\n</file>\n";
    assert_eq!(format!("{doc}"), expected);
}

#[test]
fn test_document_display_with_metadata() {
    let mut additional_props = HashMap::new();
    additional_props.insert("author".to_string(), "John Doe".to_string());
    additional_props.insert("length".to_string(), "42".to_string());

    let doc = Document {
        id: "123".to_string(),
        text: "This is a test document.".to_string(),
        additional_props,
    };

    let expected = concat!(
        "<file id: 123>\n",
        "<metadata author: \"John Doe\" length: \"42\" />\n",
        "This is a test document.\n",
        "</file>\n"
    );
    assert_eq!(format!("{doc}"), expected);
}

#[test]
fn test_normalize_documents_with_documents() {
    let doc1 = Document {
        id: "doc1".to_string(),
        text: "Document 1 text.".to_string(),
        additional_props: HashMap::new(),
    };

    let doc2 = Document {
        id: "doc2".to_string(),
        text: "Document 2 text.".to_string(),
        additional_props: HashMap::new(),
    };

    let request = CompletionRequest {
        model: None,
        chat_history: vec!["What is the capital of France?".into()],
        documents: vec![doc1, doc2],
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let expected = Message::User {
        content: vec![
            UserContent::document(
                "<file id: doc1>\nDocument 1 text.\n</file>\n".to_string(),
                Some(DocumentMediaType::TXT),
            ),
            UserContent::document(
                "<file id: doc2>\nDocument 2 text.\n</file>\n".to_string(),
                Some(DocumentMediaType::TXT),
            ),
        ],
    };

    assert_eq!(request.normalized_documents(), Some(expected));
}

#[test]
fn test_normalize_documents_without_documents() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec!["What is the capital of France?".into()],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    assert_eq!(request.normalized_documents(), None);
}

#[test]
fn preamble_builder_funnels_to_system_message() {
    let request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
            .preamble("System prompt".to_string())
            .message(Message::user("History"))
            .build();

    let history = request.chat_history.into_iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 3);
    assert!(matches!(
        &history[0],
        Message::System { content } if content == "System prompt"
    ));
    assert!(matches!(&history[1], Message::User { .. }));
    assert!(matches!(&history[2], Message::User { .. }));
}

#[test]
fn build_places_documents_after_preamble_system_message() {
    let request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
            .preamble("System prompt".to_string())
            .document(test_document("doc1", "Document text."))
            .build();

    assert_eq!(request.documents.len(), 1);

    let history = request.chat_history_with_documents();
    let history = history.iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 3);
    assert!(matches!(
        history[0],
        Message::System { content } if content == "System prompt"
    ));
    assert!(is_document_message(history[1], "doc1"));
    assert!(matches!(history[2], Message::User { .. }));
}

#[test]
fn build_places_documents_after_leading_system_messages_before_prior_history() {
    let request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
            .message(Message::system("System one"))
            .message(Message::system("System two"))
            .message(Message::user("Earlier user turn"))
            .message(Message::assistant("Earlier assistant turn"))
            .document(test_document("doc1", "Document text."))
            .build();

    let history = request.chat_history_with_documents();
    let history = history.iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 6);
    assert!(matches!(
        history[0],
        Message::System { content } if content == "System one"
    ));
    assert!(matches!(
        history[1],
        Message::System { content } if content == "System two"
    ));
    assert!(is_document_message(history[2], "doc1"));
    assert!(matches!(history[3], Message::User { .. }));
    assert!(matches!(history[4], Message::Assistant { .. }));
    assert!(matches!(history[5], Message::User { .. }));
}

#[test]
fn build_without_documents_keeps_message_order_unchanged() {
    let request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), Message::user("Prompt"))
            .message(Message::system("System prompt"))
            .message(Message::user("Earlier user turn"))
            .build();

    let history = request.chat_history.iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 3);
    assert!(matches!(
        history[0],
        Message::System { content } if content == "System prompt"
    ));
    assert!(matches!(history[1], Message::User { .. }));
    assert!(matches!(history[2], Message::User { .. }));
}

#[test]
fn chat_history_with_documents_places_documents_after_leading_system_messages() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("System prompt"),
            Message::assistant("Earlier assistant turn"),
            Message::user("Earlier user turn"),
            Message::user("Prompt"),
        ],
        documents: vec![test_document("doc1", "Document text.")],
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    assert_eq!(request.documents.len(), 1);

    let history = request.chat_history_with_documents();
    let history = history.iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 5);
    assert!(matches!(history[0], Message::System { .. }));
    assert!(is_document_message(history[1], "doc1"));
    assert!(matches!(history[2], Message::Assistant { .. }));
    assert!(matches!(history[3], Message::User { .. }));
    assert!(matches!(history[4], Message::User { .. }));
}

#[test]
fn chat_history_with_documents_places_documents_before_mid_conversation_system_messages() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("Leading system prompt"),
            Message::assistant("Earlier assistant turn"),
            Message::system("Mid-conversation instruction"),
            Message::user("Prompt"),
        ],
        documents: vec![test_document("doc1", "Document text.")],
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let history = request.chat_history_with_documents();
    let history = history.iter().collect::<Vec<_>>();
    assert_eq!(history.len(), 5);
    assert!(matches!(
        history[0],
        Message::System { content } if content == "Leading system prompt"
    ));
    assert!(is_document_message(history[1], "doc1"));
    assert!(matches!(history[2], Message::Assistant { .. }));
    assert!(matches!(
        history[3],
        Message::System { content } if content == "Mid-conversation instruction"
    ));
    assert!(matches!(history[4], Message::User { .. }));
}

#[test]
fn chat_history_with_documents_does_not_duplicate_documents() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("System prompt"),
            Message::user("Earlier user turn"),
            Message::assistant("Earlier assistant turn"),
            Message::user("Prompt"),
        ],
        documents: vec![test_document("doc1", "Document text.")],
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let history = request.chat_history_with_documents();
    let document_messages = history
        .iter()
        .filter(|message| is_document_message(message, "doc1"))
        .count();
    assert_eq!(document_messages, 1);
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"code":"rate_limit","message":"slow down"}}"#;
    let error = CompletionError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("fixture body should parse as valid JSON"),
        Some(serde_json::json!({
            "error": {
                "code": "rate_limit",
                "message": "slow down"
            }
        }))
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_status() {
    let body = r#"{"error":{"message":"too many requests"}}"#;
    let error = CompletionError::ProviderResponse(provider_response::ProviderResponseError::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
}

#[test]
fn completion_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = CompletionError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("provider exploded".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("provider exploded"));
    assert_eq!(error.provider_response_status(), None);
    assert!(error.provider_response_json().is_err());
}

#[test]
fn completion_error_provider_error_is_not_a_provider_response() {
    // `ProviderError` also carries Rig-generated diagnostics, so the helpers
    // must not report its string as a provider response body.
    let error = CompletionError::ProviderError("stream transport failed".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("no body is not an error"),
        None
    );
}

#[test]
fn completion_error_provider_response_helpers_with_http_non_success_body_and_status() {
    let body = r#"{"error":{"type":"invalid_request","message":"bad request"}}"#;
    let error = CompletionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
        http::StatusCode::BAD_REQUEST,
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": {
                "type": "invalid_request",
                "message": "bad request"
            }
        }))
    );
}

#[test]
fn completion_error_provider_response_helpers_with_unrelated_variant() {
    let error = CompletionError::ResponseError("failed to parse provider response".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("no body is not an error"),
        None
    );
}

#[test]
fn provider_response_json_returns_none_for_empty_preserved_body() {
    let error = CompletionError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(String::new()),
    );

    assert_eq!(error.provider_response_body(), Some(""));
    assert_eq!(
        error
            .provider_response_json()
            .expect("empty body is not a JSON parse error"),
        None
    );
}
