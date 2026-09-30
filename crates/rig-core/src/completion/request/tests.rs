use super::{CompletionResponse, FinishReason, ProviderCapabilities, Usage};
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::message::AssistantContent;
use crate::{http_client, provider_response};

/// An empty conversation, message or tool result parses, and the request
/// boundary rejects it by role and index.
mod empty_lists_parse_and_are_rejected_when_sent {
    use crate::message::Message;
    use crate::test_utils::{MockCompletionModel, MockTurn};
    use serde_json::json;

    /// A request carrying each empty piece, and the text its rejection names.
    fn empty_requests() -> Vec<(super::CompletionRequest, &'static str)> {
        let with = |message: serde_json::Value| {
            super::CompletionRequest::new("hello")
                .message(serde_json::from_value::<Message>(message).expect("an empty list parses"))
        };
        let mut empty_history = super::CompletionRequest::new("hello");
        empty_history.chat_history.clear();
        vec![
            (empty_history, "request has an empty chat history"),
            (
                with(json!({"role": "user", "content": []})),
                "user message at index 0 has no content",
            ),
            (
                with(json!({"role": "assistant", "id": null, "content": []})),
                "assistant message at index 0 has no content",
            ),
            (
                with(json!({
                    "role": "user",
                    "content": [{
                        "type": "toolresult",
                        "call": {"provider": {"call_id": "call_1"}},
                        "name": "lookup",
                        "content": [],
                    }],
                })),
                "tool result for `lookup` at index 0 of the user message at index 0 has no content",
            ),
        ]
    }

    #[test]
    fn an_empty_history_parses() {
        let mut request =
            serde_json::to_value(super::CompletionRequest::new("hello")).expect("serializes");
        request["chat_history"] = json!([]);
        let request = serde_json::from_value::<super::CompletionRequest>(request)
            .expect("an empty history parses");
        assert!(request.chat_history.is_empty());
    }

    #[test]
    fn each_empty_piece_is_rejected_by_role_and_index() {
        for (request, expected) in empty_requests() {
            let error = request
                .validate_message_content()
                .expect_err("the request is rejected");
            assert!(matches!(error, super::ProviderError::Request(_)), "{error}");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    /// `Model::call`, `Model::stream` and their erased twins reject each
    /// empty piece before the scripted runtime sees the request.
    #[tokio::test]
    async fn every_model_surface_rejects_each_empty_piece_before_the_transport() {
        for (request, expected) in empty_requests() {
            let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
            let erased = model.clone().erase();
            let errors = [
                model.call(request.clone()).await.err(),
                model.stream(request.clone()).err(),
                erased.call(request.clone()).await.err(),
                erased.stream(request.clone()).err(),
            ];
            for error in errors {
                let error = error.expect("the request is rejected");
                assert!(error.to_string().contains(expected), "{error}");
            }
            assert_eq!(model.request_count(), 0, "nothing reached the transport");
            assert_eq!(model.script().len(), 1, "the scripted turn is unused");
        }
    }

    /// Deserializing `[]` is not where the rule lives: the message reads and
    /// writes the same JSON, and sending it is what fails.
    #[tokio::test]
    async fn an_empty_content_list_round_trips_and_is_rejected_when_sent() {
        let json = json!({"role": "user", "content": []});
        let message = serde_json::from_value::<Message>(json.clone()).expect("parses");
        assert_eq!(serde_json::to_value(&message).expect("serializes"), json);

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let error = model
            .call(super::CompletionRequest::new(message))
            .await
            .expect_err("sending it is rejected");
        assert!(
            error
                .to_string()
                .contains("user message at index 0 has no content"),
            "{error}"
        );
        assert_eq!(model.request_count(), 0);
    }

    #[test]
    fn a_tool_result_with_one_empty_string_block_parses_and_is_accepted() {
        let result = json!({
            "role": "user",
            "content": [{
                "type": "toolresult",
                "call": {"provider": {"call_id": "call_1"}},
                "name": "lookup",
                "content": [{"type": "text", "text": ""}],
            }],
        });
        let message = serde_json::from_value::<Message>(result).expect("parses");
        assert!(
            super::CompletionRequest::new(message)
                .validate_message_content()
                .is_ok()
        );
    }
}

/// The request-boundary check accepts what providers accept.
mod message_content {
    use crate::message::{CallId, Message, ToolName, ToolResultContent, UserContent};

    #[test]
    fn a_request_with_content_in_every_message_is_accepted() {
        let request = super::CompletionRequest::new("hello")
            .message(Message::assistant("hi"))
            .preamble("be brief");
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn an_empty_system_message_is_not_checked() {
        let request = super::CompletionRequest::new("hello").preamble("");
        assert!(request.validate_message_content().is_ok());
    }

    #[test]
    fn a_tool_result_with_one_empty_text_block_is_accepted() {
        let result = UserContent::tool_result(
            CallId::from_wire("call_1"),
            ToolName::new("lookup").expect("tool name"),
            vec![ToolResultContent::text("")],
        );
        let request = super::CompletionRequest::new(Message::from(result));
        assert!(request.validate_message_content().is_ok());
    }
}

fn tool_call_choice() -> Vec<AssistantContent> {
    vec![AssistantContent::tool_call(
        "call_1",
        crate::message::ToolName::new("lookup").expect("tool name"),
        serde_json::json!({"query": "rig"}),
    )]
}

#[test]
fn normalized_response_round_trips_through_serde() {
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage {
                input_tokens: Some(3),
                output_tokens: Some(2),
                total_tokens: Some(5),
                cached_input_tokens: Some(1),
                cache_creation_input_tokens: Some(0),
                tool_use_prompt_tokens: Some(0),
                reasoning_tokens: Some(1),
            },
            "example",
            serde_json::json!({}),
        )
        .with_finish_reason(FinishReason::Stop);
        response.message_id = Some("msg_123".into());
        response.model = Some("provider-model-v2".into());
        response
    };

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
        Usage::default(),
        "example",
        serde_json::json!({}),
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
        Usage::default(),
        "example",
        serde_json::json!({}),
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
    let response = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        "example",
        serde_json::json!({}),
    )
    .with_finish_reason(FinishReason::Stop);

    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

/// The `Option` setter is what provider conversions actually reach for, so
/// it must reconcile identically — a provider holding an `Option` must not
/// have to choose between ergonomics and correctness.
#[test]
fn optional_setter_reconciles_exactly_like_the_plain_setter() {
    let via_option = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        "example",
        serde_json::json!({}),
    )
    .with_optional_finish_reason(Some(FinishReason::Stop));
    let via_plain = CompletionResponse::new(
        tool_call_choice(),
        Usage::default(),
        "example",
        serde_json::json!({}),
    )
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
        let response = CompletionResponse::new(
            tool_call_choice(),
            Usage::default(),
            "example",
            serde_json::json!({}),
        )
        .with_finish_reason(reason.clone());

        assert_eq!(response.finish_reason, Some(reason));
    }
}

#[test]
fn reconciliation_leaves_a_stop_without_tool_calls_alone() {
    let response = CompletionResponse::new(
        vec![AssistantContent::text("done")],
        Usage::default(),
        "example",
        serde_json::json!({}),
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
fn usage_is_reported_when_any_counter_is_present() {
    use super::Usage;

    assert!(!Usage::default().is_reported());

    let zero = Usage {
        reasoning_tokens: Some(0),
        ..Usage::default()
    };
    assert!(zero.is_reported(), "a reported zero is still a report");
}

#[test]
fn usage_sum_keeps_a_reported_counter_when_the_other_side_is_absent() {
    use super::Usage;

    let reported = Usage {
        input_tokens: Some(3),
        output_tokens: Some(0),
        ..Usage::default()
    };
    let sum = Usage::default() + reported + reported;
    assert_eq!(sum.input_tokens, Some(6));
    assert_eq!(sum.output_tokens, Some(0));
    assert_eq!(sum.total_tokens, None);
}

use super::*;

#[test]
fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
    let default_request = CompletionRequest::new("completion prompt");
    assert!(!default_request.record_telemetry_content);

    let default_json = serde_json::to_value(&default_request).expect("serialize request");
    assert!(
        default_json.get("record_telemetry_content").is_none(),
        "safe default should not serialize the telemetry opt-in field"
    );
    let default_roundtrip: CompletionRequest =
        serde_json::from_value(default_json).expect("deserialize default request");
    assert!(!default_roundtrip.record_telemetry_content);

    let opt_in_request = CompletionRequest::new("completion prompt").record_content_telemetry(true);
    assert!(opt_in_request.record_telemetry_content);

    let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
    assert!(
        opt_in_json.get("record_telemetry_content").is_none(),
        "local telemetry policy must not be serialized into provider requests"
    );
    let without_field: CompletionRequest =
        serde_json::from_value(opt_in_json).expect("deserialize a request without the field");
    assert!(
        !without_field.record_telemetry_content,
        "missing field should deserialize to the safe default"
    );
}

/// The deserialization mirror carries `raw`: a response with a captured
/// payload survives serialize → deserialize with the payload intact, and a
/// response written without the field is refused rather than loaded with
/// `raw` invented.
#[test]
fn normalized_response_raw_round_trips_through_serde_mirror() {
    let payload = serde_json::json!({
        "id": "chatcmpl-1",
        "system_fingerprint": "fp_abc",
        "choices": [{"finish_reason": "stop"}]
    });
    let response = {
        let mut response = CompletionResponse::new(
            vec![AssistantContent::text("hello")],
            Usage::default(),
            "example",
            payload.clone(),
        );
        response.response_id = Some("chatcmpl-1".into());
        response
    };

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

    let without_raw = serde_json::json!({
        "choice": [{"type": "text", "text": "hello"}],
        "usage": serde_json::to_value(Usage::default()).unwrap(),
        "provider": "example"
    });
    let error = serde_json::from_value::<CompletionResponse>(without_raw)
        .expect_err("a response without `raw` is refused");
    assert!(error.to_string().contains("raw"), "{error}");
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
    let request = CompletionRequest::new("prompt")
        .preamble("system")
        .message(Message::user("history"))
        .document(test_document("doc1", "static context secret"));

    let messages = request.messages_for_telemetry();
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
    let request = CompletionRequest::new(Message::user("Prompt"))
        .preamble("System prompt")
        .message(Message::user("History"));

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
    let request = CompletionRequest::new(Message::user("Prompt"))
        .preamble("System prompt")
        .document(test_document("doc1", "Document text."));

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
    let request = CompletionRequest::new(Message::user("Prompt"))
        .message(Message::system("System one"))
        .message(Message::system("System two"))
        .message(Message::user("Earlier user turn"))
        .message(Message::assistant("Earlier assistant turn"))
        .document(test_document("doc1", "Document text."));

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
    let request = CompletionRequest::new(Message::user("Prompt"))
        .message(Message::system("System prompt"))
        .message(Message::user("Earlier user turn"));

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
    let error = ProviderError::ProviderResponse(
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
    let error = ProviderError::ProviderResponse(provider_response::ProviderResponseError::new(
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
    let error = ProviderError::ProviderResponse(
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
    let error = ProviderError::Provider("stream transport failed".to_string());

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
    let error = ProviderError::from_transport_error(http_client::Error::non_success_with_details(
        http::StatusCode::BAD_REQUEST,
        http::HeaderMap::new(),
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
    let error = ProviderError::Response("failed to parse provider response".to_string());

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
    let error = ProviderError::ProviderResponse(
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

mod additional_params_precedence {
    use super::super::shadowed_typed_fields;
    use serde_json::json;

    /// A passthrough key names a typed field only when that field is set too;
    /// an unset typed field is not shadowed, and a non-object passthrough
    /// shadows nothing.
    #[test]
    fn shadowed_keys_are_the_intersection_with_set_typed_fields() {
        let params = json!({"temperature": 0.9, "top_p": 0.5, "max_tokens": 10});
        let shadowed = shadowed_typed_fields(
            Some(&params),
            &[
                ("temperature", true),
                ("max_tokens", false),
                ("model", true),
            ],
        );
        assert_eq!(shadowed, ["temperature"]);
        assert!(shadowed_typed_fields(Some(&json!([1])), &[("temperature", true)]).is_empty());
        assert!(shadowed_typed_fields(None, &[("temperature", true)]).is_empty());
    }

    /// A second `additional_params(Some(..))` merges into the first rather
    /// than replacing it; `None` clears, like every other folded setter.
    #[test]
    fn additional_params_merges_and_none_clears() {
        let request = crate::completion::CompletionRequest::new("p")
            .additional_params(json!({"a": 1}))
            .additional_params(json!({"b": 2}))
            .temperature(None);
        assert_eq!(request.additional_params, Some(json!({"a": 1, "b": 2})));
        assert_eq!(request.temperature, None);
        let cleared = crate::completion::CompletionRequest::new("p")
            .additional_params(json!({"a": 1}))
            .additional_params(None)
            .additional_params(json!({"b": 2}));
        assert_eq!(cleared.additional_params, Some(json!({"b": 2})));
    }
}
