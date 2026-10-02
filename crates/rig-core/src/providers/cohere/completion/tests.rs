use super::*;
use crate::completion::CompletionRequest;
use serde_path_to_error::deserialize;

#[test]
fn test_deserialize_completion_response() {
    let json_data = r#"
        {
            "id": "abc123",
            "message": {
                "role": "assistant",
                "tool_plan": "I will use the subtract tool to find the difference between 2 and 5.",
                "tool_calls": [
                        {
                            "id": "subtract_sm6ps6fb6y9f",
                            "type": "function",
                            "function": {
                                "name": "subtract",
                                "arguments": "{\"x\":5,\"y\":2}"
                            }
                        }
                    ]
                },
                "finish_reason": "TOOL_CALL",
                "usage": {
                "billed_units": {
                    "input_tokens": 78,
                    "output_tokens": 27
                },
                "tokens": {
                    "input_tokens": 1028,
                    "output_tokens": 63
                }
            }
        }
        "#;

    let mut deserializer = serde_json::Deserializer::from_str(json_data);
    let result: Result<CompletionResponse, _> = deserialize(&mut deserializer);

    let response = result.unwrap();
    let (_, citations, tool_calls) = response.message().expect("assistant message");
    let CompletionResponse {
        id,
        finish_reason,
        usage,
        ..
    } = response;

    assert_eq!(id, "abc123");
    assert_eq!(finish_reason, FinishReason::ToolCall);

    let Usage {
        billed_units,
        tokens,
        ..
    } = serde_json::from_value::<Usage>(usage.unwrap()).unwrap();
    let BilledUnits {
        input_tokens: billed_input_tokens,
        output_tokens: billed_output_tokens,
        ..
    } = billed_units.unwrap();
    let Tokens {
        input_tokens,
        output_tokens,
    } = tokens.unwrap();

    assert_eq!(billed_input_tokens.unwrap(), 78.0);
    assert_eq!(billed_output_tokens.unwrap(), 27.0);
    assert_eq!(input_tokens.unwrap(), 1028.0);
    assert_eq!(output_tokens.unwrap(), 63.0);

    assert!(citations.is_empty());
    assert_eq!(tool_calls.len(), 1);

    let ToolCallFunction { name, arguments } = tool_calls[0].function.clone().unwrap();

    assert_eq!(name, "subtract");
    assert_eq!(arguments, serde_json::json!({"x": 5, "y": 2}));
}

#[test]
fn finish_reason_maps_every_documented_wire_value() {
    assert_eq!(
        map_finish_reason(&FinishReason::Complete),
        completion::FinishReason::Stop
    );
    assert_eq!(
        map_finish_reason(&FinishReason::StopSequence),
        completion::FinishReason::Stop
    );
    assert_eq!(
        map_finish_reason(&FinishReason::MaxTokens),
        completion::FinishReason::Length
    );
    assert_eq!(
        map_finish_reason(&FinishReason::ToolCall),
        completion::FinishReason::ToolCalls
    );
    assert_eq!(
        map_finish_reason(&FinishReason::Error),
        completion::FinishReason::Other("ERROR".to_owned())
    );
}

#[test]
fn every_documented_finish_reason_maps_and_the_rest_fail() {
    use completion::FinishReason as Normalized;
    let failed = |reason: &str| Normalized::Other(reason.to_owned());
    for (wire, expected) in [
        ("COMPLETE", Normalized::Stop),
        ("STOP_SEQUENCE", Normalized::Stop),
        ("MAX_TOKENS", Normalized::Length),
        ("TOOL_CALL", Normalized::ToolCalls),
        ("ERROR", failed("ERROR")),
        ("ERROR_TOXIC", failed("ERROR_TOXIC")),
        ("ERROR_LIMIT", failed("ERROR_LIMIT")),
        ("USER_CANCEL", failed("USER_CANCEL")),
        ("TIMEOUT", failed("TIMEOUT")),
        ("X_RIG_INVENTED", failed("X_RIG_INVENTED")),
    ] {
        let reason: FinishReason =
            serde_json::from_value(serde_json::json!(wire)).expect("every reason deserializes");
        assert_eq!(map_finish_reason(&reason), expected, "{wire}");
    }
}

/// Fold one `/v2/chat` reply body through the bound chat wire, the way a
/// caller's `completion()` does.
async fn unary(body: &'static str) -> completion::CompletionResponse {
    let model = crate::driver::Model::new(
        crate::providers::cohere::CohereConfig::new("test-key")
            .completion(crate::providers::cohere::COMMAND_A_03_2025),
        crate::test_utils::RecordingHttpClient::new(body),
    );
    model
        .call(CompletionRequest::new("hello"))
        .await
        .expect("the reply decodes")
}

#[tokio::test]
async fn tool_call_response_normalizes_to_tool_calls_finish_reason() {
    let normalized = unary(
        r#"{
                "id": "abc123",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "subtract_1",
                        "type": "function",
                        "function": {"name": "subtract", "arguments": "{\"x\":5,\"y\":2}"}
                    }]
                },
                "finish_reason": "TOOL_CALL",
                "usage": {"tokens": {"input_tokens": 10, "output_tokens": 4}}
            }"#,
    )
    .await;

    assert_eq!(normalized.provider(), PROVIDER_NAME);
    assert_eq!(normalized.response_id(), Some("abc123"));
    assert_eq!(normalized.model(), None);
    assert_eq!(
        normalized.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
    assert_eq!(normalized.usage.input_tokens, Some(10));
    assert_eq!(normalized.usage.output_tokens, Some(4));
    assert_eq!(normalized.usage.total_tokens, Some(14));
    let Some(completion::AssistantContent::ToolCall(call)) = normalized.choice.first() else {
        panic!("expected a tool call, got {:?}", normalized.choice);
    };
    assert_eq!(
        call.id.provider().map(|provider| provider.as_str()),
        Some("subtract_1")
    );
    assert_eq!(call.function.name, "subtract");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"x": 5, "y": 2})
    );
}

#[test]
fn usage_is_mapped_from_tokens_and_carries_cached_input() {
    let usage: Usage = serde_json::from_str(
        r#"{
                "billed_units": {"input_tokens": 135, "output_tokens": 24},
                "cached_tokens": 112,
                "tokens": {"input_tokens": 1610, "output_tokens": 56}
            }"#,
    )
    .expect("usage should deserialize");

    let mapped = crate::completion::Usage::from(&usage);
    assert_eq!(mapped.input_tokens, Some(1610));
    assert_eq!(mapped.output_tokens, Some(56));
    assert_eq!(mapped.total_tokens, Some(1666));
    assert_eq!(mapped.cached_input_tokens, Some(112));
}

#[tokio::test]
async fn response_usage_matches_the_canonical_mapping() {
    const BODY: &str = r#"{
                "id": "abc123",
                "finish_reason": "COMPLETE",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                "usage": {
                    "billed_units": {"input_tokens": 135, "output_tokens": 24},
                    "cached_tokens": 112,
                    "tokens": {"input_tokens": 1610, "output_tokens": 56}
                }
            }"#;
    let response: CompletionResponse =
        serde_json::from_str(BODY).expect("response should deserialize");
    let expected = usage_of(response.usage.as_ref().expect("usage should be present"));

    let converted = unary(BODY).await;

    assert_eq!(converted.usage, expected);
    assert_eq!(converted.usage.input_tokens, Some(1610));
    assert_eq!(converted.usage.cached_input_tokens, Some(112));
    assert_eq!(
        converted
            .choice
            .first()
            .map(crate::message::AssistantContent::canonical),
        Some(completion::AssistantContent::text("hi"))
    );
}

/// Without `tokens` nothing is reported — including `cached_tokens`, which
/// is a subset of an input count Cohere did not send.
#[test]
fn usage_without_token_counts_is_unreported() {
    let usage: Usage = serde_json::from_str("{}").expect("usage should deserialize");
    assert_eq!(
        crate::completion::Usage::from(&usage),
        crate::completion::Usage::default()
    );

    let cached_only: Usage =
        serde_json::from_str(r#"{"cached_tokens": 512}"#).expect("usage should deserialize");
    assert_eq!(
        crate::completion::Usage::from(&cached_only),
        crate::completion::Usage::default()
    );
}

#[test]
fn tool_result_content_is_type_tagged() {
    let text = serde_json::to_value(ToolResultContent::Text {
        text: "-3".to_owned(),
    })
    .expect("tool result text content should serialize");
    assert_eq!(text, serde_json::json!({"type": "text", "text": "-3"}));

    let document = serde_json::to_value(ToolResultContent::Document {
        document: Document {
            id: "doc_1".to_owned(),
            data: HashMap::from([("text".to_owned(), "-3".into())]),
        },
    })
    .expect("tool result document content should serialize");
    assert_eq!(
        document,
        serde_json::json!({
            "type": "document",
            "document": {"id": "doc_1", "data": {"text": "-3"}}
        })
    );

    let roundtrip: ToolResultContent =
        serde_json::from_value(text).expect("tool result content should deserialize");
    assert_eq!(
        roundtrip,
        ToolResultContent::Text {
            text: "-3".to_owned()
        }
    );
}

#[test]
fn cohere_builder_request_serializes_documents_in_cohere_shape() {
    let request = crate::completion::CompletionRequest::new("What is glarb-glarb?").document(
        crate::completion::request::Document {
            id: "doc_1".to_string(),
            text: "Definition of glarb-glarb: an ancient tool.".to_string(),
            additional_props: HashMap::from([("source".to_string(), "field-notes".to_string())]),
        },
    );

    let request = CohereCompletionRequest::try_from(("command-a-03-2025", request))
        .expect("request conversion should succeed");

    assert_eq!(request.documents.len(), 1);
    assert_eq!(request.documents[0].id, "doc_1");

    let documents = serde_json::to_value(&request.documents)
        .expect("documents should serialize")
        .as_array()
        .cloned()
        .expect("documents should serialize as an array");
    assert_eq!(
        documents[0],
        serde_json::json!({
            "id": "doc_1",
            "data": {
                "text": "Definition of glarb-glarb: an ancient tool.",
                "source": "field-notes"
            }
        })
    );
}

#[test]
fn tool_choice_serializes_as_a_bare_cohere_string() {
    assert_eq!(
        serde_json::to_value(CohereToolChoice::Required).expect("serialize"),
        serde_json::json!("REQUIRED")
    );
    assert_eq!(
        serde_json::to_value(CohereToolChoice::None).expect("serialize"),
        serde_json::json!("NONE")
    );

    assert_eq!(
        CohereToolChoice::try_from(ToolChoice::Required).expect("required is supported"),
        CohereToolChoice::Required
    );
    assert_eq!(
        CohereToolChoice::try_from(ToolChoice::None).expect("none is supported"),
        CohereToolChoice::None
    );
}

#[test]
fn unsupported_tool_choices_are_rejected_before_the_request_is_sent() {
    for unsupported in [
        ToolChoice::Auto,
        ToolChoice::Specific {
            function_names: vec![crate::message::ToolName::new("subtract").expect("tool name")],
        },
    ] {
        let error = ProviderError::from(
            CohereToolChoice::try_from(unsupported.clone())
                .expect_err("Cohere has no encoding for this tool choice"),
        );
        assert!(
            matches!(error, ProviderError::Request(_)),
            "expected a request error for {unsupported:?}, got {error:?}"
        );
    }
}

/// Invalid REQUIRED requests cannot produce a cassette because validation
/// must stop them before the HTTP boundary.
#[tokio::test]
async fn required_tool_choice_without_tools_is_rejected_before_the_request_is_sent() {
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::new("{}");
    let model = crate::driver::Model::new(
        crate::providers::cohere::CohereConfig::new("test-key")
            .completion(crate::providers::cohere::COMMAND_A_03_2025),
        http_client.clone(),
    );
    let request = CompletionRequest::new("hello").tool_choice(ToolChoice::Required);

    let error = model
        .call(request)
        .await
        .expect_err("REQUIRED without tools should fail locally");

    assert!(matches!(error, ProviderError::Request(_)));
    let message = error.to_string();
    assert!(
        message.contains("at least one tool") && message.contains("REQUIRED"),
        "unexpected error: {error:?}"
    );
    assert!(
        http_client.requests().is_empty(),
        "invalid requests must fail before reaching the HTTP client"
    );
}

/// This internal unit test protects the raw provider-parameter escape hatch;
/// cassette coverage exercises the public typed-tool path instead.
#[test]
fn required_tool_choice_accepts_raw_tools_from_additional_params() {
    let request = crate::completion::CompletionRequest::new("hello")
        .tool_choice(ToolChoice::Required)
        .additional_params(serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Return pong",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        }));

    let request = CohereCompletionRequest::try_from(("command-a-03-2025", request))
        .expect("raw Cohere tools should satisfy REQUIRED");
    let body = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(body["tool_choice"], serde_json::json!("REQUIRED"));
    assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
}

#[test]
fn max_tokens_is_forwarded_and_omitted_when_unset() {
    let capped = crate::completion::CompletionRequest::new("hello").max_tokens(64);
    let capped = CohereCompletionRequest::try_from(("command-a-03-2025", capped))
        .expect("request conversion should succeed");
    let body = serde_json::to_value(&capped).expect("request should serialize");
    assert_eq!(body["max_tokens"], serde_json::json!(64));

    let uncapped = crate::completion::CompletionRequest::new("hello");
    let uncapped = CohereCompletionRequest::try_from(("command-a-03-2025", uncapped))
        .expect("request conversion should succeed");
    let body = serde_json::to_value(&uncapped).expect("request should serialize");
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn tool_choice_is_omitted_when_unset() {
    let request = crate::completion::CompletionRequest::new("hello");

    let request = CohereCompletionRequest::try_from(("command-a-03-2025", request))
        .expect("request conversion should succeed");
    let body = serde_json::to_value(&request).expect("request should serialize");

    assert!(body.get("tool_choice").is_none());
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = crate::driver::Model::new(
        crate::providers::cohere::CohereConfig::new("test-key")
            .completion(crate::providers::cohere::COMMAND_A_03_2025),
        http_client,
    );
    let request = CompletionRequest::new("hello");

    let error = model
        .call(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// Synthetic transcript tests required-ID request correlation without a paid call.
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    use crate::providers::internal::wire_ids::tests::{adapter_requests, assert_adapter_pairs};
    for request in adapter_requests() {
        let wire =
            CohereCompletionRequest::try_from(("command-a-03-2025", request.clone())).unwrap();
        assert_adapter_pairs(serde_json::to_value(wire).unwrap());
    }
}

/// Cohere rejects its own stream's empty `tool_plan` on replay: "invalid
/// message provided at index N: must have non-empty content or tool calls"
/// (HTTP 400, probed 2026-10-01). The plan is left out; everything else of
/// the message goes back as it came.
#[test]
fn a_replayed_message_is_rebuilt_from_its_blocks() {
    let streamed = serde_json::json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
        "citations": [{"start": 0, "end": 4, "text": "done", "sources": []}],
        "tool_calls": [],
        "tool_plan": "",
    });
    let turn = message::AssistantMessage::new(vec![message::AssistantContent::text("done")])
        .with_native(streamed);
    let converted = Vec::<Message>::try_from(message::Message::Assistant(turn)).expect("converts");
    let Some(Message::Native(sent)) = converted.first() else {
        panic!("the rebuilt message: {converted:?}");
    };
    assert_eq!(
        *sent,
        serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "tool_calls": [],
        })
    );
}
