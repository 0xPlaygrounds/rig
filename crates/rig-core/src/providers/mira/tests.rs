use super::*;
use crate::completion::FinishReason;
use crate::completion::NormalizeCompletionResponse;
use crate::providers::openai::completion::OpenAICompatibleProvider;

/// Normalize a Mira wire response the way the shared completion path does,
/// threading Mira's own descriptor name through the conversion.
fn normalized(response: CompletionResponse) -> completion::CompletionResponse {
    response
        .normalize(MiraExt::PROVIDER_NAME)
        .expect("Mira response should convert")
}

#[test]
fn test_completion_response_conversion() {
    let mira_response = CompletionResponse::Structured {
        id: "resp_123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "deepseek-r1".to_string(),
        choices: vec![ChatChoice {
            message: RawMessage {
                role: "assistant".to_string(),
                content: "Test response".to_string(),
            },
            finish_reason: Some("stop".to_string()),
            index: Some(0),
        }],
        usage: Some(Usage {
            prompt_tokens: 10,
            total_tokens: 20,
        }),
    };

    let completion_response = normalized(mira_response);

    assert_eq!(
        completion_response.choice.first(),
        Some(&completion::AssistantContent::text("Test response"))
    );
    assert_eq!(completion_response.provider, "mira");
    assert_eq!(completion_response.response_id.as_deref(), Some("resp_123"));
    assert_eq!(completion_response.message_id, None);
    assert_eq!(completion_response.model.as_deref(), Some("deepseek-r1"));
    assert_eq!(
        completion_response.finish_reason(),
        Some(FinishReason::Stop)
    );
    assert_eq!(completion_response.usage.input_tokens, 10);
    assert_eq!(completion_response.usage.output_tokens, 10);
    assert_eq!(completion_response.usage.total_tokens, 20);
}

fn structured_response_with_finish_reason(finish_reason: &str) -> CompletionResponse {
    CompletionResponse::Structured {
        id: "resp_123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "deepseek-r1".to_string(),
        choices: vec![ChatChoice {
            message: RawMessage {
                role: "assistant".to_string(),
                content: "Test response".to_string(),
            },
            finish_reason: Some(finish_reason.to_string()),
            index: Some(0),
        }],
        usage: None,
    }
}

#[test]
fn mira_finish_reasons_normalize_and_preserve_unknowns() {
    for (wire, expected) in [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("max_tokens", FinishReason::Length),
        ("tool_calls", FinishReason::ToolCalls),
        ("function_call", FinishReason::ToolCalls),
        ("content_filter", FinishReason::ContentFilter),
        // A gateway-specific reason survives verbatim rather than reading
        // as a natural stop.
        (
            "ERROR_UPSTREAM",
            FinishReason::Other("ERROR_UPSTREAM".to_owned()),
        ),
    ] {
        let converted = normalized(structured_response_with_finish_reason(wire));

        assert_eq!(converted.finish_reason(), Some(expected), "wire: {wire}");
    }
}

#[test]
fn mira_simple_response_reports_no_metadata() {
    let converted = normalized(CompletionResponse::Simple("Test response".to_string()));

    assert_eq!(converted.provider, "mira");
    assert_eq!(converted.message_id, None);
    assert_eq!(converted.model, None);
    assert_eq!(converted.finish_reason(), None);
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::mira::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::mira::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

// Proves a non-success HTTP response from `/v1/chat/completions` preserves
// the provider's status + body through the `provider_response_*` helpers
// (issue #1931).
#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("deepseek-r1");
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
