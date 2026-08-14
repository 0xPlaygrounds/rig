//! xAI completion support through its OpenAI-compatible Responses API.

pub use crate::providers::openai::responses_api::CompletionResponse;

use super::client::XAiExt;

/// xAI completion model, driven by the shared Responses implementation.
pub type CompletionModel<H = reqwest::Client> =
    crate::providers::openai::responses_api::GenericResponsesCompletionModel<XAiExt, H>;

/// xAI completion models.
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";
pub const GROK_4: &str = "grok-4-0709";

#[cfg(test)]
mod tests {
    use crate::client::CompletionClient;
    use crate::completion::{CompletionError, CompletionModel as _, CompletionRequestBuilder};
    use crate::http_client::HttpClientExt;
    use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

    fn assert_minimal_raw_stream_transport_bounds<H>(client: crate::providers::xai::Client<H>)
    where
        H: HttpClientExt + Clone + 'static,
    {
        let model = super::CompletionModel::new(client, super::GROK_4);
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "hello").build();

        std::mem::drop(model.raw_stream(request));
    }

    #[test]
    fn raw_stream_does_not_require_default_or_debug_transport_bounds() {
        let _: fn(crate::providers::xai::Client<RecordingHttpClient>) =
            assert_minimal_raw_stream_transport_bounds;
    }

    #[test]
    fn completion_keeps_xai_structured_output_capability() {
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .build()
            .expect("build client");
        let model = client.completion_model(super::GROK_4);

        assert!(!model.capabilities().composes_native_output_with_tools);
    }

    #[tokio::test]
    async fn completion_uses_xai_endpoint_and_request_shape() {
        let body = serde_json::json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "grok-4-0709",
            "usage": null,
            "output": [{
                "type": "message",
                "id": "msg_123",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "done",
                    "annotations": []
                }]
            }],
            "tools": []
        })
        .to_string();
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.completion_model(super::GROK_4);

        let response = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("completion should succeed");

        assert_eq!(response.provider, "xai");
        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].uri, "https://api.x.ai/v1/responses");
        let request: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(request["model"], super::GROK_4);
        assert_eq!(request["input"][0]["role"], "user");
        assert!(request.get("instructions").is_none());
    }

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(super::GROK_4);

        let error = model
            .completion(model.completion_request("hello").build())
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn completion_2xx_error_envelope_preserves_status_and_body() {
        let body = r#"{"error":"boom","code":"503"}"#;
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(RecordingHttpClient::new(body))
            .build()
            .expect("build client");
        let model = client.completion_model(super::GROK_4);

        let error = model
            .completion(model.completion_request("hello").build())
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
