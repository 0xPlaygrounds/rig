//! Anthropic error-envelope preservation, recorded from the real API.
//!
//! Locks down that a non-2xx provider response keeps its status and raw
//! error body recoverable through `provider_response_status()` /
//! `provider_response_body()`, for both the unary Messages path and the
//! streaming path.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.
use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn nonexistent_model_error_preserves_status_and_body() {
    with_anthropic_cassette(
        "error_envelope/nonexistent_model_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-rig-test");
            let request = model.completion_request("Say hi.").max_tokens(16).build();

            let error = model
                .completion(request)
                .await
                .expect_err("a nonexistent model should be a provider error");

            let status = error
                .provider_response_status()
                .expect("provider status should be preserved");
            assert_eq!(status.as_u16(), 404, "unexpected status: {status}");

            let body = error
                .provider_response_json()
                .expect("provider error body should be JSON")
                .expect("provider error body should be present");
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], "not_found_error");
        },
    )
    .await;
}

#[tokio::test]
async fn nonexistent_model_streaming_error_preserves_status_and_body() {
    with_anthropic_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-rig-test");
            let request = model.completion_request("Say hi.").max_tokens(16).build();

            // The SSE connection opens lazily, so the HTTP error may surface
            // either from `stream()` itself or as the first stream item.
            let error = match model.stream(request).await {
                Err(error) => error,
                Ok(mut stream) => match stream.next().await {
                    Some(Err(error)) => error,
                    Some(Ok(item)) => {
                        panic!("expected a provider error, got stream item: {item:?}")
                    }
                    None => panic!("stream ended without surfacing the provider error"),
                },
            };

            let status = error
                .provider_response_status()
                .expect("provider status should be preserved");
            assert_eq!(status.as_u16(), 404, "unexpected status: {status}");

            let body = error
                .provider_response_json()
                .expect("provider error body should be JSON")
                .expect("provider error body should be present");
            assert_eq!(body["error"]["type"], "not_found_error");
        },
    )
    .await;
}
