//! Venice error-envelope preservation, recorded from the real API.
//!
//! Venice answers failures with a *flat* `{"error": "…"}` body rather than
//! OpenAI's nested `{"error": {"message": …}}`. This locks down that the
//! shared envelope still classifies it as an error and that the raw body and
//! status survive to the caller, on both the unary and streaming paths.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_venice_cassette;

#[tokio::test]
async fn nonexistent_model_error_preserves_status_and_body() {
    with_venice_cassette(
        "error_envelope/nonexistent_model_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("venice-nonexistent-rig-test");
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
            let message = body["error"]
                .as_str()
                .expect("Venice reports a flat string error message");
            assert!(
                message.contains("venice-nonexistent-rig-test"),
                "error message should name the rejected model, got {message:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn nonexistent_model_streaming_error_preserves_status_and_body() {
    with_venice_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("venice-nonexistent-rig-test");
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
                    None => panic!("expected a provider error, got an empty stream"),
                },
            };

            let status = error
                .provider_response_status()
                .expect("provider status should be preserved");
            assert_eq!(status.as_u16(), 404, "unexpected status: {status}");
            assert!(
                error
                    .provider_response_body()
                    .is_some_and(|body| body.contains("venice-nonexistent-rig-test")),
                "streaming error should preserve Venice's raw body"
            );
        },
    )
    .await;
}
