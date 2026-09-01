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

/// rig#2210: the failed response's headers must reach the caller, on both the
/// unary and streaming paths.
///
/// Recorded against the real API, so this pins the whole chain — the
/// transport captures the response's headers, the shared driver carries them
/// through classification instead of reading one header and discarding the
/// map, and `provider_response_headers()` surfaces them. On `origin/main`
/// both cells return `None`.
///
/// The header asserted here is whatever the live 404 actually carried
/// (`content-type`), not `Retry-After`: a real 429 cannot be forced cheaply
/// or deterministically, so the rate-limit case is unit-tested against the
/// same code path in `completion_send::header_preservation_tests`. What a
/// live recording adds is proof that a genuine provider response — not a
/// hand-built `HeaderMap` — survives end to end.
#[tokio::test]
async fn nonexistent_model_error_preserves_response_headers() {
    with_anthropic_cassette(
        "error_envelope/nonexistent_model_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-rig-test");
            let request = model.completion_request("Say hi.").max_tokens(16).build();

            let error = model
                .completion(request)
                .await
                .expect_err("a nonexistent model should be a provider error");

            let headers = error
                .provider_response_headers()
                .expect("the failed response's headers should reach the caller");
            assert_eq!(
                headers
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json"),
                "recorded response headers should replay verbatim, got {headers:?}",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn nonexistent_model_streaming_error_preserves_response_headers() {
    with_anthropic_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-rig-test");
            let request = model.completion_request("Say hi.").max_tokens(16).build();

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

            let headers = error
                .provider_response_headers()
                .expect("the failed handshake's headers should reach the caller");
            assert_eq!(
                headers
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json"),
                "recorded response headers should replay verbatim, got {headers:?}",
            );
        },
    )
    .await;
}
