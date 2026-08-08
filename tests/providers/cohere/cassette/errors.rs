//! Cassette-backed coverage for Cohere error preservation.
//!
//! Cohere reports failures as `{"id": ..., "message": ...}` rather than the
//! `{"error": {...}}` envelope most providers use, so the raw body is the only
//! place the reason for the failure appears.

use axum::http;
use rig::completion::{CompletionError, CompletionModel};
use rig::prelude::*;

use super::super::support::with_cohere_cassette;
use crate::support::BASIC_PROMPT;

const UNKNOWN_MODEL: &str = "command-does-not-exist";

#[tokio::test]
async fn completion_error_preserves_status_and_body() {
    with_cohere_cassette(
        "errors/completion_error_preserves_status_and_body",
        |client| async move {
            let model = client.completion_model(UNKNOWN_MODEL);
            let request = model.completion_request(BASIC_PROMPT).build();

            let error = model
                .completion(request)
                .await
                .expect_err("an unknown model should fail");

            assert!(
                matches!(error, CompletionError::HttpError(_)),
                "expected an HTTP error, got {error:?}"
            );
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::NOT_FOUND)
            );

            let body = error
                .provider_response_body()
                .expect("the Cohere error body should be preserved");
            assert!(
                body.contains(UNKNOWN_MODEL) && body.contains("not found"),
                "expected the Cohere error message to survive, got {body}"
            );

            let json = error
                .provider_response_json()
                .expect("the Cohere error body should be valid JSON")
                .expect("the Cohere error body should be present");
            assert!(
                json.get("message")
                    .and_then(serde_json::Value::as_str)
                    .is_some(),
                "expected Cohere's bare `message` envelope, got {json}"
            );
        },
    )
    .await;
}
