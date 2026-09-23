//! Error-taxonomy identity coverage on OpenAI (rig#2314 / PR #2315
//! follow-up): auth, validation, reference errors, and the streaming
//! handshake, on both APIs where the class differs.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::error::ProviderError;
use rig::error::{ErrorKind, ErrorReport};
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::{
    with_openai_cassette, with_openai_cassette_bogus_key, with_openai_completions_cassette,
};
use crate::support::assert_transport_request_id;

/// 401 auth rejection — the fixture documents whether OpenAI's auth tier
/// sends `x-request-id` (assertion derived from the recording).
#[tokio::test]
async fn auth_rejection_carries_identity() {
    with_openai_cassette_bogus_key(
        "error_identity_edge/auth_rejection_carries_identity",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let error = model
                .completion_request("Never authenticated")
                .send()
                .await
                .expect_err("a bogus key must be rejected");
            assert!(matches!(error, ProviderError::ProviderResponse(_)));
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(401)
            );
            assert_transport_request_id(error.provider_request_id(), "401 error");
        },
    )
    .await;
}

/// 404 *reference* error: a `previous_response_id` naming a nonexistent
/// response — the id-reference path, distinct from model-not-found. The
/// error belongs to the failing call, never the referenced response.
#[tokio::test]
async fn nonexistent_previous_response_reference_carries_identity() {
    with_openai_cassette(
        "error_identity_edge/nonexistent_previous_response_reference_carries_identity",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let error = model
                .completion_request("Continue the conversation")
                .additional_params(serde_json::json!({
                    "previous_response_id": "resp_000000000000000000000000000000000000000000000000",
                }))
                .send()
                .await
                .expect_err("a nonexistent previous_response_id must be rejected");
            assert!(matches!(error, ProviderError::ProviderResponse(_)));
            assert_transport_request_id(error.provider_request_id(), "reference error");
            // The referenced id is scrub-placeholdered in replay; assert on
            // the error *class*, never the literal id.
            assert!(
                error
                    .provider_response_body()
                    .is_some_and(|body| body.contains("previous_response_not_found")),
                "the error names the failing reference class: {error:?}"
            );
        },
    )
    .await;
}

/// 400 validation on the Chat Completions API (the second unary API's
/// validation tier).
#[tokio::test]
async fn chat_completions_validation_error_carries_identity() {
    with_openai_completions_cassette(
        "error_identity_edge/chat_completions_validation_error_carries_identity",
        |client| async move {
            let model = client.chat(openai::GPT_4O);
            let error = model
                .completion_request("Never validated")
                .additional_params(serde_json::json!({"temperature": 99.0}))
                .send()
                .await
                .expect_err("an impossible temperature must be rejected");
            assert!(matches!(error, ProviderError::ProviderResponse(_)));
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(400)
            );
            assert_transport_request_id(error.provider_request_id(), "chat 400 error");
        },
    )
    .await;
}

/// Streaming connect 4xx on the Responses API: post-fix, the handshake error
/// matches its blocking twin — ProviderResponse with status, body, and id.
#[tokio::test]
async fn streaming_connect_4xx_matches_blocking_richness() {
    with_openai_cassette(
        "error_identity_edge/streaming_connect_4xx_matches_blocking_richness",
        |client| async move {
            let model = client
                .openai
                .completion("gpt-nonexistent-model-for-error-edge");
            let result = model.completion_request("Never streamed").stream().await;
            let error = match result {
                Err(error) => ErrorReport::from(&error),
                Ok(mut stream) => {
                    let mut yielded = None;
                    while let Some(item) = stream.next().await {
                        if let Err(error) = item {
                            yielded = Some(error);
                            break;
                        }
                    }
                    yielded.expect("the failed handshake must surface an error")
                }
            };
            assert_eq!(error.kind, ErrorKind::ProviderResponse, "error: {error}");
            assert_transport_request_id(error.provider_request_id(), "streaming connect 4xx");
            assert!(error.provider_response_body().is_some());
        },
    )
    .await;
}

/// Family C: an embeddings 4xx keeps the provider's status and body, and
/// carries the transport request id the recording shows on the wire — the
/// embeddings wire reports `x-request-id` like every other route on this
/// dialect, so the id is no longer dropped on the way to the caller.
#[tokio::test]
async fn embeddings_error_preserves_status_and_body() {
    with_openai_cassette(
        "error_identity_edge/embeddings_error_preserves_status_and_body",
        |client| async move {
            let model = client
                .openai
                .embedding("text-embedding-nonexistent-model", None);
            let error = model
                .embed_text("never embedded")
                .await
                .expect_err("a nonexistent embedding model must fail");
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(404),
                "the capability error reads the details-preserving variant: {error:?}"
            );
            assert!(error.provider_response_body().is_some());
            assert_transport_request_id(error.provider_request_id(), "embeddings 404");
        },
    )
    .await;
}

/// Family C: the model-listing 401 live — the #2315 review's P1 fix on the
/// wire: an auth failure keeps the provider's reply with provider and path
/// context, not a transport-error fallback.
#[tokio::test]
async fn model_listing_auth_failure_keeps_api_error_context() {
    use rig::model::ModelLister;

    with_openai_cassette_bogus_key(
        "error_identity_edge/model_listing_auth_failure_keeps_api_error_context",
        |client| async move {
            let error = client
                .openai
                .models()
                .list_all()
                .await
                .expect_err("a bogus key must fail the listing");
            assert!(
                matches!(error, ProviderError::ProviderResponse(_)),
                "the review's P1 fix, proven live: {error:?}"
            );
            let message = error.to_string();
            assert!(
                message.contains("openai") && message.contains("401"),
                "provider and status context survive: {message}"
            );
        },
    )
    .await;
}

/// Family C: `verify()` against a bogus key — the review flagged its
/// status-match arms as possibly dead under the erroring transport; this
/// recording settles what actually happens (assertion derived from it).
#[tokio::test]
async fn verify_reports_invalid_authentication() {
    with_openai_cassette_bogus_key(
        "error_identity_edge/verify_reports_invalid_authentication",
        |client| async move {
            let error = client
                .openai
                .verify()
                .await
                .expect_err("a bogus key must fail verification");
            assert!(
                matches!(
                    &error,
                    ProviderError::InvalidAuthentication(response)
                        if response.status.map(|status| status.as_u16()) == Some(401)
                ),
                "verify's 401 arm is live: {error:?}"
            );
        },
    )
    .await;
}

/// Family C: an extractor run against a failing model — the failure chain
/// exposes the identity accessors through `StructuredOutputError`'s prompt-error
/// wrap (or documents the gap).
#[tokio::test]
async fn extractor_failure_surfaces_provider_error_context() {
    #[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Probe {
        value: i64,
    }

    with_openai_cassette(
        "error_identity_edge/extractor_failure_surfaces_provider_error_context",
        |client| async move {
            let extractor = rig::extractor::ExtractorBuilder::<Probe>::new(
                client
                    .openai
                    .completion("gpt-nonexistent-model-for-error-edge"),
            )
            .build();
            let error = extractor
                .extract("never extracted")
                .await
                .expect_err("a nonexistent model must fail extraction");
            let message = error.to_string();
            assert!(
                message.contains("model") || message.contains("404") || message.contains("400"),
                "the provider failure surfaces through the extractor chain: {message}"
            );
        },
    )
    .await;
}
