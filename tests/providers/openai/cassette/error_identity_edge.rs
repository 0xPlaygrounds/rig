//! Error-taxonomy identity coverage on OpenAI (rig#2314 / PR #2315
//! follow-up): auth, validation, reference errors, and the streaming
//! handshake, on both APIs where the class differs.

use futures::StreamExt;
use rig::completion::{CompletionError, CompletionModel};
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
            let model = client.completion_model(openai::GPT_4O);
            let error = model
                .completion_request("Never authenticated")
                .send()
                .await
                .expect_err("a bogus key must be rejected");
            assert!(matches!(error, CompletionError::ProviderResponse(_)));
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
            let model = client.completion_model(openai::GPT_4O);
            let error = model
                .completion_request("Continue the conversation")
                .additional_params(serde_json::json!({
                    "previous_response_id": "resp_000000000000000000000000000000000000000000000000",
                }))
                .send()
                .await
                .expect_err("a nonexistent previous_response_id must be rejected");
            assert!(matches!(error, CompletionError::ProviderResponse(_)));
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
            let model = client.completion_model(openai::GPT_4O);
            let error = model
                .completion_request("Never validated")
                .additional_params(serde_json::json!({"temperature": 99.0}))
                .send()
                .await
                .expect_err("an impossible temperature must be rejected");
            assert!(matches!(error, CompletionError::ProviderResponse(_)));
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
            let model = client.completion_model("gpt-nonexistent-model-for-error-edge");
            let result = model.completion_request("Never streamed").stream().await;
            let error = match result {
                Err(error) => error,
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
            assert!(matches!(error, CompletionError::ProviderResponse(_)));
            assert_transport_request_id(error.provider_request_id(), "streaming connect 4xx");
            assert!(error.provider_response_body().is_some());
        },
    )
    .await;
}

/// Family C: the embeddings capability's error enum reads the new transport
/// variant — status and body survive a 4xx; embeddings has no request-id
/// contract today, so the accessor answers `None` (the recording documents
/// whether the header was nonetheless on the wire — see the PR findings).
#[tokio::test]
async fn embeddings_error_preserves_status_and_body() {
    use rig::client::EmbeddingsClient;

    with_openai_cassette(
        "error_identity_edge/embeddings_error_preserves_status_and_body",
        |client| async move {
            let model = client.embedding_model("text-embedding-nonexistent-model");
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
            assert_eq!(error.provider_request_id(), None);
        },
    )
    .await;
}

/// Family C: the model-listing 401 live — the #2315 review's P1 fix on the
/// wire: an auth failure classifies as `ApiError` with provider and path
/// context, not a `RequestError` fallback.
#[tokio::test]
async fn model_listing_auth_failure_keeps_api_error_context() {
    use rig::client::ModelListingClient;
    use rig::model::ModelListingError;

    with_openai_cassette_bogus_key(
        "error_identity_edge/model_listing_auth_failure_keeps_api_error_context",
        |client| async move {
            let error = client
                .list_models()
                .await
                .expect_err("a bogus key must fail the listing");
            assert!(
                matches!(error, ModelListingError::ApiError { .. }),
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
    use rig::client::VerifyClient;

    with_openai_cassette_bogus_key(
        "error_identity_edge/verify_reports_invalid_authentication",
        |client| async move {
            let error = client
                .verify()
                .await
                .expect_err("a bogus key must fail verification");
            assert!(
                matches!(error, rig::client::VerifyError::InvalidAuthentication),
                "verify's 401 arm is live: {error:?}"
            );
        },
    )
    .await;
}

/// Family C: an extractor run against a failing model — the failure chain
/// exposes the identity accessors through `ExtractionError`'s prompt-error
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
            let extractor = client
                .extractor::<Probe>("gpt-nonexistent-model-for-error-edge")
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
