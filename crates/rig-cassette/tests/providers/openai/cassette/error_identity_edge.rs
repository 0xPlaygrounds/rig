//! Error-taxonomy identity coverage on OpenAI (rig#2314 / PR #2315
//! follow-up): auth, validation, reference errors, and the streaming
//! handshake, on both APIs where the class differs.

use rig::error::ProviderError;

use super::super::support::{with_openai_cassette, with_openai_cassette_bogus_key};
use crate::support::assert_transport_request_id;

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
    with_openai_cassette_bogus_key(
        "error_identity_edge/model_listing_auth_failure_keeps_api_error_context",
        |client| async move {
            let error = client
                .openai
                .list_models()
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
