//! Error-taxonomy identity coverage (rig#2314 / PR #2315 follow-up): each
//! failure class exercises a different provider code path; every recorded
//! error must keep its status, body, id (or recorded absence), and
//! contract classification.

use futures::StreamExt;
use rig::completion::{CompletionError, CompletionModel};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::{with_anthropic_cassette, with_anthropic_cassette_bogus_key};
use crate::support::assert_transport_request_id;

/// 401 auth rejection: the auth tier answers before the API proper — the
/// recorded fixture documents whether the id header still rides it.
#[tokio::test]
async fn auth_rejection_carries_identity() {
    with_anthropic_cassette_bogus_key(
        "error_identity_edge/auth_rejection_carries_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let error = model
                .completion_request("Never authenticated")
                .max_tokens(16)
                .send()
                .await
                .expect_err("a bogus key must be rejected");
            assert!(
                matches!(error, CompletionError::ProviderResponse(_)),
                "contract classification holds on the auth tier: {error:?}"
            );
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(401)
            );
            // Census finding, derived from the recording: Anthropic's auth
            // tier answers 401 *without* a `request-id` header (the fixture's
            // response headers show the absence) — the same
            // success-only pattern xAI shows on its 4xx. None by design.
            assert_eq!(error.provider_request_id(), None);
        },
    )
    .await;
}

/// 400 validation: an impossible parameter, produced by the request
/// validator rather than the model-lookup path.
#[tokio::test]
async fn validation_error_carries_identity() {
    with_anthropic_cassette(
        "error_identity_edge/validation_error_carries_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let error = model
                .completion_request("Never validated")
                .max_tokens(1)
                .additional_params(serde_json::json!({"temperature": -5.0}))
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
            assert_transport_request_id(error.provider_request_id(), "400 validation error");
            assert!(
                error
                    .provider_response_json()
                    .expect("error body is JSON")
                    .is_some(),
                "the provider's error envelope survives"
            );
        },
    )
    .await;
}

// Oversized-request cell deliberately dropped: Claude Sonnet 4.6's context
// window exceeds what is affordable to overflow (a 440k-token probe was
// accepted and answered normally), and the validation tier it would exercise
// is already covered by the impossible-temperature cell above.

/// Streaming connect 4xx — records the current handshake behavior. See the
/// PR findings: pre-fix this was a bare status with body, headers, and id
/// all dropped; the fix threads the same preserved-details error the unary
/// transport uses, so the streaming 4xx now matches its blocking twin.
#[tokio::test]
async fn streaming_connect_4xx_matches_blocking_richness() {
    with_anthropic_cassette(
        "error_identity_edge/streaming_connect_4xx_matches_blocking_richness",
        |client| async move {
            let model = client.completion_model("claude-nonexistent-model-for-error-edge");
            let result = model
                .completion_request("Never streamed")
                .max_tokens(16)
                .stream()
                .await;
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
            assert!(
                matches!(error, CompletionError::ProviderResponse(_)),
                "the streaming 4xx carries the same shape as blocking: {error:?}"
            );
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(404)
            );
            assert!(
                error
                    .provider_response_body()
                    .is_some_and(|body| body.contains("not_found") || body.contains("model")),
                "the handshake error body survives: {error:?}"
            );
            assert_transport_request_id(error.provider_request_id(), "streaming connect 4xx");
        },
    )
    .await;
}

/// Streaming connect 401: the auth-tier variant of the handshake asymmetry —
/// post-fix, classified with the contract (id absent on Anthropic's auth
/// tier, per the recorded census).
#[tokio::test]
async fn streaming_connect_auth_rejection_classifies_with_contract() {
    with_anthropic_cassette_bogus_key(
        "error_identity_edge/streaming_connect_auth_rejection_classifies_with_contract",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let result = model
                .completion_request("Never streamed")
                .max_tokens(16)
                .stream()
                .await;
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
            assert!(
                matches!(error, CompletionError::ProviderResponse(_)),
                "got {error:?}"
            );
            assert_eq!(
                error
                    .provider_response_status()
                    .map(|status| status.as_u16()),
                Some(401)
            );
            assert_eq!(error.provider_request_id(), None);
        },
    )
    .await;
}

/// A streamed agent run against a rejected key: the run's surfaced
/// `PromptError` exposes the failed call's identity accessors — the agent
/// surface of the error path, streaming side. (Anthropic's auth tier sends
/// no id; the accessors answer `None`, never a secondary failure.)
#[tokio::test]
async fn streamed_agent_run_failure_exposes_error_identity_accessors() {
    with_anthropic_cassette_bogus_key(
        "error_identity_edge/streamed_agent_run_failure_exposes_error_identity_accessors",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a terse assistant.")
                .max_tokens(16)
                .build();

            let mut stream = agent
                .runner(rig::completion::Message::user("Never authenticated"))
                .stream()
                .await;
            let mut surfaced = None;
            while let Some(item) = stream.next().await {
                if let Err(error) = item {
                    surfaced = Some(error);
                    break;
                }
            }
            let error = surfaced.expect("the failed run must surface an error");
            let message = error.to_string();
            assert!(
                message.contains("401") || message.to_lowercase().contains("auth"),
                "the auth failure surfaces: {message}"
            );
        },
    )
    .await;
}
