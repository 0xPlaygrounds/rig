//! Response identity metadata (rig#2265): xAI reports `x-request-id` on its
//! Responses-shaped API; blocking and streaming turns carry it identically.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::responses_api;
use rig::providers::xai;
use rig::streaming::StreamEvent;
use serde::Deserialize;

use super::support::with_xai_cassette;

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: xAI reports an `x-request-id` response header, so \
         provider_request_id must be populated"
    );
}

#[tokio::test]
async fn nonstreaming_response_carries_identity() {
    with_xai_cassette(
        "response_identity/nonstreaming_response_carries_identity",
        |client| async move {
            let model = client.completion(xai::GROK_3_MINI);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "xAI reports a response id, got {:?}",
                response.response_id
            );
            assert_request_id(response.provider_request_id.as_deref(), "blocking");
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_terminal_carries_identity() {
    with_xai_cassette(
        "response_identity/streaming_terminal_carries_identity",
        |client| async move {
            let model = client.completion(xai::GROK_3_MINI);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .stream()
                .await
                .expect("stream should open");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamEvent::Final(final_record) = item.expect("stream item should succeed")
                {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("stream should yield a terminal record");
            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "streaming terminal",
            );
        },
    )
    .await;
}

/// Streamed agent run on xAI: hook turn events carry the SSE connection's
/// transport id.
#[tokio::test]
async fn streamed_agent_run_reports_identity() {
    use crate::support::{IdentityProbe, assert_transport_request_id};

    with_xai_cassette(
        "response_identity/streamed_agent_run_reports_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(xai::GROK_3_MINI)
                .preamble("You are a terse assistant.")
                .add_hook(probe.clone())
                .build();

            let mut stream = agent
                .prompt(rig::completion::Message::user(
                    "Reply with exactly: streamed identity probe",
                ))
                .stream();
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }

            let turns = probe.turn_identities();
            assert_eq!(turns.len(), 1);
            assert_transport_request_id(
                turns[0].provider_request_id.as_deref(),
                "xai streamed turn",
            );
        },
    )
    .await;
}

/// Family D (edge matrix): one interaction, two views — the provider reply
/// captured in [`rig::completion::CompletionResponse::raw`] and the
/// normalized response describe the same interaction, and the transport id
/// rides on the normalized view because the wire body has no slot for a
/// response header.
#[tokio::test]
async fn raw_and_normalized_views_agree_on_identity() {
    with_xai_cassette(
        "response_identity/raw_and_normalized_views_agree_on_identity",
        |client| async move {
            let model = client.completion(xai::GROK_3_MINI);
            let request = model
                .completion_request("Reply with exactly: two views probe")
                .build();
            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");
            assert_request_id(response.provider_request_id.as_deref(), "normalized view");

            let raw_view = responses_api::CompletionResponse::deserialize(&response.raw)
                .expect("`raw` is the serialized Responses CompletionResponse");
            assert_eq!(
                Some(raw_view.id.as_str()),
                response.response_id.as_deref(),
                "raw and normalized views describe the same interaction"
            );
            assert_eq!(
                raw_view.provider_request_id, None,
                "the mirrored wire body carries no response header"
            );
        },
    )
    .await;
}

/// rig#2314 census finding: xAI sends `x-request-id` on *successes* but not
/// on its 4xx error responses (verified live; this fixture's error headers
/// show the absence). The contract classification still applies — the error
/// preserves as `ProviderResponse` with status and body — and the missing
/// header is `None`, never a secondary failure.
#[tokio::test]
async fn provider_error_classifies_with_contract_but_reports_no_id() {
    with_xai_cassette(
        "response_identity/provider_error_classifies_with_contract_but_reports_no_id",
        |client| async move {
            let model = client.completion("grok-nonexistent-model-for-identity-edge");
            let error = model
                .completion_request("Never answered")
                .send()
                .await
                .expect_err("a nonexistent model must fail");
            assert!(
                matches!(error, rig::error::ProviderError::ProviderResponse(_)),
                "contract providers classify 4xx as ProviderResponse: {error:?}"
            );
            assert_eq!(
                error.provider_request_id(),
                None,
                "xAI omits x-request-id on error responses — None by design"
            );
            assert!(error.provider_response_status().is_some());
        },
    )
    .await;
}

/// 401 auth rejection (rig#2314 error matrix): contract classification holds
/// on the auth tier; the recording documents whether xAI's auth tier sends
/// the id it omits on 4xx.
#[tokio::test]
async fn auth_rejection_classifies_with_contract() {
    use super::support::with_xai_cassette_bogus_key;

    with_xai_cassette_bogus_key(
        "response_identity/auth_rejection_classifies_with_contract",
        |client| async move {
            let model = client.completion(xai::GROK_3_MINI);
            let error = model
                .completion_request("Never authenticated")
                .send()
                .await
                .expect_err("a bogus key must be rejected");
            assert!(
                matches!(error, rig::error::ProviderError::ProviderResponse(_)),
                "got {error:?}"
            );
            // Derived from the recording.
            let _ = error.provider_request_id();
        },
    )
    .await;
}
