//! Response identity metadata (rig#2265): Gemini is the documented `None`
//! provider — its live responses carry no request-id response header
//! (verified against `generativelanguage.googleapis.com`), so
//! `provider_request_id` is `None` by design, never an error. These fixtures
//! are the recorded proof of that absence, on both surfaces.

use futures::StreamExt;
use rig::providers::gemini;

use super::super::support::with_gemini_cassette;
use rig::completion::CompletionRequest;

#[tokio::test]
async fn streaming_request_id_is_none_by_design() {
    with_gemini_cassette(
        "response_identity/streaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let mut stream = model
                .stream(CompletionRequest::new(
                    "Reply with exactly: stream identity probe",
                ))
                .expect("stream should open");

            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }
            let terminal = stream
                .finish()
                .await
                .expect("stream should yield a terminal record");
            assert_eq!(
                terminal.provider_request_id, None,
                "blocking/streaming parity for the None provider"
            );
        },
    )
    .await;
}

/// Streamed parity for the `None` provider through the agent surfaces.
#[tokio::test]
async fn streamed_agent_run_reports_none_identity() {
    use crate::support::IdentityProbe;

    with_gemini_cassette(
        "response_identity/streamed_agent_run_reports_none_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
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
            assert_eq!(turns[0].provider_request_id, None);
        },
    )
    .await;
}

/// 401/403 control cell (rig#2314 error matrix): the contract-less provider's
/// auth failure is the provider's response, id `None`.
#[tokio::test]
async fn auth_rejection_keeps_transport_shape() {
    use super::super::support::with_gemini_cassette_bogus_key;

    with_gemini_cassette_bogus_key(
        "response_identity/auth_rejection_keeps_transport_shape",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let error = model
                .call(CompletionRequest::new("Never authenticated"))
                .await
                .expect_err("a bogus key must be rejected");
            assert!(
                matches!(error, rig::error::ProviderError::ProviderResponse(_)),
                "the reply is the provider's, id contract or not: {error:?}"
            );
            assert_eq!(error.provider_request_id(), None);
        },
    )
    .await;
}
