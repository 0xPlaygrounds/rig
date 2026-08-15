//! Response identity (rig#2265) for Mistral.
//!
//! Mistral labels its transport request id `mistral-correlation-id` and sends
//! it on every response. Rig left `REQUEST_ID_HEADER` at its conservative
//! `None` default, so `provider_request_id` was always `None`.
//!
//! The cell that used to live here asserted exactly that as the *contract*
//! ("Mistral sends no request-id header"). It could not have caught the gap:
//! the header was not on the cassette harness's response-header allowlist, so
//! the recorded fixture had no id in it to find, and the assertion held
//! against evidence the recording had removed. These cells assert the opposite,
//! against fixtures that now carry the header.

use anyhow::Result;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::StreamingPrompt;

use crate::support::collect_stream_final_response_and_provider_final;

use super::support::with_mistral_cassette_result;

/// The id is a UUID minted per call, so the assertion is on its presence and
/// shape rather than a literal the scrubber placeholders anyway.
fn assert_is_request_id(id: Option<&str>) {
    let id = id.expect("Mistral reports a transport request id on every response");
    assert!(
        !id.is_empty(),
        "the captured request id must not be empty; an empty id is indistinguishable from none"
    );
}

#[tokio::test]
async fn blocking_response_carries_the_correlation_id() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/blocking_response_carries_the_correlation_id",
        |client| async move {
            let model = client.completion_model(mistral::MISTRAL_SMALL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await?;
            assert_is_request_id(response.provider_request_id.as_deref());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn streaming_terminal_carries_the_correlation_id() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/streaming_terminal_carries_the_correlation_id",
        |client| async move {
            let agent = client.agent(mistral::MISTRAL_SMALL).build();
            let mut stream = agent
                .stream_prompt("Reply with exactly: identity probe")
                .await;
            let (_text, provider_final) =
                collect_stream_final_response_and_provider_final(&mut stream).await?;
            assert_is_request_id(provider_final.provider_request_id.as_deref());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// `verify()` resolves [`Provider::VERIFY_PATH`] against the client base URL,
/// which for Mistral is the bare host. The path used to be `/models`, which is
/// a gateway 404 on that host, so verification failed for every key — valid or
/// not. Recorded against a real key, so the cell fails if the path regresses.
#[tokio::test]
async fn verify_succeeds_against_the_versioned_models_route() -> Result<()> {
    use rig::client::VerifyClient;

    with_mistral_cassette_result(
        "response_identity_edge/verify_succeeds_against_the_versioned_models_route",
        |client| async move {
            client
                .verify()
                .await
                .map_err(|error| anyhow::anyhow!("verification should succeed: {error:?}"))?;
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
