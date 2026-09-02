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
use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::mistral;

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

/// An error must keep both halves of its context: the id *and* the body.
/// Losing either one is what makes a provider failure unactionable.
fn assert_error_keeps_id_and_body(error: &rig::completion::CompletionError) {
    assert_is_request_id(error.provider_request_id());
    assert!(
        error.provider_response_body().is_some(),
        "the error body must survive alongside the id: {error:?}"
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
                .stream()
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

/// The contract must hold on failures too, not just on the happy path: an
/// error that eats the id is exactly as unrecoverable as one that eats the
/// body. Mistral sends `mistral-correlation-id` on 4xx as well.
#[tokio::test]
async fn blocking_error_carries_the_correlation_id() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/blocking_error_carries_the_correlation_id",
        |client| async move {
            let error = client
                .completion_model("definitely-not-a-model")
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect_err("an unroutable model must fail");
            assert_error_keeps_id_and_body(&error);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The streaming twin of the cell above.
///
/// Derived from the recording rather than assumed: `stream()` returns `Ok` for
/// an unroutable model — Mistral answers the POST with a 400 that the SSE
/// layer surfaces on the first polled item, not at the handshake. The
/// assertion is that the id reaches the caller wherever the failure lands, so
/// a streamed turn's failures are no less diagnosable than its blocking twin's.
#[tokio::test]
async fn streaming_error_carries_the_correlation_id() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/streaming_error_carries_the_correlation_id",
        |client| async move {
            let error = match client
                .completion_model("definitely-not-a-model")
                .completion_request("Reply with exactly: identity probe")
                .stream()
                .await
            {
                Err(error) => error,
                Ok(mut stream) => {
                    let mut failure = None;
                    while let Some(item) = stream.next().await {
                        if let Err(error) = item {
                            failure = Some(error);
                            break;
                        }
                    }
                    failure.expect("an unroutable model must fail the stream")
                }
            };
            assert_error_keeps_id_and_body(&error);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The same on a 401, whose body is a different shape again (`{"detail": …}`
/// rather than Mistral's `{"object":"error", …}` envelope).
#[tokio::test]
async fn blocking_unauthorized_carries_the_correlation_id() -> Result<()> {
    super::support::with_mistral_cassette_bogus_key_result(
        "response_identity_edge/blocking_unauthorized_carries_the_correlation_id",
        |client| async move {
            let error = client
                .completion_model(mistral::MISTRAL_SMALL)
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect_err("a bogus key must fail");
            assert_is_request_id(error.provider_request_id());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
