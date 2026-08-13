//! Response identity on Groq (rig#2265): Groq reports its transport request
//! id on `x-request-id` — the same header OpenAI and xAI use — verified live
//! and now captured via `GroqExt::REQUEST_ID_HEADER`. An earlier revision of
//! this suite recorded the header arriving while the compat-default contract
//! ignored it; #2265's acceptance criterion ("providers that expose these
//! populate them") makes capture, not documentation, the fix.

use anyhow::Result;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::support::with_groq_cassette_result;

const MODEL: &str = "llama-3.3-70b-versatile";

#[tokio::test]
async fn blocking_response_carries_identity() -> Result<()> {
    with_groq_cassette_result(
        "response_identity_edge/blocking_response_carries_identity",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await?;
            anyhow::ensure!(
                response
                    .provider_request_id
                    .as_deref()
                    .is_some_and(|id| !id.trim().is_empty()),
                "Groq sends x-request-id, so provider_request_id must be populated; got {:?}",
                response.provider_request_id
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn streaming_terminal_carries_identity() -> Result<()> {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    with_groq_cassette_result(
        "response_identity_edge/streaming_terminal_carries_identity",
        |client| async move {
            let model = client.completion_model(MODEL);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .stream()
                .await?;
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) = item? {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("stream should yield a terminal record");
            anyhow::ensure!(
                terminal
                    .provider_request_id
                    .as_deref()
                    .is_some_and(|id| !id.trim().is_empty()),
                "blocking/streaming parity: the SSE connection's x-request-id \
                 reaches the terminal; got {:?}",
                terminal.provider_request_id
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// A provider 4xx carries the failed call's transport request id (rig#2314).
#[tokio::test]
async fn provider_error_response_carries_request_id() -> Result<()> {
    with_groq_cassette_result(
        "response_identity_edge/provider_error_response_carries_request_id",
        |client| async move {
            let model = client.completion_model("groq-nonexistent-model-for-identity-edge");
            let error = model
                .completion_request("Never answered")
                .send()
                .await
                .expect_err("a nonexistent model must fail");
            anyhow::ensure!(
                error
                    .provider_request_id()
                    .is_some_and(|id| !id.trim().is_empty()),
                "the 4xx error carries the x-request-id Groq sent; got {error:?}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// 401 auth rejection (rig#2314 error matrix): Groq's auth tier carries the
/// id its 4xx errors do (recorded).
#[tokio::test]
async fn auth_rejection_classifies_with_contract() -> Result<()> {
    use super::support::with_groq_cassette_bogus_key_result;

    with_groq_cassette_bogus_key_result(
        "response_identity_edge/auth_rejection_classifies_with_contract",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion_request("Never authenticated")
                .send()
                .await
                .expect_err("a bogus key must be rejected");
            anyhow::ensure!(
                matches!(error, rig::completion::CompletionError::ProviderResponse(_)),
                "got {error:?}"
            );
            anyhow::ensure!(
                error
                    .provider_request_id()
                    .is_some_and(|id| !id.trim().is_empty()),
                "Groq's auth tier sends x-request-id (see the fixture); got {error:?}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
