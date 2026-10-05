//! Response identity on Groq (rig#2265): Groq reports its transport request
//! id on `x-request-id` — the same header OpenAI and xAI use — verified live
//! and now captured via the dialect's own `GROQ.request_id_header`. An earlier revision of
//! this suite recorded the header arriving while the compat-default contract
//! ignored it; #2265's acceptance criterion ("providers that expose these
//! populate them") makes capture, not documentation, the fix.

use anyhow::Result;

use super::support::with_groq_cassette_result;
use rig::completion::CompletionRequest;

const MODEL: &str = "openai/gpt-oss-120b";

#[tokio::test]
async fn streaming_terminal_carries_identity() -> Result<()> {
    use futures::StreamExt;

    with_groq_cassette_result(
        "response_identity_edge/streaming_terminal_carries_identity",
        |client| async move {
            let model = client.completion(MODEL);
            let mut stream = model.stream(CompletionRequest::new(
                "Reply with exactly: stream identity probe",
            ))?;
            while let Some(item) = stream.next().await {
                item?;
            }
            let terminal = stream
                .finish()
                .await
                .expect("stream should yield a terminal record");
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

/// 401 auth rejection (rig#2314 error matrix): Groq's auth tier carries the
/// id its 4xx errors do (recorded).
#[tokio::test]
async fn auth_rejection_classifies_with_contract() -> Result<()> {
    use super::support::with_groq_cassette_bogus_key_result;

    with_groq_cassette_bogus_key_result(
        "response_identity_edge/auth_rejection_classifies_with_contract",
        |client| async move {
            let model = client.completion(MODEL);
            let error = model
                .call(CompletionRequest::new("Never authenticated"))
                .await
                .expect_err("a bogus key must be rejected");
            anyhow::ensure!(
                matches!(error, rig::error::ProviderError::ProviderResponse(_)),
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
