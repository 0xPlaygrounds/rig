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
