//! Response identity metadata (rig#2265): xAI reports `x-request-id` on its
//! Responses-shaped API; blocking and streaming turns carry it identically.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::xai;
use rig::streaming::StreamedAssistantContent;

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
            let model = client.completion_model(xai::completion::GROK_3_MINI);
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
            let model = client.completion_model(xai::completion::GROK_3_MINI);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .stream()
                .await
                .expect("stream should open");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) =
                    item.expect("stream item should succeed")
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
