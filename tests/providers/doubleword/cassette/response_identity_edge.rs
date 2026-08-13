//! Contract-vs-reality (rig#2265 / PR #2313 follow-up): Doubleword is an
//! Anthropic-compatible gateway, so it *inherits* the
//! `AnthropicCompatibleProvider::REQUEST_ID_HEADER` default of `request-id`.
//! These recordings document what the gateway actually sends.

use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn blocking_identity_contract_vs_reality() {
    with_doubleword_cassette(
        "response_identity_edge/blocking_identity_contract_vs_reality",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .max_tokens(128)
                .send()
                .await
                .expect("completion should succeed");
            // Assertion derived from the recording (see the fixture's response
            // headers): the gateway does not echo Anthropic's `request-id`
            // header, so the inherited contract captures `None` — harmless by
            // design, documented here.
            assert_eq!(response.provider_request_id, None);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_identity_contract_vs_reality() {
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    with_doubleword_cassette(
        "response_identity_edge/streaming_identity_contract_vs_reality",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .max_tokens(128)
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
            let terminal = terminal.expect("terminal record");
            // Derived from the recording, matching the blocking surface.
            assert_eq!(terminal.provider_request_id, None);
        },
    )
    .await;
}
