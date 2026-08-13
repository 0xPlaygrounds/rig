//! Response identity metadata (rig#2265): Gemini is the documented `None`
//! provider — its live responses carry no request-id response header
//! (verified against `generativelanguage.googleapis.com`), so
//! `provider_request_id` is `None` by design, never an error. These fixtures
//! are the recorded proof of that absence, on both surfaces.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::StreamedAssistantContent;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn nonstreaming_request_id_is_none_by_design() {
    with_gemini_cassette(
        "response_identity/nonstreaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert_eq!(
                response.provider_request_id, None,
                "Gemini reports no request-id header; None is the documented outcome"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_request_id_is_none_by_design() {
    with_gemini_cassette(
        "response_identity/streaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
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
            assert_eq!(
                terminal.provider_request_id, None,
                "blocking/streaming parity for the None provider"
            );
        },
    )
    .await;
}
