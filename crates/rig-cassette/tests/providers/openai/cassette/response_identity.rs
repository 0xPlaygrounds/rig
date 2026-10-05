//! Response identity metadata (rig#2265): OpenAI reports `x-request-id` on
//! both the Responses and Chat Completions APIs; blocking and streaming turns
//! carry it identically.

use futures::StreamExt;
use rig::providers::openai;

use super::super::support::with_openai_cassette;
use rig::completion::CompletionRequest;

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: OpenAI reports an `x-request-id` response header, so \
         provider_request_id must be populated"
    );
}

#[tokio::test]
async fn responses_streaming_carries_identity() {
    with_openai_cassette(
        "response_identity/responses_streaming_carries_identity",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
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
            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "responses streaming terminal",
            );
        },
    )
    .await;
}
