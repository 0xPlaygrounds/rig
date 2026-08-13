//! Response identity metadata (rig#2265): OpenAI reports `x-request-id` on
//! both the Responses and Chat Completions APIs; blocking and streaming turns
//! carry it identically.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;

use super::super::support::{with_openai_cassette, with_openai_completions_cassette};

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: OpenAI reports an `x-request-id` response header, so \
         provider_request_id must be populated"
    );
}

#[tokio::test]
async fn responses_nonstreaming_carries_identity() {
    with_openai_cassette(
        "response_identity/responses_nonstreaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("resp")),
                "Responses API reports resp_ ids, got {:?}",
                response.response_id
            );
            assert_request_id(
                response.provider_request_id.as_deref(),
                "responses blocking",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn responses_streaming_carries_identity() {
    with_openai_cassette(
        "response_identity/responses_streaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
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
                "responses streaming terminal",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_nonstreaming_carries_identity() {
    with_openai_completions_cassette(
        "response_identity/chat_completions_nonstreaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("chatcmpl")),
                "Chat Completions reports chatcmpl- ids, got {:?}",
                response.response_id
            );
            assert_request_id(response.provider_request_id.as_deref(), "chat blocking");
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_streaming_carries_identity() {
    with_openai_completions_cassette(
        "response_identity/chat_completions_streaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
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
                "chat streaming terminal",
            );
        },
    )
    .await;
}
