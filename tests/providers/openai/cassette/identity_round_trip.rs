//! Identity survives persistence, recorded live (#2336).
//!
//! Making `message_id` / `response_id` / `provider_request_id`
//! `Option<WireId>` moved every one of these values through a new type on
//! both the serialize and deserialize side. OpenAI is covered on **both** of
//! its APIs because they populate different axes: Responses reports a
//! message-scoped `msg_…`, Chat Completions a response-scoped `chatcmpl-…`,
//! and both report `x-request-id`.
//!
//! A cassette cannot observe the newtype — that is a compile-time property.
//! What these recordings catch is the migration having broken extraction or
//! persistence against a real provider payload.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::{with_openai_cassette, with_openai_completions_cassette};

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: OpenAI reports an `x-request-id` header, got {id:?}"
    );
}

fn assert_round_trips(response: &CompletionResponse, context: &str) {
    let json = serde_json::to_value(response).expect("response should serialize");
    let reloaded: CompletionResponse =
        serde_json::from_value(json).expect("response should round-trip");

    assert_eq!(
        reloaded.message_id, response.message_id,
        "{context}: message id must survive persistence"
    );
    assert_eq!(
        reloaded.response_id, response.response_id,
        "{context}: response id must survive persistence"
    );
    assert_eq!(
        reloaded.provider_request_id, response.provider_request_id,
        "{context}: transport request id must survive persistence"
    );
}

#[tokio::test]
async fn responses_identity_survives_a_json_round_trip() {
    with_openai_cassette(
        "identity_round_trip/responses_identity_survives_a_json_round_trip",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_4O)
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert_request_id(
                response.provider_request_id.as_deref(),
                "responses blocking",
            );
            assert_round_trips(&response, "responses blocking");
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_identity_survives_a_json_round_trip() {
    with_openai_completions_cassette(
        "identity_round_trip/chat_completions_identity_survives_a_json_round_trip",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_4O)
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("chatcmpl")),
                "Chat Completions reports a response-scoped id, got {:?}",
                response.response_id
            );
            assert_request_id(
                response.provider_request_id.as_deref(),
                "chat completions blocking",
            );
            assert_round_trips(&response, "chat completions blocking");
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_terminal_identity_survives_a_json_round_trip() {
    with_openai_completions_cassette(
        "identity_round_trip/streaming_terminal_identity_survives_a_json_round_trip",
        |client| async move {
            let mut stream = client
                .completion_model(openai::GPT_4O)
                .completion_request("Reply with exactly: stream identity probe")
                .stream()
                .await
                .expect("stream should open");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(record) =
                    item.expect("stream item should succeed")
                {
                    terminal = Some(record);
                }
            }
            let terminal = terminal.expect("stream should yield a terminal record");

            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "chat completions streaming terminal",
            );

            let json = serde_json::to_value(&terminal).expect("terminal should serialize");
            let reloaded: StreamFinal =
                serde_json::from_value(json).expect("terminal should round-trip");
            assert_eq!(
                reloaded, terminal,
                "a terminal record must reload as the same value it was written from"
            );
        },
    )
    .await;
}

/// Agent-run persistence: the identity a run records must survive being
/// written and reloaded. This is the axis with no prior coverage, and the one
/// a careless serde attribute breaks silently — `rig-agent`'s turn types
/// carry these ids with their own attribute sets, so a change that omitted an
/// absent id instead of writing `null` would alter every stored run.
#[tokio::test]
async fn agent_run_identity_survives_a_json_round_trip() {
    with_openai_cassette(
        "identity_round_trip/agent_run_identity_survives_a_json_round_trip",
        |client| async move {
            let agent = client.agent(openai::GPT_4O).build();

            let response = agent
                .prompt("Reply with exactly: identity probe")
                .extended_details()
                .await
                .expect("run should succeed");

            assert!(
                !response.completion_calls.is_empty(),
                "the run must record at least one completion call"
            );

            for (index, call) in response.completion_calls.iter().enumerate() {
                let json = serde_json::to_value(call).expect("call should serialize");
                let reloaded: rig::agent::CompletionCall =
                    serde_json::from_value(json).expect("call should round-trip");

                assert_eq!(
                    reloaded.message_id, call.message_id,
                    "call {index}: message id must survive persistence"
                );
                assert_eq!(
                    reloaded.response_id, call.response_id,
                    "call {index}: response id must survive persistence"
                );
                assert_eq!(
                    reloaded.provider_request_id, call.provider_request_id,
                    "call {index}: transport id must survive persistence"
                );
            }
        },
    )
    .await;
}
