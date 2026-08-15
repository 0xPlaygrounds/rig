//! Identity survives persistence, recorded live (#2336).
//!
//! Gemini is the provider that exercises the *absent* side of the invariant:
//! it reports a response-scoped id but no request-id header, so
//! `provider_request_id` is `None` by design. Retyping these fields moved
//! both the present and the absent case through a new type, and the absent
//! case is the one that would regress silently — an `Option<WireId>` that
//! came back `Some("")` instead of `None` is exactly the sentinel the change
//! exists to make unrepresentable.
//!
//! A cassette cannot observe the newtype itself; what these recordings catch
//! is the migration having broken extraction or persistence against a real
//! Gemini payload.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn blocking_identity_survives_a_json_round_trip() {
    with_gemini_cassette(
        "identity_round_trip/blocking_identity_survives_a_json_round_trip",
        |client| async move {
            let response = client
                .completion_model(gemini::completion::GEMINI_2_5_FLASH)
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert_eq!(
                response.provider_request_id, None,
                "Gemini reports no request-id header; None is the documented outcome"
            );

            let json = serde_json::to_value(&response).expect("response should serialize");
            let reloaded: CompletionResponse =
                serde_json::from_value(json).expect("response should round-trip");

            assert_eq!(
                reloaded.provider_request_id, None,
                "an absent transport id must reload as None, never as an empty id"
            );
            assert_eq!(
                reloaded.response_id, response.response_id,
                "the response id Gemini did report must survive persistence"
            );
            assert_eq!(reloaded.message_id, response.message_id);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_identity_survives_a_json_round_trip() {
    with_gemini_cassette(
        "identity_round_trip/streaming_identity_survives_a_json_round_trip",
        |client| async move {
            let mut stream = client
                .completion_model(gemini::completion::GEMINI_2_5_FLASH)
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

            assert_eq!(
                terminal.provider_request_id, None,
                "Gemini's streamed terminal reports no transport id either"
            );

            let json = serde_json::to_value(&terminal).expect("terminal should serialize");
            let reloaded: StreamFinal =
                serde_json::from_value(json).expect("terminal should round-trip");

            assert_eq!(
                reloaded, terminal,
                "a terminal record must reload as the same value it was written from"
            );
            assert_eq!(
                reloaded.provider_request_id, None,
                "an absent transport id must reload as None, never as an empty id"
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
    with_gemini_cassette(
        "identity_round_trip/agent_run_identity_survives_a_json_round_trip",
        |client| async move {
            let agent = client.agent(gemini::completion::GEMINI_2_5_FLASH).build();

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
