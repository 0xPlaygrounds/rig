//! Identity survives persistence, recorded live (#2336).
//!
//! Making `message_id` / `response_id` / `provider_request_id`
//! `Option<WireId>` moved every one of these values through a new type on
//! both the serialize and deserialize side. These cells are the recorded
//! proof that the values a real Anthropic turn reports still reach the
//! caller, still survive a JSON round-trip, and still come back as the same
//! identity — including the axes Anthropic leaves absent, which must stay
//! `None` rather than becoming an empty string.
//!
//! What a cassette can and cannot prove here: the newtype is a compile-time
//! property, so no recording can observe it. What these recordings *do*
//! catch is the migration having broken extraction or persistence on a real
//! provider payload — which is the part that would silently ship.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionResponse};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::with_anthropic_cassette;

/// Anthropic's contract: a `msg_…` message id and a `request-id` transport
/// header, and no response-scoped id.
fn assert_anthropic_axes(
    message_id: Option<&str>,
    response_id: Option<&str>,
    provider_request_id: Option<&str>,
    context: &str,
) {
    assert!(
        message_id.is_some_and(|id| id.starts_with("msg")),
        "{context}: Anthropic reports message.id, got {message_id:?}"
    );
    assert!(
        provider_request_id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: Anthropic reports a `request-id` header, got {provider_request_id:?}"
    );
    assert_eq!(
        response_id, None,
        "{context}: Anthropic has no response-scoped id; absence must stay None"
    );
}

#[tokio::test]
async fn blocking_identity_survives_a_json_round_trip() {
    with_anthropic_cassette(
        "identity_round_trip/blocking_identity_survives_a_json_round_trip",
        |client| async move {
            let response = client
                .completion_model(CLAUDE_SONNET_4_6)
                .completion_request("Reply with exactly: identity probe")
                .max_tokens(32)
                .send()
                .await
                .expect("completion should succeed");

            assert_anthropic_axes(
                response.message_id.as_deref(),
                response.response_id.as_deref(),
                response.provider_request_id.as_deref(),
                "live blocking response",
            );

            let json = serde_json::to_value(&response).expect("response should serialize");
            let reloaded: CompletionResponse =
                serde_json::from_value(json).expect("response should round-trip");

            assert_anthropic_axes(
                reloaded.message_id.as_deref(),
                reloaded.response_id.as_deref(),
                reloaded.provider_request_id.as_deref(),
                "round-tripped blocking response",
            );
            assert_eq!(reloaded.message_id, response.message_id);
            assert_eq!(reloaded.provider_request_id, response.provider_request_id);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_identity_survives_a_json_round_trip() {
    with_anthropic_cassette(
        "identity_round_trip/streaming_identity_survives_a_json_round_trip",
        |client| async move {
            let mut stream = client
                .completion_model(CLAUDE_SONNET_4_6)
                .completion_request("Reply with exactly: stream identity probe")
                .max_tokens(32)
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

            assert_anthropic_axes(
                terminal.message_id.as_deref(),
                terminal.response_id.as_deref(),
                terminal.provider_request_id.as_deref(),
                "live streaming terminal",
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
    with_anthropic_cassette(
        "identity_round_trip/agent_run_identity_survives_a_json_round_trip",
        |client| async move {
            let agent = client.agent(CLAUDE_SONNET_4_6).max_tokens(32).build();

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
