//! Copilot reasoning roundtrip tests.

use rig::client::CompletionClient;
use rig::providers::copilot::CopilotStreamingResponse;

use crate::copilot::{live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_copilot_cassette("reasoning_roundtrip/streaming", |client| async move {
        let expected = serde_json::json!({
            "context": "current_turn",
            "effort": "medium",
            "summary": null
        });
        reasoning::run_reasoning_roundtrip_streaming_with_final(
            ReasoningRoundtripAgent::new(
                client.completion_model(live_responses_model()),
                Some(serde_json::json!({
                    "reasoning": { "effort": "medium" }
                })),
            ),
            |response| {
                let CopilotStreamingResponse::Responses(response) = response else {
                    panic!("Copilot reasoning stream should use the Responses route");
                };
                assert_eq!(response.reasoning_context.as_deref(), Some("current_turn"));
                assert_eq!(response.reasoning_metadata.as_ref(), expected.as_object());
            },
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(live_responses_model()),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}
