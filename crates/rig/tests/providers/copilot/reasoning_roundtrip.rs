//! Copilot reasoning roundtrip tests.

use rig::client::CompletionClient;

use crate::copilot::{live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_copilot_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model(live_responses_model()),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
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
