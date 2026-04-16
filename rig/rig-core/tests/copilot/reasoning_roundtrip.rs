//! Copilot reasoning roundtrip tests.

use rig::client::CompletionClient;

use crate::copilot::{LIVE_RESPONSES_MODEL, live_client};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming() {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        live_client().completion_model(LIVE_RESPONSES_MODEL),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn nonstreaming() {
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        live_client().completion_model(LIVE_RESPONSES_MODEL),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
}
