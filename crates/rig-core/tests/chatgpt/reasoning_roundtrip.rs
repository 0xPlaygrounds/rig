//! ChatGPT reasoning roundtrip tests.

use rig_core::client::CompletionClient;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        live_client().completion_model(LIVE_MODEL),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
}
