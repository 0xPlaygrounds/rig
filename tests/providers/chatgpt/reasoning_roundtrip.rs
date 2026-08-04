//! ChatGPT reasoning roundtrip tests.

use crate::chatgpt::{LIVE_MODEL, live_provider};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        live_provider(LIVE_MODEL).await,
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
}
