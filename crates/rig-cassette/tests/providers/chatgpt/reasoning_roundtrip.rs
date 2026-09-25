//! ChatGPT reasoning roundtrip tests.

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::reasoning::{self, ReasoningRoundtripAgent};
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" }
        })),
    ))
    .await;
}
