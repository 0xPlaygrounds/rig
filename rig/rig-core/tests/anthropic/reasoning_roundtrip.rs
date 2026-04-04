//! Anthropic reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test anthropic reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::anthropic;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming() {
    let client = anthropic::Client::from_env();
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model("claude-sonnet-4-5-20250929"),
        Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 }
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn nonstreaming() {
    let client = anthropic::Client::from_env();
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model("claude-sonnet-4-5-20250929"),
        Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 2048 }
        })),
    ))
    .await;
}
