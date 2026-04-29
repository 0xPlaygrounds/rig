//! Anthropic reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test anthropic anthropic::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming() {
    let client = anthropic::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn nonstreaming() {
    let client = anthropic::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;
}
