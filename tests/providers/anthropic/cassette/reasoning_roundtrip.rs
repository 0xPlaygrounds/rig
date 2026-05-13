//! Anthropic reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test anthropic anthropic::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::CompletionClient;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("reasoning_roundtrip/streaming").await;
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;

    cassette.finish().await;
}

#[tokio::test]
async fn nonstreaming() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("reasoning_roundtrip/nonstreaming").await;
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(CLAUDE_SONNET_4_6),
        Some(serde_json::json!({
            "thinking": { "type": "adaptive" }
        })),
    ))
    .await;

    cassette.finish().await;
}
