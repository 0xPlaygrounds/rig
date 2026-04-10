//! OpenRouter reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test openrouter openrouter::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openrouter;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming() {
    let client = openrouter::Client::from_env();
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model("openai/gpt-5.2"),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" },
            "include_reasoning": true
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn nonstreaming() {
    let client = openrouter::Client::from_env();
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model("openai/gpt-5.2"),
        Some(serde_json::json!({
            "reasoning": { "effort": "medium" },
            "include_reasoning": true
        })),
    ))
    .await;
}
