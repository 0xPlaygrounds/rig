//! Gemini reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test gemini gemini::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::gemini;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming() {
    let client = gemini::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model("gemini-2.5-flash"),
        Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn nonstreaming() {
    let client = gemini::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model("gemini-2.5-flash"),
        Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    ))
    .await;
}
