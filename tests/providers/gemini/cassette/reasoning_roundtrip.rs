//! Gemini reasoning roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    let (cassette, client) =
        super::super::support::gemini_cassette("reasoning_roundtrip/streaming").await;
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model("gemini-2.5-flash"),
        Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    ))
    .await;

    cassette.finish().await;
}

#[tokio::test]
async fn nonstreaming() {
    let (cassette, client) =
        super::super::support::gemini_cassette("reasoning_roundtrip/nonstreaming").await;
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model("gemini-2.5-flash"),
        Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
            }
        })),
    ))
    .await;

    cassette.finish().await;
}
