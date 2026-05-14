//! Gemini reasoning roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;

use super::super::support::with_gemini_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_gemini_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model("gemini-2.5-flash"),
            Some(serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
                }
            })),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_gemini_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model("gemini-2.5-flash"),
            Some(serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": { "thinkingBudget": 2048, "includeThoughts": true }
                }
            })),
        ))
        .await;
    })
    .await;
}
