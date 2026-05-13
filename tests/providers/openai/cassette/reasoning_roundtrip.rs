//! OpenAI reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test openai openai::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::CompletionClient;

use super::super::support::with_openai_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_openai_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model("gpt-5.2"),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_openai_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model("gpt-5.2"),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}
