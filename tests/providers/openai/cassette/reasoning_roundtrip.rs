//! OpenAI reasoning roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::prelude::*;
use rig::providers::openai;

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

#[tokio::test]
async fn reasoning_delta_hook_streaming() {
    with_openai_cassette("reasoning_delta_hook/streaming", |client| async move {
        reasoning::run_reasoning_delta_hook_streaming(
            client.completion_model(openai::GPT_5_6),
            serde_json::json!({
                "reasoning": { "effort": "high", "summary": "detailed" }
            }),
            "openai",
        )
        .await;
    })
    .await;
}
