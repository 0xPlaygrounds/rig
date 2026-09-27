//! Cassette-backed OpenRouter reasoning roundtrip tests.

use crate::reasoning::{self, ReasoningRoundtripAgent};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn streaming() {
    with_openrouter_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion("openai/gpt-5.2"),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" },
                "include_reasoning": true
            })),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_openrouter_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion("openai/gpt-5.2"),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" },
                "include_reasoning": true
            })),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn reasoning_delta_hook_streaming() {
    with_openrouter_cassette("reasoning_delta_hook/streaming", |client| async move {
        reasoning::run_reasoning_delta_hook_streaming(
            client.completion("openai/gpt-5.2"),
            serde_json::json!({
                "reasoning": { "effort": "medium" },
                "include_reasoning": true
            }),
            "openrouter",
        )
        .await;
    })
    .await;
}
