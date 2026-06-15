//! DeepSeek reasoning roundtrip tests.

use rig::client::CompletionClient;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
async fn streaming() {
    with_deepseek_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
            Some(thinking_params()),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_deepseek_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
            Some(thinking_params()),
        ))
        .await;
    })
    .await;
}
