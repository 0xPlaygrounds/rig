//! DeepSeek reasoning roundtrip tests.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::deepseek;

use crate::reasoning::{self, ReasoningRoundtripAgent};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming() {
    let client = deepseek::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
        Some(thinking_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn nonstreaming() {
    let client = deepseek::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
        Some(thinking_params()),
    ))
    .await;
}
