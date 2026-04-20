//! DeepSeek reasoning roundtrip tests.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::deepseek;

use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming() {
    let client = deepseek::Client::from_env();
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(deepseek::DEEPSEEK_REASONER),
        None,
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn nonstreaming() {
    let client = deepseek::Client::from_env();
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(deepseek::DEEPSEEK_REASONER),
        None,
    ))
    .await;
}
