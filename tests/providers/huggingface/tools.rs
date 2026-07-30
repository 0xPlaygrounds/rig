//! Hugging Face tools smoke test.

use rig::prelude::*;
use rig::providers::huggingface;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tools_smoke() {
    let cfg = huggingface::functions::Config::from_env("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::HuggingFace(cfg))
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
