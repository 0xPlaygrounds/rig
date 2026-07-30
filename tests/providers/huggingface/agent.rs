//! Hugging Face agent completion smoke test.

use rig::prelude::*;
use rig::providers::huggingface;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn completion_smoke() {
    let cfg = huggingface::functions::Config::from_env("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::HuggingFace(cfg))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
