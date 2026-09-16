//! Hugging Face agent completion smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn completion_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .agent("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
