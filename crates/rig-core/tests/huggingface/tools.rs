//! Hugging Face tools smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::huggingface;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tools_smoke() {
    let client = huggingface::Client::from_env().expect("client should build");
    let agent = client
        .agent("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
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
