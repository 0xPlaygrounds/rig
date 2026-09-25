//! Hugging Face tools smoke test.

use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};
use rig::wire::Wire as _;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn tools_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE).expect("config should build from env");
    let agent = rig::AgentBuilder::new(
        provider
            .completion("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
            .on(rig::transport()),
    )
    .preamble(TOOLS_PREAMBLE)
    .tool(Adder)
    .tool(Subtract)
    .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response.output, -3);
}
