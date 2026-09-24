//! Mira tools smoke test.

use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::openai::wire::{MIRA, OpenAI};

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn tools_smoke() {
    let provider = OpenAI::from_env_with(&MIRA)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .endpoint(|provider_config| {
            provider_config.completion(anthropic::completion::CLAUDE_SONNET_4_6)
        })
        .into_agent_builder()
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed")
        .output;

    assert_mentions_expected_number(&response, -3);
}
