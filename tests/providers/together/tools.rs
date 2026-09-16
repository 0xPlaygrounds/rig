//! Together tools smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn tools_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .agent(together::MIXTRAL_8X7B_INSTRUCT_V0_1)
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
