//! DeepSeek tools smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn tools_smoke() {
    let client = deepseek::Client::from_env().expect("client should build");
    let agent = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
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
