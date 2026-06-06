//! DeepSeek tools smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
async fn tools_smoke() {
    with_deepseek_cassette("tools/tools_smoke", |client| async move {
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
    })
    .await;
}
