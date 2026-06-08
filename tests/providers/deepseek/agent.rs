//! DeepSeek agent completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_deepseek_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(deepseek::DEEPSEEK_V4_FLASH)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
