//! Anthropic agent completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
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
