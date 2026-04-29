//! Anthropic agent completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::anthropic;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn completion_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
