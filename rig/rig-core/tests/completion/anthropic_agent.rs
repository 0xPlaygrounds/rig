//! Anthropic agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::anthropic;

use crate::support::{PREAMBLE, PROMPT, assert_nontrivial_response};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn completion_smoke() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_3_5_SONNET)
        .preamble(PREAMBLE)
        .build();

    let response = agent
        .prompt(PROMPT)
        .await
        .expect("completion should succeed");

    assert_nontrivial_response(&response);
}
