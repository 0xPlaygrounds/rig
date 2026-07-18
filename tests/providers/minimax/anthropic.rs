//! MiniMax Anthropic-compatible completion smoke test.
use rig::prelude::AgentClientExt;

use rig::client::ProviderClient;
use rig::completion::Prompt;
use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = minimax::AnthropicClient::from_env()
        .expect("client should build")
        .agent(minimax::MINIMAX_M2)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
