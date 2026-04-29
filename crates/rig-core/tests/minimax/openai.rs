//! MiniMax OpenAI-compatible completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn openai_compatible_completion_smoke() {
    let response = minimax::Client::from_env()
        .expect("client should build")
        .agent(minimax::MINIMAX_M2_7)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax OpenAI-compatible completion should succeed");

    assert_nonempty_response(&response);
}
