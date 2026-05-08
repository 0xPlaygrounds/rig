//! Live instruction-routing checks for MiniMax OpenAI-compatible completions.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::minimax;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn openai_compatible_preamble_is_honored() {
    let response = minimax::Client::from_env()
        .expect("client should build")
        .agent(minimax::MINIMAX_M2_7)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("MiniMax OpenAI-compatible completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
