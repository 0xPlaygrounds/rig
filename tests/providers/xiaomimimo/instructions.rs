//! Live instruction-routing checks for Xiaomi MiMo OpenAI-compatible completions.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::xiaomimimo;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn openai_compatible_preamble_is_honored() {
    let response = xiaomimimo::Client::from_env()
        .expect("client should build")
        .agent(xiaomimimo::MIMO_V2_5_PRO)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Xiaomi MiMo OpenAI-compatible completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
