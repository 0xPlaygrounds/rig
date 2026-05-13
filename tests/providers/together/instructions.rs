//! Live instruction-routing checks for Together.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::together;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn preamble_is_honored() {
    let response = together::Client::from_env()
        .expect("client should build")
        .agent(together::LLAMA_3_1_8B_INSTRUCT_TURBO)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Together completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
