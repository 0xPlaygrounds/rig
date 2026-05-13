//! Live instruction-routing checks for Mira.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::{mira, openai};

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn preamble_is_honored() {
    let response = mira::Client::from_env()
        .expect("client should build")
        .agent(openai::GPT_4O)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Mira completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
