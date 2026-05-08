//! Live instruction-routing checks for Perplexity.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::perplexity;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires PERPLEXITY_API_KEY"]
async fn preamble_is_honored() {
    let response = perplexity::Client::from_env()
        .expect("client should build")
        .agent(perplexity::SONAR)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Perplexity completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
