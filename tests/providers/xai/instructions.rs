//! Live instruction-routing checks for xAI Responses.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::xai;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn responses_preamble_system_message_is_honored() {
    let response = xai::Client::from_env()
        .expect("client should build")
        .agent(xai::completion::GROK_3_MINI)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("xAI Responses completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
