//! Live instruction-routing checks for Moonshot OpenAI-compatible completions.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::moonshot;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn openai_compatible_preamble_is_honored() {
    let response = moonshot::Client::from_env()
        .expect("client should build")
        .agent(moonshot::KIMI_K2_5)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Moonshot OpenAI-compatible completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
