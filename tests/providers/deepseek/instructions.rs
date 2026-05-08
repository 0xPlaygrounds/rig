//! Live instruction-routing checks for DeepSeek.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::deepseek;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn preamble_is_honored() {
    let response = deepseek::Client::from_env()
        .expect("client should build")
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("DeepSeek completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
