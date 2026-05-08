//! Live instruction-routing checks for the ChatGPT subscription provider.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn preamble_and_system_normalization_are_honored() {
    let response = live_client()
        .agent(LIVE_MODEL)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("ChatGPT completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
