//! Live instruction-routing checks for Z.AI OpenAI-compatible completions.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::zai;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};
use crate::zai::{coding_client, general_client};

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn general_openai_compatible_preamble_is_honored() {
    let response = general_client()
        .agent(zai::GLM_4_6)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Z.AI general completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn coding_openai_compatible_preamble_is_honored() {
    let response = coding_client()
        .agent(zai::GLM_4_6)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Z.AI coding completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
