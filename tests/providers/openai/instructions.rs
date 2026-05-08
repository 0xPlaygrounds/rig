//! Live instruction-routing checks for OpenAI Responses and Chat Completions.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_api_preamble_is_honored_as_instructions() {
    let response = openai::Client::from_env()
        .expect("client should build")
        .agent(openai::GPT_4O)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("OpenAI Responses completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_api_preamble_is_honored_as_system_message() {
    let response = openai::Client::from_env()
        .expect("client should build")
        .completions_api()
        .agent(openai::GPT_4O)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("OpenAI Chat Completions completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
