//! Live instruction-routing checks for Azure OpenAI Chat Completions.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::azure;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

fn completion_deployment() -> String {
    std::env::var("AZURE_COMPLETION_DEPLOYMENT")
        .or_else(|_| std::env::var("AZURE_OPENAI_DEPLOYMENT"))
        .unwrap_or_else(|_| azure::GPT_4O.to_string())
}

#[tokio::test]
#[ignore = "requires AZURE_API_KEY or AZURE_TOKEN, AZURE_API_VERSION, AZURE_ENDPOINT, and a chat deployment"]
async fn chat_completions_preamble_is_honored() {
    let response = azure::Client::from_env()
        .expect("client should build")
        .agent(completion_deployment())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Azure OpenAI completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
