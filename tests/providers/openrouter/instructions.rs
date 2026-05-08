//! Live instruction-routing checks for OpenRouter native and OpenAI-compatible Responses paths.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::{openai, openrouter};

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

use super::DEFAULT_MODEL;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";

fn openai_compatible_model() -> String {
    std::env::var("OPENROUTER_OPENAI_COMPAT_MODEL")
        .unwrap_or_else(|_| DEFAULT_OPENAI_COMPAT_MODEL.to_string())
}

fn openrouter_api_key() -> String {
    std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set")
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn native_provider_preamble_is_honored() {
    let response = openrouter::Client::from_env()
        .expect("client should build")
        .agent(DEFAULT_MODEL)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("OpenRouter native completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn openai_responses_compat_preamble_is_honored_or_reveals_need_for_compat_shim() {
    let response = openai::Client::builder()
        .api_key(openrouter_api_key())
        .base_url(OPENROUTER_BASE_URL)
        .build()
        .expect("OpenRouter OpenAI-compatible client should build")
        .agent(openai_compatible_model())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("OpenRouter via OpenAI Responses completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
