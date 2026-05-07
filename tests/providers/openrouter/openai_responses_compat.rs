//! OpenRouter compatibility coverage through Rig's OpenAI Responses provider.
//!
//! These tests exercise the issue #1729 path:
//! `openai::Client::builder().base_url("https://openrouter.ai/api/v1")`.
//!
//! `cargo test -p rig --test openrouter openrouter::openai_responses_compat -- --ignored --nocapture`

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::providers::openai;
use rig::providers::openai::responses_api::OpenAIServiceTier;
use rig::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";
const OPENAI_COMPAT_MODEL_ENV: &str = "OPENROUTER_OPENAI_COMPAT_MODEL";

fn openrouter_api_key() -> String {
    std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set")
}

fn openai_compatible_model() -> String {
    std::env::var(OPENAI_COMPAT_MODEL_ENV)
        .unwrap_or_else(|_| DEFAULT_OPENAI_COMPAT_MODEL.to_string())
}

fn openrouter_openai_client() -> openai::Client {
    openai::Client::builder()
        .api_key(openrouter_api_key())
        .base_url(OPENROUTER_BASE_URL)
        .build()
        .expect("OpenRouter OpenAI-compatible client should build")
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY and an OpenRouter model that returns service_tier=standard"]
async fn openai_responses_raw_response_accepts_standard_service_tier() {
    let model_name = openai_compatible_model();
    let response = openrouter_openai_client()
        .completion_model(&model_name)
        .completion_request("Reply with exactly: openrouter responses service tier ok")
        .preamble("Return the requested text exactly, with no extra commentary.".to_string())
        .send()
        .await
        .expect("OpenRouter Responses API completion should deserialize");

    let service_tier = response
        .raw_response
        .additional_parameters
        .service_tier
        .as_ref()
        .expect("OpenRouter response should include service_tier");

    assert!(
        matches!(service_tier, OpenAIServiceTier::Standard),
        "expected OpenRouter model {model_name} to return service_tier=standard, got {service_tier:?}"
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn openai_responses_agent_prompt_against_openrouter_completes() {
    let agent = openrouter_openai_client()
        .agent(openai_compatible_model())
        .preamble("You are concise. Answer with one short sentence.")
        .build();

    let response = agent
        .prompt("Say that OpenRouter via the OpenAI Responses provider works.")
        .await
        .expect("agent.prompt should not fail on OpenRouter service_tier metadata");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn openai_responses_stream_against_openrouter_completes() {
    let agent = openrouter_openai_client()
        .agent(openai_compatible_model())
        .preamble("You are concise. Answer directly.")
        .build();

    let mut stream = agent
        .stream_prompt("In one sentence, confirm this streaming response works.")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should not fail on OpenRouter service_tier metadata");

    assert_nonempty_response(&response);
}
