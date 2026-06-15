//! Cassette-backed OpenRouter compatibility coverage through Rig's OpenAI Responses provider.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::super::support::with_openrouter_openai_cassette;

const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";

#[tokio::test]
async fn openai_responses_raw_response_accepts_service_tier_metadata() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata",
        |client| async move {
            let response = client
                .completion_model(DEFAULT_OPENAI_COMPAT_MODEL)
                .completion_request("Reply with exactly: openrouter responses service tier ok")
                .preamble(
                    "Return the requested text exactly, with no extra commentary.".to_string(),
                )
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
                !format!("{service_tier:?}").is_empty(),
                "expected OpenRouter model {DEFAULT_OPENAI_COMPAT_MODEL} to return service_tier metadata"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn openai_responses_agent_prompt_against_openrouter_completes() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_agent_prompt_against_openrouter_completes",
        |client| async move {
            let agent = client
                .agent(DEFAULT_OPENAI_COMPAT_MODEL)
                .preamble("You are concise. Answer with one short sentence.")
                .build();

            let response = agent
                .prompt("Say that OpenRouter via the OpenAI Responses provider works.")
                .await
                .expect("agent.prompt should not fail on OpenRouter service_tier metadata");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn openai_responses_stream_against_openrouter_completes() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_stream_against_openrouter_completes",
        |client| async move {
            let agent = client
                .agent(DEFAULT_OPENAI_COMPAT_MODEL)
                .preamble("You are concise. Answer directly.")
                .build();

            let mut stream = agent
                .stream_prompt("In one sentence, confirm this streaming response works.")
                .await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should not fail on OpenRouter service_tier metadata");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
