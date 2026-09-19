//! Cassette-backed OpenRouter compatibility coverage through Rig's OpenAI
//! Responses wire: the `OPENROUTER` dialect routed to `/responses` once.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::responses_api::CompletionResponse;
use serde::Deserialize as _;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::super::support::with_openrouter_openai_cassette;

const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";

#[tokio::test]
async fn openai_responses_raw_response_accepts_service_tier_metadata() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata",
        |client| async move {
            let model = client.completion(DEFAULT_OPENAI_COMPAT_MODEL);
            let request = model
                .completion_request("Reply with exactly: openrouter responses service tier ok")
                .preamble(
                    "Return the requested text exactly, with no extra commentary.".to_string(),
                )
                .build();

            // `service_tier` is Responses-API metadata rig does not normalize,
            // so it is read off the provider's own reply document, which the
            // driver keeps verbatim on `raw`. One interaction either way.
            let response = model
                .completion(request)
                .await
                .expect("OpenRouter Responses API completion should deserialize");

            let document = CompletionResponse::deserialize(&response.raw)
                .expect("raw is the Responses API's own response");
            let service_tier = document
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

            assert_nonempty_response(&response.output);
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
                .prompt("In one sentence, confirm this streaming response works.")
                .stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should not fail on OpenRouter service_tier metadata");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
