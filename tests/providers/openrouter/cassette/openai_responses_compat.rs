//! Cassette-backed OpenRouter compatibility coverage through Rig's OpenAI Responses provider.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::super::support::with_openrouter_openai_cassette;

const DEFAULT_OPENAI_COMPAT_MODEL: &str = "google/gemini-3-flash-preview";

#[tokio::test]
async fn openai_responses_raw_response_accepts_service_tier_metadata() {
    const SCENARIO: &str =
        "openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata";
    with_openrouter_openai_cassette("openai_responses_compat/openai_responses_raw_response_accepts_service_tier_metadata", |client| async move {
        let model = client
            .completion_model(DEFAULT_OPENAI_COMPAT_MODEL)
            .with_system_instructions_as_messages();
        let request = CompletionRequest::with_history(
            Some("Return the requested text exactly, with no extra commentary."),
            Vec::new(),
            "Reply with exactly: openrouter responses service tier ok",
        );
        let response = model
            .completion(request)
            .await
            .expect("OpenRouter Responses API completion should deserialize");

        assert!(
            response.choice.iter().next().is_some(),
            "response should contain assistant content"
        );

        // `service_tier` is provider wire metadata; the normalized response no
        // longer carries the provider-typed raw response, so parse the
        // recorded body (replay only: the cassette file is written after the
        // test body in record mode).
        if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
            let bodies = crate::cassettes::recorded_response_bodies("openrouter", SCENARIO);
            assert_eq!(bodies.len(), 1, "scenario should record a single interaction");
            let raw: rig::providers::openai::responses_api::CompletionResponse =
                serde_json::from_str(&bodies[0])
                    .expect("recorded body should deserialize as a Responses API response");
            let service_tier = raw
                .additional_parameters
                .service_tier
                .as_ref()
                .expect("OpenRouter response should include service_tier");

            assert!(
                !format!("{service_tier:?}").is_empty(),
                "expected OpenRouter model {DEFAULT_OPENAI_COMPAT_MODEL} to return service_tier metadata"
            );
        }
    })
    .await;
}

#[tokio::test]
async fn openai_responses_agent_prompt_against_openrouter_completes() {
    with_openrouter_openai_cassette(
        "openai_responses_compat/openai_responses_agent_prompt_against_openrouter_completes",
        |client| async move {
            let agent = client
                .with_system_instructions_as_messages()
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
                .with_system_instructions_as_messages()
                .agent(DEFAULT_OPENAI_COMPAT_MODEL)
                .preamble("You are concise. Answer directly.")
                .build();

            let mut stream = Box::pin(
                agent
                    .runner("In one sentence, confirm this streaming response works.")
                    .stream_run(),
            );
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should not fail on OpenRouter service_tier metadata");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
