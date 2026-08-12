//! Cassette-backed Cohere non-streaming completion coverage.

use rig::completion::{AssistantContent, CompletionModel, Message, Prompt};
use rig::prelude::*;
use rig::providers::cohere::completion::FinishReason;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_contains_any_case_insensitive, assert_nonempty_response,
};

#[tokio::test]
async fn completion_smoke() {
    with_cohere_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(BASIC_PREAMBLE)
            .temperature(0.2)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn usage_is_reported_from_token_counts() {
    with_cohere_cassette(
        "agent/usage_is_reported_from_token_counts",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request(BASIC_PROMPT)
                .preamble(BASIC_PREAMBLE.to_string())
                .build();

            // `raw_completion` and `completion` share one network request; taking
            // the raw response and normalizing it in-process (rather than issuing
            // a second request) keeps this to the cassette's one recorded
            // interaction.
            let raw_response = model
                .raw_completion(request)
                .await
                .expect("completion should succeed");

            assert_eq!(raw_response.finish_reason, FinishReason::Complete);

            let raw_usage = raw_response
                .usage
                .as_ref()
                .expect("Cohere should report usage");
            let tokens = raw_usage
                .tokens
                .as_ref()
                .expect("Cohere should report `usage.tokens`");
            let raw_input_tokens = tokens.input_tokens;
            let expected_input_tokens = tokens.input_tokens.expect("input token count") as u64;
            let expected_output_tokens = tokens.output_tokens.expect("output token count") as u64;
            let billed_input_tokens = raw_usage
                .billed_units
                .as_ref()
                .expect("Cohere should report `usage.billed_units`")
                .input_tokens;
            let cached = raw_usage
                .cached_tokens
                .expect("Cohere should report `usage.cached_tokens`");
            let expected_usage = rig::completion::Usage::from(raw_usage);

            let response: rig::completion::CompletionResponse = raw_response
                .try_into()
                .expect("normalization should succeed");

            assert_eq!(response.usage.input_tokens, expected_input_tokens);
            assert_eq!(response.usage.output_tokens, expected_output_tokens);
            assert_eq!(
                response.usage.total_tokens,
                response.usage.input_tokens + response.usage.output_tokens
            );

            assert_ne!(
                raw_input_tokens, billed_input_tokens,
                "expected Cohere's two input counters to differ, so the assertions above are meaningful"
            );

            assert_eq!(response.usage.cached_input_tokens, cached as u64);
            assert_eq!(expected_usage, response.usage);
        },
    )
    .await;
}

#[tokio::test]
async fn max_tokens_sets_max_tokens_finish_reason() {
    with_cohere_cassette(
        "agent/max_tokens_sets_max_tokens_finish_reason",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Write a detailed fifty-word description of the ocean.")
                .max_tokens(4)
                .build();

            let response = model
                .raw_completion(request)
                .await
                .expect("capped completion should succeed");

            assert_eq!(response.finish_reason, FinishReason::MaxTokens);
        },
    )
    .await;
}

#[tokio::test]
async fn multiturn_history_is_accepted() {
    with_cohere_cassette("agent/multiturn_history_is_accepted", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let request = model
            .completion_request("What code word did I ask you to remember?")
            .message(Message::user(
                "Remember the code word cobalt-orchid for my next question.",
            ))
            .message(Message::assistant(
                "Understood. I will remember the code word cobalt-orchid.",
            ))
            .max_tokens(32)
            .build();

        let response = model
            .completion(request)
            .await
            .expect("multi-turn history should be accepted");
        let text = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>();

        assert_contains_any_case_insensitive(&text, &["cobalt-orchid", "cobalt orchid"]);
    })
    .await;
}

#[tokio::test]
async fn stop_sequences_are_forwarded() {
    with_cohere_cassette("agent/stop_sequences_are_forwarded", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let request = model
            .completion_request("Output exactly this sequence: alpha<END>omega")
            .temperature(0.0)
            .max_tokens(32)
            .additional_params(serde_json::json!({
                "seed": 7,
                "stop_sequences": ["<END>"]
            }))
            .build();

        let response = model
            .raw_completion(request)
            .await
            .expect("stop sequence request should succeed");

        assert_eq!(response.finish_reason, FinishReason::StopSequence);
    })
    .await;
}

#[tokio::test]
async fn sampling_parameters_are_forwarded() {
    with_cohere_cassette(
        "agent/sampling_parameters_are_forwarded",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Reply with one short sentence about rain.")
                .temperature(0.2)
                .max_tokens(24)
                .additional_params(serde_json::json!({
                    "seed": 11,
                    "p": 0.8,
                    "k": 20,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("documented sampling parameters should be accepted");
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();

            assert_nonempty_response(&text);
        },
    )
    .await;
}
