//! Cassette-backed Cohere non-streaming completion coverage.

use rig::completion::{CompletionModel, Prompt};
use rig::prelude::*;
use rig::providers::cohere::completion::FinishReason;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

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
