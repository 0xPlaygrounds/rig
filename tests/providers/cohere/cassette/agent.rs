//! Cassette-backed Cohere non-streaming completion coverage.

use rig::completion::{CompletionModel, GetTokenUsage, Prompt};
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

/// Cohere reports two token counters, and they differ substantially: `billed_units`
/// excludes the system overhead that `tokens` counts. Rig's `Usage` must track
/// `usage.tokens`, so this asserts against the recorded response rather than against
/// literals — it keeps meaning if the cassette is re-recorded.
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

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            assert_eq!(response.raw_response.finish_reason, FinishReason::Complete);

            let raw_usage = response
                .raw_response
                .usage
                .as_ref()
                .expect("Cohere should report usage");
            let tokens = raw_usage
                .tokens
                .as_ref()
                .expect("Cohere should report `usage.tokens`");
            let billed = raw_usage
                .billed_units
                .as_ref()
                .expect("Cohere should report `usage.billed_units`");

            assert_eq!(
                response.usage.input_tokens,
                tokens.input_tokens.expect("input token count") as u64
            );
            assert_eq!(
                response.usage.output_tokens,
                tokens.output_tokens.expect("output token count") as u64
            );
            assert_eq!(
                response.usage.total_tokens,
                response.usage.input_tokens + response.usage.output_tokens
            );

            // Guards the counter choice: if `Usage` were ever read from
            // `billed_units`, the assertions above would still pass on a response
            // where the two counters happen to agree, but not on a real one.
            assert_ne!(
                tokens.input_tokens, billed.input_tokens,
                "expected Cohere's two input counters to differ, so the assertions above are meaningful"
            );

            // Reported beside the two counters and previously discarded.
            let cached = raw_usage
                .cached_tokens
                .expect("Cohere should report `usage.cached_tokens`");
            assert_eq!(response.usage.cached_input_tokens, cached as u64);

            // The telemetry span reads the same mapping, so the two agree.
            assert_eq!(raw_usage.token_usage(), response.usage);
        },
    )
    .await;
}
