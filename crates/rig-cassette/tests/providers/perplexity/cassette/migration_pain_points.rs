//! Perplexity cassette coverage for regressions found during the #2040 provider migration.
use rig::providers::perplexity;
use serde_json::json;

use crate::support::{SmokeStructuredOutput, assert_nonempty_response, assistant_text_response};

use super::super::support::with_perplexity_cassette;
use rig::completion::CompletionRequest;

#[tokio::test]
async fn output_schema_is_dropped_instead_of_sent_as_response_format() {
    with_perplexity_cassette(
        "migration_pain_points/output_schema_is_dropped_instead_of_sent_as_response_format",
        |client| async move {
            let model = client.completion(perplexity::SONAR);
            let response = model
                .call(
                    CompletionRequest::new(
                        "Name one Rust programming language benefit in a short sentence.",
                    )
                    .preamble("Answer briefly.")
                    .output_schema(schemars::schema_for!(SmokeStructuredOutput))
                    .max_tokens(48)
                    .additional_params(json!({"search_context_size": "low"})),
                )
                .await
                .expect("Perplexity should ignore unsupported response_format mapping");

            let text = assistant_text_response(&response.choice)
                .expect("response should contain assistant text");
            assert_nonempty_response(&text);
        },
    )
    .await;
}
