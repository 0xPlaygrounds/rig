//! Perplexity non-streaming completion cassette coverage.

use rig::completion::Prompt;
use rig::prelude::AgentClientExt;
use rig::providers::perplexity;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn completion_smoke() {
    with_perplexity_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(perplexity::SONAR)
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
async fn completion_with_perplexity_options() {
    with_perplexity_cassette(
        "agent/completion_with_perplexity_options",
        |client| async move {
            let agent = client
                .agent(perplexity::SONAR)
                .preamble("Answer briefly and include the date or time context if relevant.")
                .additional_params(serde_json::json!({
                    "return_related_questions": true,
                    "search_context_size": "low"
                }))
                .build();

            let response = agent
                .prompt("Name one notable recent development in Rust programming language tooling.")
                .await
                .expect("completion with Perplexity options should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
