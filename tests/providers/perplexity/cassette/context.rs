//! Perplexity context/document cassette coverage.

use rig::completion::Prompt;
use rig::prelude::AgentClientExt;
use rig::providers::perplexity;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn context_smoke() {
    with_perplexity_cassette("context/context_smoke", |client| async move {
        let agent = CONTEXT_DOCS
            .iter()
            .copied()
            .fold(client.agent(perplexity::SONAR), |builder, doc| {
                builder.context(doc)
            })
            .preamble(
                "Use the provided context documents as the authoritative source. Answer concisely.",
            )
            .build();

        let response = agent
            .prompt(CONTEXT_PROMPT)
            .await
            .expect("context prompt should succeed");

        assert_contains_any_case_insensitive(
            &response,
            &[
                "ancient tool",
                "farming tool",
                "farm the land",
                "used by the ancestors",
            ],
        );
    })
    .await;
}
