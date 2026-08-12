//! Cassette-backed Cohere context-document coverage.

use rig::completion::Prompt;
use rig::prelude::*;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
async fn context_documents_are_accepted() {
    with_cohere_cassette("context/context_documents_are_accepted", |client| async move {
        let agent = CONTEXT_DOCS
            .iter()
            .copied()
            .fold(client.agent(CASSETTE_MODEL), |builder, doc| {
                builder.context(doc)
            })
            .preamble("Use the provided context documents as the authoritative source. Answer concisely.")
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
