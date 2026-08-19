//! Llamafile context smoke test.

use rig::completion::Prompt;
use rig::prelude::*;

use super::super::cassette_support::*;
use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
async fn context_smoke() {
    with_llamafile_cassette("context/context_smoke", |client| async move {
        let agent = CONTEXT_DOCS
            .iter()
            .copied()
            .fold(client.agent(CASSETTE_CHAT_MODEL), |builder, doc| {
                builder.context(doc)
            })
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
