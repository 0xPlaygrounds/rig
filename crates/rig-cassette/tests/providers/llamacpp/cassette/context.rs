//! llama.cpp context smoke test.

use super::super::cassette_support::*;
use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};
use rig::wire::Wire as _;

#[tokio::test]
async fn context_smoke() {
    with_llamacpp_cassette("context/context_smoke", |client| async move {
        let agent = CONTEXT_DOCS
            .iter()
            .copied()
            .fold(
                rig::AgentBuilder::new(client.completion(CASSETTE_MODEL).on(rig::transport())),
                |builder, doc| builder.context(doc),
            )
            .build();

        let response = agent
            .prompt(CONTEXT_PROMPT)
            .await
            .expect("context prompt should succeed");

        assert_contains_any_case_insensitive(
            &response.output,
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
