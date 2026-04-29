//! Llamafile context smoke test.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn context_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(client.agent(support::model_name()), |builder, doc| {
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
}
