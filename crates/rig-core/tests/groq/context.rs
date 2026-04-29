//! Groq context smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::groq;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

use super::CONTEXT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn context_smoke() {
    let client = groq::Client::from_env().expect("client should build");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(client.agent(CONTEXT_MODEL), |builder, doc| {
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
