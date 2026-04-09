//! Together context smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::together;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn context_smoke() {
    let client = together::Client::from_env();
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            client.agent(together::MIXTRAL_8X7B_INSTRUCT_V0_1),
            |builder, doc| builder.context(doc),
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
}
