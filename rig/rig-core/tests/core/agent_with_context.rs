//! Migrated from `examples/agent_with_context.rs`.

use rig::agent::AgentBuilder;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::cohere::{self, COMMAND_R};

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn context_documents_prompt() {
    let client = cohere::Client::from_env();
    let model = client.completion_model(COMMAND_R);
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(AgentBuilder::new(model), |builder, doc| {
            builder.context(doc)
        })
        .build();

    let response = agent
        .prompt(CONTEXT_PROMPT)
        .await
        .expect("context prompt should succeed");

    assert_contains_any_case_insensitive(
        &response,
        &["ancient tool", "farm the land", "used by the ancestors"],
    );
}
