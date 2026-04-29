//! Hugging Face context smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::huggingface;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn context_smoke() {
    let client = huggingface::Client::from_env().expect("client should build");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            client.agent("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
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
