//! Hugging Face context smoke test.

use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn context_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE).expect("config should build from env");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            rig::AgentBuilder::new(rig::model(
                provider.completion("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
            )),
            rig::AgentBuilder::context,
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
}
