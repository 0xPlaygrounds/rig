//! Together context smoke test.

use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn context_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER).expect("config should build from env");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            rig::AgentBuilder::new(rig::model(
                provider.completion(together::MIXTRAL_8X7B_INSTRUCT_V0_1),
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
