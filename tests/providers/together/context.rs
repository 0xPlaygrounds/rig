//! Together context smoke test.

use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;
use rig_test_support::endpoint::Endpoint;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn context_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&TOGETHER).expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            provider.agent(together::MIXTRAL_8X7B_INSTRUCT_V0_1),
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
