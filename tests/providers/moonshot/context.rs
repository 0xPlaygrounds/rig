//! Moonshot context smoke test.

use rig::prelude::*;
use rig::providers::moonshot;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn context_smoke() {
    let client = OpenAI::from_env_with(&openai_wire::MOONSHOT)
        .expect("MOONSHOT_API_KEY should be set")
        .bound()
        .expect("moonshot client should build");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(client.agent(moonshot::KIMI_K3), |builder, doc| {
            builder.context(doc)
        })
        .build();

    let response = agent
        .prompt(CONTEXT_PROMPT)
        .await
        .expect("context prompt should succeed")
        .output;

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
