//! Moonshot context smoke test.

use rig::prelude::*;
use rig::providers::moonshot;

use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn context_smoke() {
    let cfg = moonshot::functions::Config::from_env(moonshot::MOONSHOT_CHAT)
        .expect("moonshot config should build");
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(
            AgentBuilder::new(ProviderConfig::Moonshot(cfg)),
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
