//! Mira tools smoke test.

use rig::prelude::*;
use rig::providers::{anthropic, mira};

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn tools_smoke() {
    let cfg = mira::functions::Config::from_env(anthropic::completion::CLAUDE_SONNET_4_6)
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Mira(cfg))
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
