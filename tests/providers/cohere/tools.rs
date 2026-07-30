//! Cohere tools smoke test.

use rig::prelude::*;
use rig::providers::cohere;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn tools_smoke() {
    let cfg = cohere::functions::Config::from_env(cohere::COMMAND_R).expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Cohere(cfg))
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
