//! Together tools smoke test.

use rig::prelude::*;
use rig::providers::together;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn tools_smoke() {
    let cfg = together::functions::Config::from_env(together::MIXTRAL_8X7B_INSTRUCT_V0_1)
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Together(cfg))
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
