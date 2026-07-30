//! Moonshot required-tool-choice smoke test.

use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::moonshot;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn required_tool_choice_agent_roundtrip() {
    let cfg = moonshot::functions::Config::from_env(moonshot::KIMI_K2_5)
        .expect("moonshot config should build");
    let agent = AgentBuilder::new(ProviderConfig::Moonshot(cfg))
        .preamble(TOOLS_PREAMBLE)
        .tool_choice(ToolChoice::Required)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .runner(TOOLS_PROMPT)
        .max_turns(3)
        .run()
        .await
        .expect("required-tool-choice prompt should succeed")
        .output;

    assert_mentions_expected_number(&response, -3);
}
