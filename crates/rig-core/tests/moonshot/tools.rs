//! Moonshot required-tool-choice smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::message::ToolChoice;
use rig_core::providers::moonshot;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn required_tool_choice_agent_roundtrip() {
    let agent = moonshot::Client::from_env()
        .expect("moonshot client should build")
        .agent(moonshot::KIMI_K2_5)
        .preamble(TOOLS_PREAMBLE)
        .tool_choice(ToolChoice::Required)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .max_turns(3)
        .await
        .expect("required-tool-choice prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
