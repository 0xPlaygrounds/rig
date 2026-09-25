//! Moonshot required-tool-choice smoke test.

use rig::message::ToolChoice;
use rig::providers::moonshot;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};
use rig::wire::Wire as _;

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn required_tool_choice_agent_roundtrip() {
    let agent = rig::AgentBuilder::new(
        OpenAI::from_env_with(&openai_wire::MOONSHOT)
            .expect("MOONSHOT_API_KEY should be set")
            .completion(moonshot::KIMI_K3)
            .on(rig::transport()),
    )
    .preamble(TOOLS_PREAMBLE)
    .tool_choice(ToolChoice::Required)
    .tool(Adder)
    .tool(Subtract)
    .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .max_turns(3)
        .await
        .expect("required-tool-choice prompt should succeed")
        .output;

    assert_mentions_expected_number(&response, -3);
}
