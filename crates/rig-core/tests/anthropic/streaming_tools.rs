//! Anthropic streaming tools smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::anthropic;
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming_tools_smoke() {
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
