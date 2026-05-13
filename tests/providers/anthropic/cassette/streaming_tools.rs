//! Anthropic streaming tools smoke test.

use rig::client::CompletionClient;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_tools_smoke() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("streaming_tools/streaming_tools_smoke").await;
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

    cassette.finish().await;
}
