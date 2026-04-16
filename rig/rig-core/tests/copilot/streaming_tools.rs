//! Copilot streaming tools coverage, including the migrated example path.

use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use crate::copilot::{LIVE_MODEL, live_client};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming_tools_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
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

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn example_streaming_with_tools() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tools prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
