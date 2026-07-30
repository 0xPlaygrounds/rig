//! ChatGPT streaming tools coverage.

use crate::chatgpt::{LIVE_MODEL, live_agent};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming_tools_smoke() {
    let agent = live_agent(LIVE_MODEL)
        .await
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_TOOLS_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn example_streaming_with_tools() {
    let agent = live_agent(LIVE_MODEL)
        .await
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = Box::pin(agent.runner("Calculate 2 - 5").stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tools prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
