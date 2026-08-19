//! llama.cpp streaming tools coverage, including the migrated example path.

use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

use super::super::cassette_support::*;

#[tokio::test]
async fn streaming_tools_smoke() {
    with_llamacpp_completions_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            // A tool call needs a second turn to turn the result into an
            // answer; `stream_prompt` defaults to one. The llamafile cassette
            // twin of this cell has always passed `max_turns(4)` — this copy
            // never ran, so it never noticed.
            let mut stream = agent
                .stream_prompt(STREAMING_TOOLS_PROMPT)
                .max_turns(4)
                .await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn example_streaming_with_tools() {
    with_llamacpp_completions_cassette("streaming_tools/example_streaming_with_tools", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(
                "You are a calculator here to help the user perform arithmetic operations. \
                 Use the tools provided to answer the user's question and answer in a full sentence.",
            )
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let mut stream = agent.stream_prompt("Calculate 2 - 5").max_turns(4).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming tools prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
