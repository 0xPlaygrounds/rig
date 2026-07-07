//! Llamafile streaming tool round-trip (exercises llama.cpp-style
//! complete-single-chunk tool call streaming through the shared OpenAI path).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use super::super::cassette_support::{CASSETTE_CHAT_MODEL, with_llamafile_cassette};
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_tools_smoke() {
    with_llamafile_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(CASSETTE_CHAT_MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent
                .stream_prompt(STREAMING_TOOLS_PROMPT)
                .multi_turn(3)
                .await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            // STREAMING_TOOLS_PROMPT is "Calculate 2 - 5." => -3.
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}
