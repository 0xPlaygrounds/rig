//! Ollama streaming tools smoke test (calculator over streaming, multi-turn).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use super::super::support::with_ollama_cassette;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn streaming_tools_smoke() {
    with_ollama_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .additional_params(serde_json::json!({ "think": false }))
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
