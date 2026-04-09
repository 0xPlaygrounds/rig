//! Migrated from `examples/ollama_streaming_with_tools.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::ollama;
use rig::streaming::StreamingPrompt;

use crate::support::{
    Adder, Subtract, assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn example_streaming_with_tools() {
    let agent = ollama::Client::from_env()
        .agent("llama3.2")
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
