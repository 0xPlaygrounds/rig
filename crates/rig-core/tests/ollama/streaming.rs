//! Migrated from `examples/ollama_streaming.rs`.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::ollama;
use rig_core::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn example_streaming_prompt() {
    let agent = ollama::Client::from_env()
        .expect("client should build")
        .agent("llama3.2")
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
