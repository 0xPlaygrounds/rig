//! Migrated from `examples/ollama_streaming.rs`.

use crate::support::{assert_nonempty_response, collect_stream_final_response};
use rig::prelude::*;
use rig::providers::ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn example_streaming_prompt() {
    let cfg = ollama::functions::Config::from_env("llama3.2").expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Ollama(cfg))
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = Box::pin(
        agent
            .runner("When and where and what type is the next solar eclipse?")
            .stream_run(),
    );
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
