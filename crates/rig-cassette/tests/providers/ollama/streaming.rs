//! Migrated from `examples/ollama_streaming.rs`.

use rig::providers::ollama::wire::Ollama;
use rig::wire::Wire as _;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn example_streaming_prompt() {
    let agent = rig::AgentBuilder::new(
        Ollama::from_env()
            .expect("config should build from env")
            .completion("llama3.2")
            .on(rig::transport()),
    )
    .preamble("Be precise and concise.")
    .temperature(0.5)
    .build();

    let mut stream = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
