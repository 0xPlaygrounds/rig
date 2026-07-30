//! Migrated from `examples/agent_with_ollama.rs`.

use rig::prelude::*;
use rig::providers::ollama;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() {
    let agent = AgentBuilder::new(ProviderConfig::Ollama(ollama::functions::Config::new(
        "qwen3:4b",
    )))
    .preamble("You are a comedian here to entertain the user using humour and jokes.")
    .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
