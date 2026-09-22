//! Migrated from `examples/agent_with_ollama.rs`.

use rig::prelude::*;
use rig::providers::ollama::wire::Ollama;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() {
    let ollama = Ollama::new().bound().expect("transport should build");
    let agent = ollama
        .agent("qwen3:4b")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response.output);
}
