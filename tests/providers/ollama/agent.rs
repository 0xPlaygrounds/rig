//! Migrated from `examples/agent_with_ollama.rs`.

use rig::client::CompletionClient;
use rig::client::Nothing;
use rig::completion::Prompt;
use rig::providers::ollama;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() {
    let client = ollama::Client::new(Nothing).expect("client should build");
    let agent = client
        .agent("qwen3:4b")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
