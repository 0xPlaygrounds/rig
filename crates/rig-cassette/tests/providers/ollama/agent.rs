//! Migrated from `examples/agent_with_ollama.rs`.

use rig::providers::ollama::wire::Ollama;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() {
    let ollama = Ollama::new();
    let agent = rig::AgentBuilder::new(rig::model(ollama.completion("qwen3:4b")))
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response.output);
}
