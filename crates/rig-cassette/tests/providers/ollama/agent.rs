//! Migrated from `examples/agent_with_ollama.rs`.

use rig::providers::ollama::wire::OllamaConfig;
use rig_test_support::cassette_models::OllamaModels;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn completion_smoke() {
    let ollama = OllamaModels::new(OllamaConfig::new(), rig::rig_reqwest::shared());
    let agent = rig::AgentBuilder::new(ollama.completion("qwen3:4b"))
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response.output());
}
