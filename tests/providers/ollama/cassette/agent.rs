//! Ollama agent completion smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::support::with_ollama_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn completion_smoke() {
    with_ollama_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(BASIC_PREAMBLE)
            .additional_params(serde_json::json!({ "think": false }))
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
