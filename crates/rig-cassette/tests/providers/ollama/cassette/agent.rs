//! Ollama agent completion smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use super::super::support::with_ollama_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn completion_smoke() {
    with_ollama_cassette("agent/completion_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(MODEL))
            .preamble(BASIC_PREAMBLE)
            .options(
                rig::completion::GenerationOptions::default()
                    .reasoning(rig::completion::Reasoning::Off),
            )
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}

/// Guards the token limit on the wire: the recorded request carries
/// `"max_tokens":24`, and the cassette matcher compares request bodies, so a
/// request that dropped it would stop matching. The recorded response
/// finishes on `"length"`, the daemon confirming it honored the budget.
#[tokio::test]
async fn completion_respects_max_tokens() {
    with_ollama_cassette("agent/max_tokens", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(MODEL))
            .preamble(BASIC_PREAMBLE)
            // Small enough to truncate the answer well before the model would
            // stop on its own, so the budget is what ends generation.
            .max_tokens(24)
            .options(
                rig::completion::GenerationOptions::default()
                    .reasoning(rig::completion::Reasoning::Off),
            )
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}
