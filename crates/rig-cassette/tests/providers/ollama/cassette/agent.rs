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
            .additional_params(serde_json::json!({ "think": false }))
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}

/// Guards the native token-limit mapping on the wire. `max_tokens` has no
/// top-level field in Ollama's `/api/chat`; the equivalent is the
/// `num_predict` model parameter inside `options`. The recorded request
/// carries `"options":{"num_predict":24}`, and its request snapshot pins the
/// body, so a request that dropped `num_predict` or moved it to the top level
/// would fail here. The recorded response has `done_reason: "length"`, the
/// daemon confirming it honored the budget.
#[tokio::test]
async fn completion_respects_max_tokens() {
    with_ollama_cassette("agent/max_tokens", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(MODEL))
            .preamble(BASIC_PREAMBLE)
            // Small enough to truncate the answer well before the model would
            // stop on its own, so the budget is what ends generation.
            .max_tokens(24)
            .additional_params(serde_json::json!({ "think": false }))
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}
