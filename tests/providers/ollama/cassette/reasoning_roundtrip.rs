//! Ollama reasoning roundtrip tests (thinking model, multi-turn).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server. Exercises that reasoning emitted on turn 1 survives into
//! history and is accepted back by Ollama on turn 2 (regression for #1926).

use rig::client::CompletionClient;

use super::super::support::with_ollama_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

const MODEL: &str = "qwen3:4b";

fn think_params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "think": true }))
}

#[tokio::test]
async fn nonstreaming() {
    with_ollama_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(MODEL),
            think_params(),
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn streaming() {
    with_ollama_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model(MODEL),
            think_params(),
        ))
        .await;
    })
    .await;
}
