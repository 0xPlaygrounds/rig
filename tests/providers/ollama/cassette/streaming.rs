//! Ollama streaming completion smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use super::super::support::with_ollama_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn streaming_smoke() {
    with_ollama_cassette("streaming/streaming_smoke", |env| async move {
        let agent = env
            .agent(MODEL)
            .preamble(STREAMING_PREAMBLE)
            .additional_params(serde_json::json!({ "think": false }))
            .build();

        let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
