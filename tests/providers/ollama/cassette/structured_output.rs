//! Ollama structured output smoke test (JSON schema via the `format` field).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::support::with_ollama_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn structured_output_smoke() {
    with_ollama_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent(MODEL)
                .output_schema::<SmokeStructuredOutput>()
                .additional_params(serde_json::json!({ "think": false }))
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                serde_json::from_str(&response).expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
