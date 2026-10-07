//! Ollama structured output smoke test (JSON schema via `response_format`).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

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
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .output_schema::<SmokeStructuredOutput>()
                .options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Reasoning::Off),
                )
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput = serde_json::from_str(&response.output())
                .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
