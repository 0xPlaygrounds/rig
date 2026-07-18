//! Ollama runs of the provider-neutral tool-conformance scenarios.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server. Transport request matching stays in this provider suite;
//! the behavioral assertions are shared with artifact-backed local models.

use rig::client::CompletionClient;
use rig_agent::test_utils::{optional_argument, sequential_tools};
use serde_json::json;

use super::super::support::with_ollama_cassette;

const MODEL: &str = "qwen3:4b";

#[tokio::test]
async fn tool_with_optional_argument() {
    with_ollama_cassette("tools/optional_argument", |client| async move {
        let report = optional_argument(client.completion_model(MODEL), |builder| {
            builder.additional_params(json!({ "think": false }))
        })
        .await
        .expect("optional-argument conformance scenario should succeed");
        eprintln!("[ollama] {report:?}");
    })
    .await;
}

#[tokio::test]
async fn two_tools_nonstreaming_chain() {
    with_ollama_cassette("tools/two_tools_nonstreaming", |client| async move {
        let report = sequential_tools(client.completion_model(MODEL), |builder| {
            builder.additional_params(json!({ "think": false }))
        })
        .await
        .expect("sequential-tool conformance scenario should succeed");
        eprintln!("[ollama] {report:?}");
    })
    .await;
}
