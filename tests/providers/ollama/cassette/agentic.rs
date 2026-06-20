//! Ollama agentic structured-output coverage.
//!
//! These mirror the way real consumers drive Ollama (e.g. repo-tagger):
//! `output_schema_raw` (raw JSON Schema) combined with thinking, tools, and the
//! multi-turn non-streaming agent loop — a combination the other cassettes only
//! exercise separately. rig maps `output_schema` to Ollama's `format` field and
//! sends `tools` on the same request, so this checks that the model's JSON
//! answer is produced and parsed correctly when `format` + `tools` + `think` are
//! all set at once (the path where the #1926 reasoning drop bit).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::Prompt;
use serde_json::json;

use super::super::support::with_ollama_cassette;
use crate::reasoning::WeatherTool;

const MODEL: &str = "qwen3:4b";

fn raw_schema(value: serde_json::Value) -> schemars::Schema {
    serde_json::from_value(value).expect("raw JSON schema should parse")
}

/// `output_schema_raw` + `think: true` (no tools): structured output must still
/// be produced and parsed when the model also emits a `thinking` trace. The
/// existing `structured_output` cassette uses `think: false`, so this covers the
/// thinking interaction repo-tagger relies on.
#[tokio::test]
async fn structured_output_raw_with_thinking() {
    with_ollama_cassette("structured_output/raw_with_thinking", |client| async move {
        let schema = raw_schema(json!({
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "summary": { "type": "string" }
            },
            "required": ["title", "summary"]
        }));

        let agent = client
            .agent(MODEL)
            .output_schema_raw(schema)
            .additional_params(json!({ "think": true }))
            .build();

        let response = agent
            .prompt(
                "Summarize the Rust programming language: give a short title and a \
                 one-sentence summary.",
            )
            .await
            .expect("structured output with thinking should succeed");

        let parsed: serde_json::Value =
            serde_json::from_str(&response).expect("response should be schema JSON");
        for key in ["title", "summary"] {
            assert!(
                parsed
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty()),
                "[ollama] structured output missing non-empty `{key}`: {parsed}"
            );
        }
    })
    .await;
}

/// The repo-tagger builder config: `output_schema_raw` + a tool + `think: true`
/// + the multi-turn non-streaming agent loop. rig must construct one request
/// carrying `format` + `tools` + `think` together and parse the structured
/// answer.
///
/// Note: when `format` (output schema) is set, Ollama biases the model toward
/// emitting the schema JSON immediately rather than calling tools, so the tool
/// may not be invoked (observed with qwen3:4b). That is Ollama/model behavior,
/// not a rig concern; this test asserts the rig-owned invariant — the combined
/// request succeeds and the structured answer parses — and reports the tool
/// invocation count for diagnostics.
#[tokio::test]
async fn structured_output_with_tools_and_thinking() {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette(
        "agentic/structured_output_with_tools",
        |client| async move {
            let schema = raw_schema(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "summary": { "type": "string" }
                },
                "required": ["city", "summary"]
            }));

            let agent = client
                .agent(MODEL)
                .preamble(
                    "You are a weather assistant. Use the get_weather tool to look up weather, \
                     then answer.",
                )
                .tool(WeatherTool::new(call_count.clone()))
                .output_schema_raw(schema)
                .additional_params(json!({ "think": true }))
                .default_max_turns(5)
                .build();

            let response = agent
                .prompt(
                    "What is the current weather in Tokyo? Use the get_weather tool, then return \
                     the city and a one-sentence summary of the conditions.",
                )
                .await
                .expect("agentic structured output should succeed");

            let parsed: serde_json::Value =
                serde_json::from_str(&response).expect("response should be schema JSON");
            for key in ["city", "summary"] {
                assert!(
                    parsed
                        .get(key)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|s| !s.trim().is_empty()),
                    "[ollama] agentic structured output missing non-empty `{key}`: {parsed}"
                );
            }

            eprintln!(
                "[ollama] get_weather invoked {} time(s)",
                call_count.load(Ordering::SeqCst)
            );
        },
    )
    .await;
}
