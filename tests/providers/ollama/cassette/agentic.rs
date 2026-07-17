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

use rig::agent::OutputMode;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::streaming::StreamingPrompt;
use serde_json::json;

use super::super::support::with_ollama_cassette;
use crate::reasoning::WeatherTool;
use crate::support::collect_stream_final_response;

const MODEL: &str = "qwen3:4b";

/// Parse the first JSON object out of free-form model text (Prompted mode
/// returns raw text; a real caller strips prose/code fences and reads the first
/// object like this).
fn first_json_object(s: &str) -> serde_json::Value {
    let start = s
        .find('{')
        .expect("prompted response should contain a JSON object");
    serde_json::Deserializer::from_str(&s[start..])
        .into_iter::<serde_json::Value>()
        .next()
        .expect("a JSON value")
        .expect("prompted response should contain a valid JSON object")
}

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
/// + the multi-turn non-streaming agent loop. This is the #1928 regression: by
/// default rig now uses Tool output mode, so the schema is offered as the
/// synthetic `final_result` tool (no native `format` constraint), letting the
/// model call `get_weather` *and* return structured output. Pre-fix (native
/// `format` on every turn), the model skipped the tool and answered immediately.
///
/// Asserts both: the real tool is invoked, and the final answer parses against
/// the schema.
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

            // #1928: under Tool output mode the model is free to use its tools,
            // so the real tool is actually invoked (not suppressed by `format`).
            assert!(
                call_count.load(Ordering::SeqCst) >= 1,
                "[ollama] get_weather should be invoked under Tool output mode (#1928)"
            );
        },
    )
    .await;
}

/// Streaming counterpart of `structured_output_with_tools_and_thinking`. Under
/// the default Tool output mode the streamed run finalizes via the synthetic
/// output-tool call, which carries no assistant text. #1928's streaming fix
/// surfaces that call's arguments as the final response string, so
/// `collect_stream_final_response` returns parseable JSON while the real
/// `get_weather` tool is still invoked.
#[tokio::test]
async fn streaming_structured_output_with_tools() {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette(
        "agentic/streaming_structured_output_with_tools",
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
                .additional_params(json!({ "think": false }))
                .default_max_turns(5)
                .build();

            let mut stream = agent
                .stream_prompt(
                    "What is the current weather in Tokyo? Use the get_weather tool, then return \
                     the city and a one-sentence summary of the conditions.",
                )
                .multi_turn(5)
                .await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming agentic structured output should succeed");

            // The final streamed response must be the structured output (the
            // output-tool arguments), not an empty string (#1928 streaming fix).
            let parsed = first_json_object(&response);
            for key in ["city", "summary"] {
                assert!(
                    parsed
                        .get(key)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|s| !s.trim().is_empty()),
                    "[ollama] streaming agentic structured output missing non-empty `{key}`: {parsed}"
                );
            }

            assert!(
                call_count.load(Ordering::SeqCst) >= 1,
                "[ollama] get_weather should be invoked under streaming Tool output mode (#1928)"
            );
        },
    )
    .await;
}

/// Explicit `OutputMode::Native` is the opt-out / escape hatch: the schema is
/// sent as the provider's native `format` constraint (the pre-#1928 behavior),
/// which still produces valid structured output. Callers who know their model
/// handles tools + native structured output together can select it.
#[tokio::test]
async fn native_mode_emits_structured_output() {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette("agentic/native_mode", |client| async move {
        let schema = raw_schema(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "summary": { "type": "string" } },
            "required": ["city", "summary"]
        }));

        let agent = client
            .agent(MODEL)
            .preamble("You are a weather assistant.")
            .tool(WeatherTool::new(call_count.clone()))
            .output_schema_raw(schema)
            .output_mode(OutputMode::Native)
            .additional_params(json!({ "think": false }))
            .default_max_turns(3)
            .build();

        let response = agent
            .prompt("Give me the weather in Tokyo as JSON with a city and a one-sentence summary.")
            .await
            .expect("native structured output should succeed");

        let parsed: serde_json::Value =
            serde_json::from_str(&response).expect("response should be schema JSON");
        for key in ["city", "summary"] {
            assert!(
                parsed
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty()),
                "[ollama] native structured output missing non-empty `{key}`: {parsed}"
            );
        }
    })
    .await;
}

/// `OutputMode::Prompted` injects the schema into the system prompt and parses
/// the model's final text — no native `format`, no output tool. Useful for
/// models lacking reliable tool calling or native structured output.
#[tokio::test]
async fn prompted_mode_returns_parseable_json() {
    with_ollama_cassette("agentic/prompted_mode", |client| async move {
        let schema = raw_schema(json!({
            "type": "object",
            "properties": { "title": { "type": "string" }, "summary": { "type": "string" } },
            "required": ["title", "summary"]
        }));

        let agent = client
            .agent(MODEL)
            .output_schema_raw(schema)
            .output_mode(OutputMode::Prompted)
            .additional_params(json!({ "think": false }))
            .build();

        let response = agent
            .prompt("Summarize the Rust programming language with a short title and a one-sentence summary.")
            .await
            .expect("prompted structured output should succeed");

        let parsed: serde_json::Value = first_json_object(&response);
        for key in ["title", "summary"] {
            assert!(
                parsed
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty()),
                "[ollama] prompted structured output missing non-empty `{key}`: {parsed}"
            );
        }
    })
    .await;
}
