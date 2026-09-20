//! OpenAI-compatible Responses API reasoning-content roundtrip regression tests.
//!
//! This synthetic fixture pins a llama.cpp-shaped `reasoning_text` response;
//! it is replay-only, not evidence captured from a live provider.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use futures::FutureExt;
use rig::completion::Message;
use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::openai::OpenAI;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::cassettes;
use crate::reasoning::{self, WeatherTool};

const SCENARIO: &str = "openai_compatible/reasoning_content_tool_roundtrip";
const REASONING_TEXT: &str =
    "The user asked for current weather, so I need to call get_weather before answering.";

#[tokio::test]
async fn nonstreaming_reasoning_content_tool_roundtrip() {
    with_local_reasoning_content_cassette(
        "openai_compatible/reasoning_content_tool_roundtrip",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent("llama-cpp-reasoning-model")
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(json!({
                    "reasoning": { "effort": "medium" }
                }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("OpenAI-compatible provider should accept replayed reasoning content")
                .output;

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openai-compatible");
        },
    )
    .await;

    assert_cassette_preserves_reasoning_content(SCENARIO);
}

async fn with_local_reasoning_content_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette =
        crate::cassettes::start_provider_cassette("openai", scenario, "http://scripted.invalid/v1")
            .await;
    let client = OpenAI::new("dummy-openai-compatible-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("OpenAI-compatible cassette client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

fn request_preserves_reasoning_content(body: &Value) -> bool {
    body.get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .any(reasoning_item_preserves_content)
}

fn reasoning_item_preserves_content(item: &Value) -> bool {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|content| {
            content.get("type").and_then(Value::as_str) == Some("reasoning_text")
                && content.get("text").and_then(Value::as_str) == Some(REASONING_TEXT)
        })
}

fn assert_cassette_preserves_reasoning_content(scenario: &str) {
    let cassette_path = cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });
    let interactions = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .collect::<Vec<_>>();

    assert!(
        interactions.iter().any(|interaction| {
            let Some(body) = interaction
                .get("when")
                .and_then(|when| when.get("body"))
                .and_then(serde_yaml::Value::as_str)
            else {
                return false;
            };
            let Ok(body) = serde_json::from_str::<Value>(body) else {
                return false;
            };
            request_preserves_reasoning_content(&body)
        }),
        "expected cassette {} to contain a continuation request preserving reasoning.content",
        cassette_path.display()
    );
}
