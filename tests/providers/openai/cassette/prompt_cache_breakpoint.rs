//! Cassette-backed OpenAI Responses coverage for manually placed prompt-cache
//! breakpoints (GPT-5.6+).
//!
//! A breakpoint attached to a generic message block under the
//! `OPENAI_PROMPT_CACHE_BREAKPOINT_KEY` `additional_params` key must serialize
//! as the `prompt_cache_breakpoint` field of the corresponding `input_text`
//! content part, and `prompt_cache_options` set through request
//! `additional_params` must reach the top level of the request body.
//! See <https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-breakpoints>.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, Message, Text, UserContent};
use rig::providers::openai;
use rig::providers::openai::responses_api::{
    OPENAI_PROMPT_CACHE_BREAKPOINT_KEY, PromptCacheBreakpoint,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::support::assert_contains_any_case_insensitive;

const CODENAME: &str = "aurora-lattice-7";
const PROMPT: &str = "According to the reference document, what is the project codename? Reply with the codename only.";

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

/// A reference document comfortably above the ~1024-token minimum cacheable
/// prefix size, so the marked block is a real cache candidate when recorded.
fn reference_text() -> String {
    let mut text = format!("Project reference document. The project codename is {CODENAME}.\n");
    for section in 1..=160 {
        text.push_str(&format!(
            "Section {section}: every subsystem reports its status to the central coordinator \
             at a fixed interval, and each configuration change is recorded in the audit log \
             with a timestamp and the identity of the operator who made the change.\n"
        ));
    }
    text
}

fn breakpoint_prompt() -> Message {
    let reference = Text {
        text: reference_text(),
        additional_params: Some(json!({
            OPENAI_PROMPT_CACHE_BREAKPOINT_KEY: PromptCacheBreakpoint::explicit(),
        })),
    };

    Message::User {
        content: OneOrMany::many(vec![
            UserContent::Text(reference),
            UserContent::text(PROMPT),
        ])
        .expect("prompt content should be non-empty"),
    }
}

fn assistant_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[tokio::test]
async fn manual_breakpoint_roundtrip() {
    super::super::support::with_openai_cassette(
        "prompt_cache_breakpoint/manual_breakpoint_roundtrip",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_5_6)
                .completion_request(breakpoint_prompt())
                .additional_params(json!({
                    "prompt_cache_key": "rig-tests:prompt-cache-breakpoint",
                    "prompt_cache_options": { "mode": "explicit" }
                }))
                .send()
                .await
                .expect("prompt-cache breakpoint request should succeed");

            assert_contains_any_case_insensitive(&assistant_text(&response.choice), &[CODENAME]);
        },
    )
    .await;

    assert_recorded_request_carries_cache_fields(
        "prompt_cache_breakpoint/manual_breakpoint_roundtrip",
    );
}

fn assert_recorded_request_carries_cache_fields(scenario: &str) {
    let body = recorded_request_body(scenario);

    assert_eq!(
        body["prompt_cache_options"]["mode"], "explicit",
        "request should carry prompt_cache_options: {body:#}"
    );
    // The cassette recorder redacts identifier-like values, so assert presence
    // rather than the exact key value.
    assert!(
        body["prompt_cache_key"].is_string(),
        "request should carry prompt_cache_key: {body:#}"
    );

    let marked = &body["input"][0]["content"][0];
    assert_eq!(
        marked["prompt_cache_breakpoint"]["mode"], "explicit",
        "the marked block should carry the breakpoint: {body:#}"
    );
    let unmarked = &body["input"][0]["content"][1];
    assert!(
        unmarked.get("prompt_cache_breakpoint").is_none(),
        "the unmarked block should not carry a breakpoint: {body:#}"
    );
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .find_map(|document| {
            let interaction = RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize");
            interaction
                .when
                .body
                .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        })
        .unwrap_or_else(|| panic!("expected cassette {scenario} to contain a JSON request body"))
}
