//! Cassette-backed Anthropic coverage for manually placed prompt-cache
//! breakpoints.
//!
//! A [`CacheControl`] marker attached to a generic message block under the
//! `ANTHROPIC_CACHE_CONTROL_KEY` `additional_params` key must serialize as the
//! `cache_control` field of the corresponding Anthropic content block, at
//! exactly the marked position.
//! See <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{Message, Text, UserContent};
use rig::providers::anthropic::completion::{
    ANTHROPIC_CACHE_CONTROL_KEY, CLAUDE_SONNET_4_6, CacheControl,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
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
            ANTHROPIC_CACHE_CONTROL_KEY: CacheControl::ephemeral(),
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

#[tokio::test]
async fn manual_cache_control_roundtrip() {
    with_anthropic_cassette(
        "manual_cache_control/manual_cache_control_roundtrip",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a helpful assistant that analyzes documents.")
                .temperature(0.0)
                .build();

            let response = agent
                .prompt(breakpoint_prompt())
                .await
                .expect("manual cache-control request should succeed");

            assert_contains_any_case_insensitive(&response, &[CODENAME]);
        },
    )
    .await;

    assert_recorded_request_carries_cache_control(
        "manual_cache_control/manual_cache_control_roundtrip",
    );
}

fn assert_recorded_request_carries_cache_control(scenario: &str) {
    let body = recorded_request_body(scenario);

    let marked = &body["messages"][0]["content"][0];
    assert_eq!(
        marked["cache_control"]["type"], "ephemeral",
        "the marked block should carry cache_control: {body:#}"
    );
    let unmarked = &body["messages"][0]["content"][1];
    assert!(
        unmarked.get("cache_control").is_none(),
        "the unmarked block should not carry cache_control: {body:#}"
    );
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
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
