//! AWS Bedrock raw completion cassette coverage ported from OpenAI completions tests.

use rig::bedrock;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response,
};

const SCENARIO: &str = "raw_completion/raw_response_text_matches_normalized_choice_text";

/// Extracts the assistant text from the recorded Bedrock Converse wire body,
/// mirroring the provider's raw `get_text_response` semantics (all text blocks
/// joined with newlines).
fn recorded_raw_text() -> String {
    let bodies = crate::cassettes::recorded_response_bodies("bedrock", SCENARIO);
    let body: serde_json::Value = serde_json::from_str(
        bodies
            .last()
            .expect("cassette should contain a recorded response body"),
    )
    .expect("recorded Bedrock Converse body should be JSON");
    let segments = body["output"]["message"]["content"]
        .as_array()
        .expect("raw Bedrock output should contain message content blocks")
        .iter()
        .filter_map(|block| block.get("text").and_then(serde_json::Value::as_str))
        .collect::<Vec<_>>();
    assert!(
        !segments.is_empty(),
        "raw Bedrock response should contain assistant text"
    );
    segments.join("\n")
}

#[tokio::test]
async fn raw_response_text_matches_normalized_choice_text() {
    with_bedrock_cassette("raw_completion/raw_response_text_matches_normalized_choice_text", |client| async move {
        let response = client
            .completion_model(bedrock::completion::AMAZON_NOVA_LITE)
            .completion_request(RAW_TEXT_RESPONSE_PROMPT)
            .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
            .temperature(0.0)
            .send()
            .await
            .expect("raw Bedrock request should succeed");

        let normalized_text = assistant_text_response(&response.choice)
            .expect("normalized Bedrock response should contain assistant text");
        let raw_text = recorded_raw_text();

        assert_nonempty_response(&normalized_text);
        assert_nonempty_response(&raw_text);
        assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
        assert_eq!(raw_text.trim(), normalized_text.trim());
    })
    .await;
}
