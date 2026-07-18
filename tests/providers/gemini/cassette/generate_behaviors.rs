//! Gemini generateContent behavior regression tests.
//!
//! Locks down provider-response preservation for unusual but valid response
//! shapes (`MAX_TOKENS` truncation with finish reason and model version
//! intact) and structured output with nested objects, arrays, and optional
//! fields.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::message::AssistantContent;
use rig::prelude::AgentClientExt;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::FinishReason;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_gemini_cassette;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct EventLocation {
    city: String,
    venue: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct EventRecord {
    title: String,
    location: EventLocation,
    attendees: Vec<String>,
    #[schemars(required)]
    note: Option<String>,
}

#[tokio::test]
async fn max_tokens_truncation_preserves_finish_reason_and_partial_text() {
    with_gemini_cassette(
        "generate_behaviors/max_tokens_truncation_preserves_finish_reason_and_partial_text",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            // Thinking is disabled so the token budget is spent on visible
            // text and the truncated candidate still carries partial output.
            let request = model
                .completion_request(
                    "Write a story of at least 150 words about a lighthouse keeper.",
                )
                .preamble("You are a storyteller.".to_string())
                .temperature(0.0)
                .max_tokens(48)
                .additional_params(serde_json::json!({
                    "generationConfig": {
                        "thinkingConfig": { "thinkingBudget": 0 }
                    }
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a truncated response should still convert, not error");

            let candidate = response
                .raw_response
                .candidates
                .first()
                .expect("response should carry a candidate");
            assert!(
                matches!(candidate.finish_reason, Some(FinishReason::MaxTokens)),
                "hitting maxOutputTokens should preserve the MAX_TOKENS finish reason, \
                 got {:?}",
                candidate.finish_reason
            );
            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.trim().is_empty(),
                "partial output text should still be surfaced"
            );
            assert!(
                response
                    .raw_response
                    .model_version
                    .as_deref()
                    .is_some_and(|version| !version.is_empty()),
                "provider response should preserve the model version"
            );
            assert!(
                response.usage.output_tokens > 0,
                "usage should reflect the truncated candidate, got {:?}",
                response.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn structured_output_nested_arrays_and_optional_fields() {
    with_gemini_cassette(
        "generate_behaviors/structured_output_nested_arrays_and_optional_fields",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .output_schema::<EventRecord>()
                .temperature(0.0)
                .build();

            let response = agent
                .prompt(
                    "Return an event record for a Rust meetup titled \"Rust Seattle June\" \
                     at the venue \"Fremont Hall\" in Seattle, with attendees Alice and Bob. \
                     No note is needed.",
                )
                .await
                .expect("structured output prompt should succeed");
            let record: EventRecord =
                serde_json::from_str(&response).expect("structured output should deserialize");

            assert!(!record.title.trim().is_empty(), "title should be populated");
            assert_eq!(
                record.location.city.to_ascii_lowercase(),
                "seattle",
                "nested object field should follow the prompt"
            );
            assert!(
                !record.location.venue.trim().is_empty(),
                "nested venue should be populated"
            );
            assert_eq!(
                record.attendees.len(),
                2,
                "array field should carry both attendees: {:?}",
                record.attendees
            );
        },
    )
    .await;
}
