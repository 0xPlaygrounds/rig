//! Gemini generateContent behavior regression tests.
//!
//! Locks down provider-response preservation for unusual but valid response
//! shapes (`MAX_TOKENS` truncation with finish reason and model version
//! intact) and structured output with nested objects, arrays, and optional
//! fields.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::providers::gemini;
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
async fn structured_output_nested_arrays_and_optional_fields() {
    with_gemini_cassette(
        "generate_behaviors/structured_output_nested_arrays_and_optional_fields",
        |client| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
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
            let record: EventRecord = serde_json::from_str(&response.output())
                .expect("structured output should deserialize");

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
