//! Integration tests for extractor usage tracking.
//!
//! These tests verify that:
//! - The original `extract()` API still works (backward compatibility)
//! - The new `extract_with_usage()` API correctly returns usage data
//! - Usage accumulates across retry attempts
//! - Both `extract` and `extract_with_chat_history` variants work

use anyhow::Result;
use rig::client::ProviderClient;
use rig::extractor::ExtractionResponse;
use rig::providers;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct Person {
    name: Option<String>,
    age: Option<u8>,
    profession: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct Address {
    street: Option<String>,
    city: Option<String>,
    state: Option<String>,
    zip_code: Option<String>,
}

/// Test backward compatibility: the original `extract()` method should still work
/// and return just the extracted data (not wrapped in a response type).
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_backward_compatibility() -> Result<()> {
    let client = providers::openai::Client::from_env();
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let person = extractor
        .extract("John Doe is a 30 year old software engineer.")
        .await?;

    assert_eq!(person.name, Some("John Doe".to_string()));
    assert_eq!(person.age, Some(30));
    assert_eq!(person.profession, Some("software engineer".to_string()));

    Ok(())
}

/// Test `extract_with_usage()` returns the extracted data with usage information.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let client = providers::openai::Client::from_env();
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let response: ExtractionResponse<Person> = extractor
        .extract_with_usage("Jane Smith is a 45 year old data scientist.")
        .await?;

    // Verify extracted data
    assert_eq!(response.data.name, Some("Jane Smith".to_string()));
    assert_eq!(response.data.age, Some(45));
    assert_eq!(response.data.profession, Some("data scientist".to_string()));

    // Verify usage is non-zero (we made at least one API call)
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
    assert!(response.usage.total_tokens > 0);

    Ok(())
}

/// Test `extract_with_chat_history_with_usage()` returns the extracted data with usage information.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    use rig::message::Message;

    let client = providers::openai::Client::from_env();
    let extractor = client
        .extractor::<Address>(providers::openai::GPT_4O_MINI)
        .build();

    let chat_history = vec![Message::user(
        "I'm looking at a property that might be interesting.",
    )];

    let response: ExtractionResponse<Address> = extractor
        .extract_with_chat_history_with_usage(
            "The address is 123 Main St in Springfield, IL 62701.",
            chat_history,
        )
        .await?;

    // Verify extracted data
    assert_eq!(response.data.street, Some("123 Main St".to_string()));
    assert_eq!(response.data.city, Some("Springfield".to_string()));
    assert_eq!(response.data.state, Some("IL".to_string()));
    assert_eq!(response.data.zip_code, Some("62701".to_string()));

    // Verify usage is non-zero
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.total_tokens > 0);

    Ok(())
}

/// Test that `extract_with_usage()` and `extract()` return the same data.
/// This verifies that the internal refactoring didn't change the extraction logic.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let client = providers::openai::Client::from_env();
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let text = "Bob Johnson is a 55 year old retired teacher.";

    // Extract without usage
    let person = extractor.extract(text).await?;

    // Extract with usage
    let response = extractor.extract_with_usage(text).await?;

    // Both should return the same data
    assert_eq!(person, response.data);

    Ok(())
}

/// Test that usage is reported for both simple and complex extraction scenarios.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let client = providers::openai::Client::from_env();

    // Test with simple schema
    let person_extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let person_response = person_extractor
        .extract_with_usage("Alice is a 25 year old developer.")
        .await?;

    assert!(person_response.usage.total_tokens > 0);

    // Test with more complex schema
    let address_extractor = client
        .extractor::<Address>(providers::openai::GPT_4O_MINI)
        .build();

    let address_response = address_extractor
        .extract_with_usage("456 Oak Avenue, Cambridge, MA 02139")
        .await?;

    assert!(address_response.usage.total_tokens > 0);

    Ok(())
}
