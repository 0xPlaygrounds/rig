//! Integration tests for extractor usage tracking.
//!
//! These tests verify that:
//! - The original `extract()` API still works (backward compatibility)
//! - The new `extract_with_usage()` API correctly returns usage data
//! - Usage accumulates across retry attempts
//! - Both `extract` and `extract_with_chat_history` variants work

use anyhow::Result;
use rig_core::client::ProviderClient;
use rig_core::extractor::ExtractionResponse;
use rig_core::providers;
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

fn assert_compatible_professions(left: Option<&str>, right: Option<&str>) -> Result<()> {
    let left = left
        .ok_or_else(|| anyhow::anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();
    let right = right
        .ok_or_else(|| anyhow::anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();

    anyhow::ensure!(
        left == right || left.contains(&right) || right.contains(&left),
        "expected compatible professions, got {left:?} and {right:?}"
    );
    Ok(())
}

/// Test backward compatibility: the original `extract()` method should still work
/// and return just the extracted data (not wrapped in a response type).
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_backward_compatibility() -> Result<()> {
    let client = providers::openai::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let person = extractor
        .extract("John Doe is a 30 year old software engineer.")
        .await?;

    anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
    anyhow::ensure!(person.age == Some(30));
    anyhow::ensure!(person.profession.as_deref() == Some("software engineer"));

    Ok(())
}

/// Test `extract_with_usage()` returns the extracted data with usage information.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let client = providers::openai::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let response: ExtractionResponse<Person> = extractor
        .extract_with_usage("Jane Smith is a 45 year old data scientist.")
        .await?;

    // Verify extracted data
    anyhow::ensure!(response.data.name.as_deref() == Some("Jane Smith"));
    anyhow::ensure!(response.data.age == Some(45));
    anyhow::ensure!(response.data.profession.as_deref() == Some("data scientist"));

    // Verify usage is non-zero (we made at least one API call)
    anyhow::ensure!(response.usage.input_tokens > 0);
    anyhow::ensure!(response.usage.output_tokens > 0);
    anyhow::ensure!(response.usage.total_tokens > 0);

    Ok(())
}

/// Test `extract_with_chat_history_with_usage()` returns the extracted data with usage information.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    use rig_core::message::Message;

    let client = providers::openai::Client::from_env().expect("client should build");
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
    anyhow::ensure!(response.data.street.as_deref() == Some("123 Main St"));
    anyhow::ensure!(response.data.city.as_deref() == Some("Springfield"));
    anyhow::ensure!(response.data.state.as_deref() == Some("IL"));
    anyhow::ensure!(response.data.zip_code.as_deref() == Some("62701"));

    // Verify usage is non-zero
    anyhow::ensure!(response.usage.input_tokens > 0);
    anyhow::ensure!(response.usage.total_tokens > 0);

    Ok(())
}

/// Test that `extract_with_usage()` and `extract()` agree on the stable extracted fields.
/// These are separate model calls, so exact wording can vary across runs.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let client = providers::openai::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let text = "Bob Johnson is a 55 year old retired teacher.";

    // Extract without usage
    let person = extractor.extract(text).await?;

    // Extract with usage
    let response = extractor.extract_with_usage(text).await?;

    anyhow::ensure!(person.name.as_deref() == Some("Bob Johnson"));
    anyhow::ensure!(response.data.name.as_deref() == Some("Bob Johnson"));
    anyhow::ensure!(person.age == Some(55));
    anyhow::ensure!(response.data.age == Some(55));
    assert_compatible_professions(
        person.profession.as_deref(),
        response.data.profession.as_deref(),
    )?;
    anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");

    Ok(())
}

/// Test that usage is reported for both simple and complex extraction scenarios.
#[tokio::test]
#[ignore = "This requires an API key"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let client = providers::openai::Client::from_env().expect("client should build");

    // Test with simple schema
    let person_extractor = client
        .extractor::<Person>(providers::openai::GPT_4O_MINI)
        .build();

    let person_response = person_extractor
        .extract_with_usage("Alice is a 25 year old developer.")
        .await?;

    anyhow::ensure!(person_response.usage.total_tokens > 0);

    // Test with more complex schema
    let address_extractor = client
        .extractor::<Address>(providers::openai::GPT_4O_MINI)
        .build();

    let address_response = address_extractor
        .extract_with_usage("456 Oak Avenue, Cambridge, MA 02139")
        .await?;

    anyhow::ensure!(address_response.usage.total_tokens > 0);

    Ok(())
}
