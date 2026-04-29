//! Integration tests for xAI extractor usage tracking.

use anyhow::Result;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::extractor::ExtractionResponse;
use rig_core::message::Message;
use rig_core::providers::xai;
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

fn assert_compatible_professions(left: Option<&str>, right: &str) -> Result<()> {
    let left = left
        .ok_or_else(|| anyhow::anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();
    let right = right.trim().to_ascii_lowercase();

    anyhow::ensure!(
        left == right || left.contains(&right) || right.contains(&left),
        "expected compatible professions, got {left:?} and {right:?}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn extract_backward_compatibility() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<Person>(xai::GROK_3_MINI).build();

    let person = extractor
        .extract("John Doe is a 30 year old software engineer.")
        .await?;

    anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
    anyhow::ensure!(person.age == Some(30));
    assert_compatible_professions(person.profession.as_deref(), "software engineer")?;

    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<Person>(xai::GROK_3_MINI).build();

    let response: ExtractionResponse<Person> = extractor
        .extract_with_usage("Jane Smith is a 45 year old data scientist.")
        .await?;

    anyhow::ensure!(response.data.name.as_deref() == Some("Jane Smith"));
    anyhow::ensure!(response.data.age == Some(45));
    assert_compatible_professions(response.data.profession.as_deref(), "data scientist")?;
    anyhow::ensure!(response.usage.input_tokens > 0);
    anyhow::ensure!(response.usage.output_tokens > 0);
    anyhow::ensure!(response.usage.total_tokens > 0);

    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<Address>(xai::GROK_3_MINI).build();

    let chat_history = vec![Message::user(
        "I'm looking at a property that might be interesting.",
    )];

    let response: ExtractionResponse<Address> = extractor
        .extract_with_chat_history_with_usage(
            "The address is 123 Main St in Springfield, IL 62701.",
            chat_history,
        )
        .await?;

    anyhow::ensure!(response.data.street.as_deref() == Some("123 Main St"));
    anyhow::ensure!(response.data.city.as_deref() == Some("Springfield"));
    anyhow::ensure!(response.data.state.as_deref() == Some("IL"));
    anyhow::ensure!(response.data.zip_code.as_deref() == Some("62701"));
    anyhow::ensure!(response.usage.input_tokens > 0);
    anyhow::ensure!(response.usage.total_tokens > 0);

    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<Person>(xai::GROK_3_MINI).build();

    let text = "Bob Johnson is a 55 year old retired teacher.";
    let person = extractor.extract(text).await?;
    let response = extractor.extract_with_usage(text).await?;

    anyhow::ensure!(person.name.as_deref() == Some("Bob Johnson"));
    anyhow::ensure!(response.data.name.as_deref() == Some("Bob Johnson"));
    anyhow::ensure!(person.age == Some(55));
    anyhow::ensure!(response.data.age == Some(55));
    assert_compatible_professions(person.profession.as_deref(), "retired teacher")?;
    assert_compatible_professions(response.data.profession.as_deref(), "retired teacher")?;
    anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");

    Ok(())
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");

    let person_extractor = client.extractor::<Person>(xai::GROK_3_MINI).build();
    let person_response = person_extractor
        .extract_with_usage("Alice is a 25 year old developer.")
        .await?;
    anyhow::ensure!(person_response.usage.total_tokens > 0);

    let address_extractor = client.extractor::<Address>(xai::GROK_3_MINI).build();
    let address_response = address_extractor
        .extract_with_usage("456 Oak Avenue, Cambridge, MA 02139")
        .await?;
    anyhow::ensure!(address_response.usage.total_tokens > 0);

    Ok(())
}
