//! Integration tests for DeepSeek extractor usage tracking.

use anyhow::{Result, anyhow};
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::extractor::ExtractionResponse;
use rig_core::message::Message;
use rig_core::providers::deepseek;
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
        .ok_or_else(|| anyhow!("profession should be present"))?
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
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extract_backward_compatibility() -> Result<()> {
    let client = deepseek::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(deepseek::DEEPSEEK_V4_FLASH)
        .build();

    let person = extractor
        .extract("John Doe is a 30 year old software engineer.")
        .await?;

    anyhow::ensure!(
        person.name == Some("John Doe".to_string()),
        "expected name John Doe, got {:?}",
        person.name
    );
    anyhow::ensure!(
        person.age == Some(30),
        "expected age 30, got {:?}",
        person.age
    );
    assert_compatible_professions(person.profession.as_deref(), "software engineer")?;

    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let client = deepseek::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(deepseek::DEEPSEEK_V4_FLASH)
        .build();

    let response: ExtractionResponse<Person> = extractor
        .extract_with_usage("Jane Smith is a 45 year old data scientist.")
        .await?;

    anyhow::ensure!(
        response.data.name == Some("Jane Smith".to_string()),
        "expected name Jane Smith, got {:?}",
        response.data.name
    );
    anyhow::ensure!(
        response.data.age == Some(45),
        "expected age 45, got {:?}",
        response.data.age
    );
    assert_compatible_professions(response.data.profession.as_deref(), "data scientist")?;
    anyhow::ensure!(response.usage.input_tokens > 0, "expected input tokens");
    anyhow::ensure!(response.usage.output_tokens > 0, "expected output tokens");
    anyhow::ensure!(response.usage.total_tokens > 0, "expected total tokens");

    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    let client = deepseek::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Address>(deepseek::DEEPSEEK_V4_FLASH)
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

    anyhow::ensure!(
        response.data.street == Some("123 Main St".to_string()),
        "expected street 123 Main St, got {:?}",
        response.data.street
    );
    anyhow::ensure!(
        response.data.city == Some("Springfield".to_string()),
        "expected city Springfield, got {:?}",
        response.data.city
    );
    anyhow::ensure!(
        response.data.state == Some("IL".to_string()),
        "expected state IL, got {:?}",
        response.data.state
    );
    anyhow::ensure!(
        response.data.zip_code == Some("62701".to_string()),
        "expected zip code 62701, got {:?}",
        response.data.zip_code
    );
    anyhow::ensure!(response.usage.input_tokens > 0, "expected input tokens");
    anyhow::ensure!(response.usage.total_tokens > 0, "expected total tokens");

    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let client = deepseek::Client::from_env().expect("client should build");
    let extractor = client
        .extractor::<Person>(deepseek::DEEPSEEK_V4_FLASH)
        .build();

    let text = "Bob Johnson is a 55 year old retired teacher.";
    let person = extractor.extract(text).await?;
    let response = extractor.extract_with_usage(text).await?;

    anyhow::ensure!(
        person.name == Some("Bob Johnson".to_string()),
        "expected extracted name Bob Johnson, got {:?}",
        person.name
    );
    anyhow::ensure!(
        response.data.name == Some("Bob Johnson".to_string()),
        "expected usage response name Bob Johnson, got {:?}",
        response.data.name
    );
    anyhow::ensure!(
        person.age == Some(55),
        "expected extracted age 55, got {:?}",
        person.age
    );
    anyhow::ensure!(
        response.data.age == Some(55),
        "expected usage response age 55, got {:?}",
        response.data.age
    );
    assert_compatible_professions(person.profession.as_deref(), "retired teacher")?;
    assert_compatible_professions(response.data.profession.as_deref(), "retired teacher")?;
    anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");

    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let client = deepseek::Client::from_env().expect("client should build");

    let person_extractor = client
        .extractor::<Person>(deepseek::DEEPSEEK_V4_FLASH)
        .build();
    let person_response = person_extractor
        .extract_with_usage("Alice is a 25 year old developer.")
        .await?;
    anyhow::ensure!(
        person_response.usage.total_tokens > 0,
        "expected person usage tokens"
    );

    let address_extractor = client
        .extractor::<Address>(deepseek::DEEPSEEK_V4_FLASH)
        .build();
    let address_response = address_extractor
        .extract_with_usage("456 Oak Avenue, Cambridge, MA 02139")
        .await?;
    anyhow::ensure!(
        address_response.usage.total_tokens > 0,
        "expected address usage tokens"
    );

    Ok(())
}
