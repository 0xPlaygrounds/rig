//! Integration tests for DeepSeek extractor usage tracking.

use anyhow::{Result, anyhow};
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, ExtractionOutcome, extract_with_options};
use rig::message::Message;
use rig::provider::{ProviderConfig, Runtime};
use rig::providers::deepseek;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::support::with_deepseek_cassette_result;

/// One classic-`Extractor<T>` exchange expressed through the free-function
/// extraction surface that replaced it.
async fn classic_extract<T>(
    provider: ProviderConfig,
    text: &str,
    options: ExtractOptions,
) -> anyhow::Result<ExtractionOutcome<T>>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    Ok(extract_with_options::<T>(
        AgentConfig::new(),
        provider,
        Arc::new(Runtime::new()),
        text,
        options,
    )
    .await?)
}

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
async fn extract_backward_compatibility() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/extract_backward_compatibility",
        |env| async move {
            let person = classic_extract::<Person>(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                "John Doe is a 30 year old software engineer.",
                ExtractOptions::classic_extractor(),
            )
            .await?
            .value;

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
        },
    )
    .await
}

#[tokio::test]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/extract_with_usage_returns_data_and_usage",
        |env| async move {
            let response: ExtractionOutcome<Person> = classic_extract(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                "Jane Smith is a 45 year old data scientist.",
                ExtractOptions::classic_extractor(),
            )
            .await?;

            anyhow::ensure!(
                response.value.name == Some("Jane Smith".to_string()),
                "expected name Jane Smith, got {:?}",
                response.value.name
            );
            anyhow::ensure!(
                response.value.age == Some(45),
                "expected age 45, got {:?}",
                response.value.age
            );
            assert_compatible_professions(response.value.profession.as_deref(), "data scientist")?;
            anyhow::ensure!(response.usage.input_tokens > 0, "expected input tokens");
            anyhow::ensure!(response.usage.output_tokens > 0, "expected output tokens");
            anyhow::ensure!(response.usage.total_tokens > 0, "expected total tokens");

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/extract_with_chat_history_with_usage_works",
        |env| async move {
            let chat_history = vec![Message::user(
                "I'm looking at a property that might be interesting.",
            )];

            let response: ExtractionOutcome<Address> = classic_extract(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                "The address is 123 Main St in Springfield, IL 62701.",
                ExtractOptions::classic_extractor().with_history(chat_history),
            )
            .await?;

            anyhow::ensure!(
                response.value.street == Some("123 Main St".to_string()),
                "expected street 123 Main St, got {:?}",
                response.value.street
            );
            anyhow::ensure!(
                response.value.city == Some("Springfield".to_string()),
                "expected city Springfield, got {:?}",
                response.value.city
            );
            anyhow::ensure!(
                response.value.state == Some("IL".to_string()),
                "expected state IL, got {:?}",
                response.value.state
            );
            anyhow::ensure!(
                response.value.zip_code == Some("62701".to_string()),
                "expected zip code 62701, got {:?}",
                response.value.zip_code
            );
            anyhow::ensure!(response.usage.input_tokens > 0, "expected input tokens");
            anyhow::ensure!(response.usage.total_tokens > 0, "expected total tokens");

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/extract_and_extract_with_usage_return_same_data",
        |env| async move {
            let text = "Bob Johnson is a 55 year old retired teacher.";
            let person = classic_extract::<Person>(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                text,
                ExtractOptions::classic_extractor(),
            )
            .await?
            .value;
            let response = classic_extract::<Person>(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                text,
                ExtractOptions::classic_extractor(),
            )
            .await?;

            anyhow::ensure!(
                person.name == Some("Bob Johnson".to_string()),
                "expected extracted name Bob Johnson, got {:?}",
                person.name
            );
            anyhow::ensure!(
                response.value.name == Some("Bob Johnson".to_string()),
                "expected usage response name Bob Johnson, got {:?}",
                response.value.name
            );
            anyhow::ensure!(
                person.age == Some(55),
                "expected extracted age 55, got {:?}",
                person.age
            );
            anyhow::ensure!(
                response.value.age == Some(55),
                "expected usage response age 55, got {:?}",
                response.value.age
            );
            assert_compatible_professions(person.profession.as_deref(), "retired teacher")?;
            assert_compatible_professions(response.value.profession.as_deref(), "retired teacher")?;
            anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/usage_tracking_works_for_different_schemas",
        |env| async move {
            let person_response = classic_extract::<Person>(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                "Alice is a 25 year old developer.",
                ExtractOptions::classic_extractor(),
            )
            .await?;
            anyhow::ensure!(
                person_response.usage.total_tokens > 0,
                "expected person usage tokens"
            );

            let address_response = classic_extract::<Address>(
                env.provider(deepseek::DEEPSEEK_V4_FLASH),
                "456 Oak Avenue, Cambridge, MA 02139",
                ExtractOptions::classic_extractor(),
            )
            .await?;
            anyhow::ensure!(
                address_response.usage.total_tokens > 0,
                "expected address usage tokens"
            );

            Ok(())
        },
    )
    .await
}
