//! Integration tests for xAI extractor usage tracking.

use std::sync::Arc;

use anyhow::Result;
use rig::agent::AgentConfig;
use rig::extract::{ExtractError, ExtractOptions, ExtractionOutcome, extract_with_options};
use rig::message::Message;
use rig::provider::Runtime;
use rig::providers::xai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::support::{XaiCassetteEnv, with_xai_cassette_result};

/// Run a classic-extractor-shaped extraction against `env`.
async fn classic_extract<T>(
    env: &XaiCassetteEnv,
    prompt: impl Into<Message>,
    options: ExtractOptions,
) -> Result<ExtractionOutcome<T>, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    extract_with_options(
        AgentConfig::new(),
        env.provider_config(xai::GROK_3_MINI),
        Arc::new(Runtime::new()),
        prompt,
        options,
    )
    .await
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
async fn extract_backward_compatibility() -> Result<()> {
    with_xai_cassette_result(
        "extractor_usage/extract_backward_compatibility",
        |env| async move {
            let person = classic_extract::<Person>(
                &env,
                "John Doe is a 30 year old software engineer.",
                ExtractOptions::classic_extractor(),
            )
            .await?
            .value;

            anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
            anyhow::ensure!(person.age == Some(30));
            assert_compatible_professions(person.profession.as_deref(), "software engineer")?;

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    with_xai_cassette_result(
        "extractor_usage/extract_with_usage_returns_data_and_usage",
        |env| async move {
            let response = classic_extract::<Person>(
                &env,
                "Jane Smith is a 45 year old data scientist.",
                ExtractOptions::classic_extractor(),
            )
            .await?;

            anyhow::ensure!(response.value.name.as_deref() == Some("Jane Smith"));
            anyhow::ensure!(response.value.age == Some(45));
            assert_compatible_professions(response.value.profession.as_deref(), "data scientist")?;
            anyhow::ensure!(response.usage.input_tokens > 0);
            anyhow::ensure!(response.usage.output_tokens > 0);
            anyhow::ensure!(response.usage.total_tokens > 0);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    with_xai_cassette_result(
        "extractor_usage/extract_with_chat_history_with_usage_works",
        |env| async move {
            let chat_history = vec![Message::user(
                "I'm looking at a property that might be interesting.",
            )];

            let response = classic_extract::<Address>(
                &env,
                "The address is 123 Main St in Springfield, IL 62701.",
                ExtractOptions::classic_extractor().with_history(chat_history),
            )
            .await?;

            anyhow::ensure!(response.value.street.as_deref() == Some("123 Main St"));
            anyhow::ensure!(response.value.city.as_deref() == Some("Springfield"));
            anyhow::ensure!(response.value.state.as_deref() == Some("IL"));
            anyhow::ensure!(response.value.zip_code.as_deref() == Some("62701"));
            anyhow::ensure!(response.usage.input_tokens > 0);
            anyhow::ensure!(response.usage.total_tokens > 0);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    with_xai_cassette_result(
        "extractor_usage/extract_and_extract_with_usage_return_same_data",
        |env| async move {
            let text = "Bob Johnson is a 55 year old retired teacher.";
            let person = classic_extract::<Person>(&env, text, ExtractOptions::classic_extractor())
                .await?
                .value;
            let response =
                classic_extract::<Person>(&env, text, ExtractOptions::classic_extractor()).await?;

            anyhow::ensure!(person.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(response.value.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(person.age == Some(55));
            anyhow::ensure!(response.value.age == Some(55));
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
    with_xai_cassette_result(
        "extractor_usage/usage_tracking_works_for_different_schemas",
        |env| async move {
            let person_response = classic_extract::<Person>(
                &env,
                "Alice is a 25 year old developer.",
                ExtractOptions::classic_extractor(),
            )
            .await?;
            anyhow::ensure!(person_response.usage.total_tokens > 0);

            let address_response = classic_extract::<Address>(
                &env,
                "456 Oak Avenue, Cambridge, MA 02139",
                ExtractOptions::classic_extractor(),
            )
            .await?;
            anyhow::ensure!(address_response.usage.total_tokens > 0);

            Ok(())
        },
    )
    .await
}
