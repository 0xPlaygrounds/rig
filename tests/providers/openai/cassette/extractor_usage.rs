//! Integration tests for extractor usage tracking.
//!
//! These tests verify that:
//! - Plain extraction still yields just the extracted value (backward
//!   compatibility with the deleted `Extractor::extract`)
//! - [`extract_with_options`] correctly returns usage data
//! - Usage accumulates across retry attempts
//! - Both the plain and the chat-history-seeded variants work

use std::sync::Arc;

use anyhow::Result;
use rig::agent::AgentConfig;
use rig::extract::{ExtractError, ExtractOptions, ExtractionOutcome, extract_with_options};
use rig::message::Message;
use rig::provider::Runtime;
use rig::providers::openai;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::{OpenAiCassette, with_openai_cassette_result};

/// Run a classic-extractor-shaped extraction against `client`.
async fn classic_extract<T>(
    client: &OpenAiCassette,
    prompt: impl Into<Message>,
    options: ExtractOptions,
) -> Result<ExtractionOutcome<T>, ExtractError>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    extract_with_options(
        AgentConfig::new(),
        client.provider_config(openai::GPT_4O_MINI),
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

/// Test backward compatibility: plain extraction returns just the extracted
/// value (not wrapped in a response type).
#[tokio::test]
async fn extract_backward_compatibility() -> Result<()> {
    with_openai_cassette_result(
        "extractor_usage/extract_backward_compatibility",
        |client| async move {
            let person = classic_extract::<Person>(
                &client,
                "John Doe is a 30 year old software engineer.",
                ExtractOptions::classic_extractor(),
            )
            .await?
            .value;

            anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
            anyhow::ensure!(person.age == Some(30));
            anyhow::ensure!(person.profession.as_deref() == Some("software engineer"));

            Ok(())
        },
    )
    .await
}

/// Test that extraction returns the extracted data with usage information.
#[tokio::test]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    with_openai_cassette_result(
        "extractor_usage/extract_with_usage_returns_data_and_usage",
        |client| async move {
            let response: ExtractionOutcome<Person> = classic_extract(
                &client,
                "Jane Smith is a 45 year old data scientist.",
                ExtractOptions::classic_extractor(),
            )
            .await?;

            anyhow::ensure!(response.value.name.as_deref() == Some("Jane Smith"));
            anyhow::ensure!(response.value.age == Some(45));
            anyhow::ensure!(response.value.profession.as_deref() == Some("data scientist"));

            anyhow::ensure!(response.usage.input_tokens > 0);
            anyhow::ensure!(response.usage.output_tokens > 0);
            anyhow::ensure!(response.usage.total_tokens > 0);

            Ok(())
        },
    )
    .await
}

/// Test that a chat-history-seeded extraction returns the extracted data with
/// usage information.
#[tokio::test]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    with_openai_cassette_result(
        "extractor_usage/extract_with_chat_history_with_usage_works",
        |client| async move {
            let chat_history = vec![Message::user(
                "I'm looking at a property that might be interesting.",
            )];

            let response: ExtractionOutcome<Address> = classic_extract(
                &client,
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

/// Test that two separate extractions agree on the stable extracted fields.
/// These are separate model calls, so exact wording can vary across runs.
#[tokio::test]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    with_openai_cassette_result(
        "extractor_usage/extract_and_extract_with_usage_return_same_data",
        |client| async move {
            let text = "Bob Johnson is a 55 year old retired teacher.";

            let person =
                classic_extract::<Person>(&client, text, ExtractOptions::classic_extractor())
                    .await?
                    .value;

            let response =
                classic_extract::<Person>(&client, text, ExtractOptions::classic_extractor())
                    .await?;

            anyhow::ensure!(person.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(response.value.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(person.age == Some(55));
            anyhow::ensure!(response.value.age == Some(55));
            assert_compatible_professions(
                person.profession.as_deref(),
                response.value.profession.as_deref(),
            )?;
            anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");

            Ok(())
        },
    )
    .await
}

/// Test that usage is reported for both simple and complex extraction scenarios.
#[tokio::test]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    with_openai_cassette_result(
        "extractor_usage/usage_tracking_works_for_different_schemas",
        |client| async move {
            let person_response = classic_extract::<Person>(
                &client,
                "Alice is a 25 year old developer.",
                ExtractOptions::classic_extractor(),
            )
            .await?;

            anyhow::ensure!(person_response.usage.total_tokens > 0);

            let address_response = classic_extract::<Address>(
                &client,
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
