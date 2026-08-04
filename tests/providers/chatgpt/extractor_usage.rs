//! Integration tests for ChatGPT extractor usage tracking.

use anyhow::{Result, anyhow};
use rig::agent::Agent;
use rig::extract::{ExtractOptions, ExtractionOutcome};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::chatgpt::{LIVE_MODEL, live_agent};

/// One classic-`Extractor<T>` exchange expressed through the free-function
/// extraction surface that replaced it.
async fn classic_extract<T>(
    agent: &Agent,
    text: &str,
    options: ExtractOptions,
) -> anyhow::Result<ExtractionOutcome<T>>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    let mut runner = agent
        .extractor(text)
        .history(options.history)
        .retries(options.retries);
    if let Some(preamble) = options.preamble {
        runner = runner.preamble(preamble);
    }
    Ok(runner.run_with_usage::<T>().await?)
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
        .ok_or_else(|| anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();
    let right = right
        .ok_or_else(|| anyhow!("profession should be present"))?
        .trim()
        .to_ascii_lowercase();

    anyhow::ensure!(
        left == right || left.contains(&right) || right.contains(&left),
        "expected compatible professions, got {left:?} and {right:?}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extract_backward_compatibility() -> Result<()> {
    let agent = live_agent(LIVE_MODEL).await.build();
    let person = classic_extract::<Person>(
        &agent,
        "John Doe is a 30 year old software engineer.",
        ExtractOptions::classic_extractor(),
    )
    .await?
    .value;

    anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
    anyhow::ensure!(person.age == Some(30));
    anyhow::ensure!(person.profession.as_deref() == Some("software engineer"));

    Ok(())
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    let agent = live_agent(LIVE_MODEL).await.build();
    let response: ExtractionOutcome<Person> = classic_extract(
        &agent,
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
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    use rig::message::Message;

    let agent = live_agent(LIVE_MODEL).await.build();
    let chat_history = vec![Message::user(
        "I'm looking at a property that might be interesting.",
    )];

    let response: ExtractionOutcome<Address> = classic_extract(
        &agent,
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
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    let agent = live_agent(LIVE_MODEL).await.build();
    let text = "Bob Johnson is a 55 year old retired teacher.";

    let person = classic_extract::<Person>(&agent, text, ExtractOptions::classic_extractor())
        .await?
        .value;
    let response =
        classic_extract::<Person>(&agent, text, ExtractOptions::classic_extractor()).await?;

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
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    let agent = live_agent(LIVE_MODEL).await.build();

    let person_response = classic_extract::<Person>(
        &agent,
        "Alice is a 25 year old developer.",
        ExtractOptions::classic_extractor(),
    )
    .await?;
    anyhow::ensure!(person_response.usage.total_tokens > 0);

    let address_response = classic_extract::<Address>(
        &agent,
        "456 Oak Avenue, Cambridge, MA 02139",
        ExtractOptions::classic_extractor(),
    )
    .await?;
    anyhow::ensure!(address_response.usage.total_tokens > 0);

    Ok(())
}
