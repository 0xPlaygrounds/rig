//! Native ECS extraction with the original typed assertions.
use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette_result};
use super::extractor_usage::{Address, Person, assert_compatible_professions};
use crate::ecs_extractor::{EcsExtractor, Extracted as TypedPromptResponse};
use anyhow::Result;
use rig::message::Message;
use rig::prelude::*;
#[tokio::test]
async fn extract_backward_compatibility() -> Result<()> {
    with_openrouter_cassette_result(
        "extractor_usage/extract_backward_compatibility",
        |client| async move {
            let mut extractor =
                EcsExtractor::<Person>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let person = extractor
                .extract("John Doe is a 30 year old software engineer.", &[])
                .await?
                .output;
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
    with_openrouter_cassette_result(
        "extractor_usage/extract_with_usage_returns_data_and_usage",
        |client| async move {
            let mut extractor =
                EcsExtractor::<Person>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let response: TypedPromptResponse<Person> = extractor
                .extract("Jane Smith is a 45 year old data scientist.", &[])
                .await?;
            anyhow::ensure!(response.output.name.as_deref() == Some("Jane Smith"));
            anyhow::ensure!(response.output.age == Some(45));
            assert_compatible_professions(response.output.profession.as_deref(), "data scientist")?;
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
    with_openrouter_cassette_result(
        "extractor_usage/extract_with_chat_history_with_usage_works",
        |client| async move {
            let mut extractor =
                EcsExtractor::<Address>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let chat_history = vec![Message::user(
                "I'm looking at a property that might be interesting.",
            )];
            let response: TypedPromptResponse<Address> = extractor
                .extract(
                    "The address is 123 Main St in Springfield, IL 62701.",
                    &chat_history,
                )
                .await?;
            anyhow::ensure!(response.output.street.as_deref() == Some("123 Main St"));
            anyhow::ensure!(response.output.city.as_deref() == Some("Springfield"));
            anyhow::ensure!(response.output.state.as_deref() == Some("IL"));
            anyhow::ensure!(response.output.zip_code.as_deref() == Some("62701"));
            anyhow::ensure!(response.usage.input_tokens > 0);
            anyhow::ensure!(response.usage.total_tokens > 0);
            Ok(())
        },
    )
    .await
}
#[tokio::test]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    with_openrouter_cassette_result(
        "extractor_usage/extract_and_extract_with_usage_return_same_data",
        |client| async move {
            let mut extractor =
                EcsExtractor::<Person>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let text = "Bob Johnson is a 55 year old retired teacher.";
            let person = extractor.extract(text, &[]).await?.output;
            let response = extractor.extract(text, &[]).await?;
            anyhow::ensure!(person.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(response.output.name.as_deref() == Some("Bob Johnson"));
            anyhow::ensure!(person.age == Some(55));
            anyhow::ensure!(response.output.age == Some(55));
            assert_compatible_professions(person.profession.as_deref(), "retired teacher")?;
            assert_compatible_professions(
                response.output.profession.as_deref(),
                "retired teacher",
            )?;
            anyhow::ensure!(response.usage.total_tokens > 0, "usage should be populated");
            Ok(())
        },
    )
    .await
}
#[tokio::test]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    with_openrouter_cassette_result(
        "extractor_usage/usage_tracking_works_for_different_schemas",
        |client| async move {
            let mut person_extractor =
                EcsExtractor::<Person>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let person_response = person_extractor
                .extract("Alice is a 25 year old developer.", &[])
                .await?;
            anyhow::ensure!(person_response.usage.total_tokens > 0);
            let mut address_extractor =
                EcsExtractor::<Address>::new(client.completion_model(DEFAULT_MODEL), None, None);
            let address_response = address_extractor
                .extract("456 Oak Avenue, Cambridge, MA 02139", &[])
                .await?;
            anyhow::ensure!(address_response.usage.total_tokens > 0);
            Ok(())
        },
    )
    .await
}
