//! Native ECS extraction with the original typed assertions.
use super::extractor_usage::{Address, Person, assert_compatible_professions};
use super::support::with_deepseek_cassette_result;
use crate::ecs_extractor::{EcsExtractor, Extracted as TypedPromptResponse};
use anyhow::Result;
use rig::message::Message;
use rig::prelude::*;
use rig::providers::deepseek;
#[tokio::test]
async fn extract_backward_compatibility() -> Result<()> {
    with_deepseek_cassette_result(
        "extractor_usage/extract_backward_compatibility",
        |client| async move {
            let mut extractor = EcsExtractor::<Person>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let person = extractor
                .extract("John Doe is a 30 year old software engineer.", &[])
                .await?
                .output;
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
        |client| async move {
            let mut extractor = EcsExtractor::<Person>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let response: TypedPromptResponse<Person> = extractor
                .extract("Jane Smith is a 45 year old data scientist.", &[])
                .await?;
            anyhow::ensure!(
                response.output.name == Some("Jane Smith".to_string()),
                "expected name Jane Smith, got {:?}",
                response.output.name
            );
            anyhow::ensure!(
                response.output.age == Some(45),
                "expected age 45, got {:?}",
                response.output.age
            );
            assert_compatible_professions(response.output.profession.as_deref(), "data scientist")?;
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
        |client| async move {
            let mut extractor = EcsExtractor::<Address>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let chat_history = vec![Message::user(
                "I'm looking at a property that might be interesting.",
            )];
            let response: TypedPromptResponse<Address> = extractor
                .extract(
                    "The address is 123 Main St in Springfield, IL 62701.",
                    &chat_history,
                )
                .await?;
            anyhow::ensure!(
                response.output.street == Some("123 Main St".to_string()),
                "expected street 123 Main St, got {:?}",
                response.output.street
            );
            anyhow::ensure!(
                response.output.city == Some("Springfield".to_string()),
                "expected city Springfield, got {:?}",
                response.output.city
            );
            anyhow::ensure!(
                response.output.state == Some("IL".to_string()),
                "expected state IL, got {:?}",
                response.output.state
            );
            anyhow::ensure!(
                response.output.zip_code == Some("62701".to_string()),
                "expected zip code 62701, got {:?}",
                response.output.zip_code
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
        |client| async move {
            let mut extractor = EcsExtractor::<Person>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let text = "Bob Johnson is a 55 year old retired teacher.";
            let person = extractor.extract(text, &[]).await?.output;
            let response = extractor.extract(text, &[]).await?;
            anyhow::ensure!(
                person.name == Some("Bob Johnson".to_string()),
                "expected extracted name Bob Johnson, got {:?}",
                person.name
            );
            anyhow::ensure!(
                response.output.name == Some("Bob Johnson".to_string()),
                "expected usage response name Bob Johnson, got {:?}",
                response.output.name
            );
            anyhow::ensure!(
                person.age == Some(55),
                "expected extracted age 55, got {:?}",
                person.age
            );
            anyhow::ensure!(
                response.output.age == Some(55),
                "expected usage response age 55, got {:?}",
                response.output.age
            );
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
    with_deepseek_cassette_result(
        "extractor_usage/usage_tracking_works_for_different_schemas",
        |client| async move {
            let mut person_extractor = EcsExtractor::<Person>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let person_response = person_extractor
                .extract("Alice is a 25 year old developer.", &[])
                .await?;
            anyhow::ensure!(
                person_response.usage.total_tokens > 0,
                "expected person usage tokens"
            );
            let mut address_extractor = EcsExtractor::<Address>::new(
                client.completion_model(deepseek::DEEPSEEK_V4_FLASH),
                None,
                None,
            );
            let address_response = address_extractor
                .extract("456 Oak Avenue, Cambridge, MA 02139", &[])
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
