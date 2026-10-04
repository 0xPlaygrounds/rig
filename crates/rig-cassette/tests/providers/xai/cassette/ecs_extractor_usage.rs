//! Native ECS extraction with the original typed assertions.
use super::extractor_usage::{Address, Person, assert_compatible_professions};
use super::support::with_xai_cassette_result;
use crate::ecs_extractor::{EcsExtractor, Extracted as TypedPromptResponse};
use anyhow::Result;
use rig::message::Message;
use rig::providers::xai;
#[tokio::test]
async fn extract_with_chat_history_with_usage_works() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette_result(
                "extractor_usage/extract_with_chat_history_with_usage_works",
                |client| async move {
                    let mut extractor = EcsExtractor::<Address>::new(
                        client.completion(xai::GROK_3_MINI),
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
                    anyhow::ensure!(response.output.street.as_deref() == Some("123 Main St"));
                    anyhow::ensure!(response.output.city.as_deref() == Some("Springfield"));
                    anyhow::ensure!(response.output.state.as_deref() == Some("IL"));
                    anyhow::ensure!(response.output.zip_code.as_deref() == Some("62701"));
                    anyhow::ensure!(response.usage.input_tokens.is_some_and(|n| n > 0));
                    anyhow::ensure!(response.usage.total_tokens.is_some_and(|n| n > 0));
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "xai_extractor_usage_extract_with_chat_history_with_usage_works",
                log,
            )
        },
    )
    .await
}
#[tokio::test]
async fn extract_and_extract_with_usage_return_same_data() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette_result(
                "extractor_usage/extract_and_extract_with_usage_return_same_data",
                |client| async move {
                    let mut extractor = EcsExtractor::<Person>::new(
                        client.completion(xai::GROK_3_MINI),
                        None,
                        None,
                    );
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
                    anyhow::ensure!(
                        response.usage.total_tokens.is_some_and(|n| n > 0),
                        "usage should be populated"
                    );
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "xai_extractor_usage_extract_and_extract_with_usage_return_same_data",
                log,
            )
        },
    )
    .await
}
