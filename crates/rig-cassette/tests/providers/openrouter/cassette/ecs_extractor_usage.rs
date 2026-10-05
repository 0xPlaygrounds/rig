//! Native ECS extraction with the original typed assertions.
use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette_result};
use super::extractor_usage::{Person, assert_compatible_professions};
use crate::ecs_extractor::{EcsExtractor, Extracted as TypedPromptResponse};
use anyhow::Result;
#[tokio::test]
async fn extract_with_usage_returns_data_and_usage() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openrouter_cassette_result(
                "extractor_usage/extract_with_usage_returns_data_and_usage",
                |client| async move {
                    let mut extractor =
                        EcsExtractor::<Person>::new(client.completion(DEFAULT_MODEL), None, None);
                    let response: TypedPromptResponse<Person> = extractor
                        .extract("Jane Smith is a 45 year old data scientist.", &[])
                        .await?;
                    anyhow::ensure!(response.output.name.as_deref() == Some("Jane Smith"));
                    anyhow::ensure!(response.output.age == Some(45));
                    assert_compatible_professions(
                        response.output.profession.as_deref(),
                        "data scientist",
                    )?;
                    anyhow::ensure!(response.usage.input_tokens.is_some_and(|n| n > 0));
                    anyhow::ensure!(response.usage.output_tokens.is_some_and(|n| n > 0));
                    anyhow::ensure!(response.usage.total_tokens.is_some_and(|n| n > 0));
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openrouter_extractor_usage_extract_with_usage_returns_data_and_usage",
                log,
            )
        },
    )
    .await
}
