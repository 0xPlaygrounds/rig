//! Native ECS extraction with the original typed assertions.
use super::extractor_usage::{Person, assert_compatible_professions};
use super::support::with_deepseek_cassette_result;
use crate::ecs_extractor::EcsExtractor;
use anyhow::Result;
use rig::providers::deepseek;
#[tokio::test]
async fn extract_backward_compatibility() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_deepseek_cassette_result(
                "extractor_usage/extract_backward_compatibility",
                |client| async move {
                    let mut extractor = EcsExtractor::<Person>::new(
                        client.completion(deepseek::DEEPSEEK_V4_FLASH),
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
                    assert_compatible_professions(
                        person.profession.as_deref(),
                        "software engineer",
                    )?;
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "deepseek_extractor_usage_extract_backward_compatibility",
                log,
            )
        },
    )
    .await
}
