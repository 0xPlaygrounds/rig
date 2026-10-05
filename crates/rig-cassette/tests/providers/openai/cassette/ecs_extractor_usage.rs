//! Native ECS extraction with the original typed assertions.
use super::super::support::with_openai_cassette_result;
use super::extractor_usage::{Address, Person};
use crate::ecs_extractor::EcsExtractor;
use anyhow::Result;
use rig::providers;
/// Test that usage is reported for both simple and complex extraction scenarios.
#[tokio::test]
async fn usage_tracking_works_for_different_schemas() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openai_cassette_result(
                "extractor_usage/usage_tracking_works_for_different_schemas",
                |client| async move {
                    let mut person_extractor = EcsExtractor::<Person>::new(
                        client.openai.completion(providers::openai::GPT_4O_MINI),
                        None,
                        None,
                    );
                    let person_response = person_extractor
                        .extract("Alice is a 25 year old developer.", &[])
                        .await?;
                    anyhow::ensure!(person_response.usage.total_tokens.is_some_and(|n| n > 0));
                    let mut address_extractor = person_extractor.with_schema::<Address>();
                    let address_response = address_extractor
                        .extract("456 Oak Avenue, Cambridge, MA 02139", &[])
                        .await?;
                    anyhow::ensure!(address_response.usage.total_tokens.is_some_and(|n| n > 0));
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openai_extractor_usage_usage_tracking_works_for_different_schemas",
                log,
            )
        },
    )
    .await
}
