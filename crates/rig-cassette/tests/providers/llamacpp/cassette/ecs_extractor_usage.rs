//! Native ECS extraction with the original typed assertions.
//!
//! | Cell | Contract |
//! | --- | --- |
//! | `extract_backward_compatibility` | Extracted person fields match the original assertions. |
use super::super::cassette_support::*;
use super::extractor_usage::{EXTRACTOR_PREAMBLE, Person};
use crate::ecs_extractor::EcsExtractor;
use anyhow::Result;
use serde_json::json;
#[tokio::test]
async fn extract_backward_compatibility() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_llamacpp_cassette_result(
                "extractor_usage/extract_backward_compatibility",
                |client| async move {
                    let model = CASSETTE_MODEL;
                    let mut extractor = EcsExtractor::<Person>::new(
                        client.completion(model),
                        Some(EXTRACTOR_PREAMBLE),
                        Some(json!({ "temperature" : 0.0 })),
                    );
                    let person = extractor
                        .extract("John Doe is a 30 year old software engineer.", &[])
                        .await?
                        .output;
                    anyhow::ensure!(person.name.as_deref() == Some("John Doe"));
                    anyhow::ensure!(person.age == Some(30));
                    anyhow::ensure!(person.profession.as_deref() == Some("software engineer"));
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "llamacpp_extractor_usage_extract_backward_compatibility",
                log,
            )
        },
    )
    .await
}
