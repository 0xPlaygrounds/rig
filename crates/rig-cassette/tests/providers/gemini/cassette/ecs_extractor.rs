//! Native extraction preserving the original provider assertions.
use super::extractor::Person;
use crate::ecs_extractor::EcsExtractor;
use rig::providers::gemini;

use crate::support::assert_nonempty_response;
#[tokio::test]
async fn extractor_with_additional_params() {
    rig_test_support::goldens::world_golden_test(
        async {
            let params = serde_json::json!({ "generationConfig": {} });
            super::super::support::with_gemini_cassette(
                "extractor/extractor_with_additional_params",
                |client| async move {
                    let mut extractor = EcsExtractor::<Person>::new(
                        client.completion(gemini::completion::GEMINI_2_5_FLASH),
                        None,
                        Some(params),
                    );
                    let person = extractor
                        .extract("Hello my name is John Doe! I am a software engineer.", &[])
                        .await
                        .expect("extract should succeed")
                        .output;
                    assert_eq!(person.first_name.as_deref(), Some("John"));
                    assert_eq!(person.last_name.as_deref(), Some("Doe"));
                    assert_nonempty_response(person.job.as_deref().unwrap_or_default());
                },
            )
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_extractor_extractor_with_additional_params",
                log,
            )
        },
    )
    .await
}
