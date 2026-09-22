//! Native extraction preserving the original provider assertions.
use super::super::support::with_openai_cassette;
use crate::ecs_extractor::EcsExtractor;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};
use rig::providers::openai;
use rig_agent::test_utils::validate_extraction_fields;
#[tokio::test]
async fn extractor_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openai_cassette("extractor/extractor_smoke", |client| async move {
                let mut extractor = EcsExtractor::<SmokePerson>::new(
                    client.openai.completion(openai::GPT_4O),
                    None,
                    None,
                );
                let response = extractor
                    .extract(EXTRACTOR_TEXT, &[])
                    .await
                    .expect("extractor request should succeed");
                validate_extraction_fields(
                    "openai_extractor_smoke",
                    response.output.first_name.as_deref(),
                    response.output.last_name.as_deref(),
                    response.output.job.as_deref(),
                    response.usage,
                )
                .expect("portable extraction contract should hold");
                let first_name = response.output.first_name.as_deref().unwrap_or_default();
                let last_name = response.output.last_name.as_deref().unwrap_or_default();
                let job = response.output.job.as_deref().unwrap_or_default();
                assert_nonempty_response(first_name);
                assert_nonempty_response(last_name);
                assert_nonempty_response(job);
                assert!(
                    response.usage.total_tokens.is_some_and(|n| n > 0),
                    "usage should be populated"
                );
            })
            .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects("openai_extractor_extractor_smoke", log)
        },
    )
    .await
}
