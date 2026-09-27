//! Native extraction preserving the original provider assertions.
use super::support::with_xai_cassette;
use crate::ecs_extractor::EcsExtractor;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};
use rig::providers::xai;
#[tokio::test]
async fn extractor_smoke() {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette("extractor/extractor_smoke", |client| async move {
                let mut extractor = EcsExtractor::<SmokePerson>::new(
                    client.completion(xai::GROK_3_MINI),
                    None,
                    None,
                );
                let response = extractor
                    .extract(EXTRACTOR_TEXT, &[])
                    .await
                    .expect("extractor request should succeed");
                let first_name = response
                    .output
                    .first_name
                    .as_deref()
                    .expect("first_name should be present");
                let last_name = response
                    .output
                    .last_name
                    .as_deref()
                    .expect("last_name should be present");
                let job = response
                    .output
                    .job
                    .as_deref()
                    .expect("job should be present");
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
        |log| rig_test_support::goldens::world_golden_effects("xai_extractor_extractor_smoke", log),
    )
    .await
}
