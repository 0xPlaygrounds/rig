//! Native extraction preserving the original provider assertions.
use crate::copilot::{LIVE_MODEL, with_copilot_cassette};
use crate::ecs_extractor::EcsExtractor;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};
use rig::prelude::*;
#[tokio::test]
async fn extractor_smoke() {
    with_copilot_cassette("extractor/extractor_smoke", |client| async move {
        let mut extractor =
            EcsExtractor::<SmokePerson>::new(client.completion_model(LIVE_MODEL), None, None);
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
        assert!(response.usage.total_tokens > 0, "usage should be populated");
    })
    .await;
}
