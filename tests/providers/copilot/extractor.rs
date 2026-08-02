//! Copilot extractor smoke test.

use rig::prelude::*;

use crate::copilot::{LIVE_MODEL, with_copilot_cassette};
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
async fn extractor_smoke() {
    with_copilot_cassette("extractor/extractor_smoke", |client| async move {
        let response = client
            .agent(LIVE_MODEL)
            .build()
            .extractor(EXTRACTOR_TEXT)
            .run_with_usage::<SmokePerson>()
            .await
            .expect("extractor request should succeed");

        let first_name = response
            .value
            .first_name
            .as_deref()
            .expect("first_name should be present");
        let last_name = response
            .value
            .last_name
            .as_deref()
            .expect("last_name should be present");
        let job = response
            .value
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
