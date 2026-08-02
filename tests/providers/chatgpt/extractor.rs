//! ChatGPT extractor smoke test.

use crate::chatgpt::{LIVE_MODEL, live_agent};
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extractor_smoke() {
    let response = live_agent(LIVE_MODEL)
        .await
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
}
