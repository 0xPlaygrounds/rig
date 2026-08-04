//! Llamafile extractor smoke test.

use rig::prelude::*;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn extractor_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let response = support::client()
        .agent(&support::model_name())
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
