//! Llamafile extractor smoke test.

use rig_core::client::CompletionClient;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn extractor_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let extractor = client
        .extractor::<SmokePerson>(support::model_name())
        .build();

    let response = extractor
        .extract_with_usage(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    let first_name = response
        .data
        .first_name
        .as_deref()
        .expect("first_name should be present");
    let last_name = response
        .data
        .last_name
        .as_deref()
        .expect("last_name should be present");
    let job = response.data.job.as_deref().expect("job should be present");

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}
