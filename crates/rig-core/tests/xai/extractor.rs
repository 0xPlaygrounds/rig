//! xAI extractor smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::xai;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn extractor_smoke() {
    let client = xai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<SmokePerson>(xai::GROK_3_MINI).build();

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
