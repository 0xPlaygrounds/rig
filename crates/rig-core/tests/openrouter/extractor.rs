//! OpenRouter extractor smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::openrouter;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn extractor_smoke() {
    let client = openrouter::Client::from_env().expect("client should build");
    let extractor = client.extractor::<SmokePerson>(DEFAULT_MODEL).build();

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
