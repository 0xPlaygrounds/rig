//! OpenAI extractor smoke test.

use rig::client::ProviderClient;
use rig::providers::openai;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn extractor_smoke() {
    let client = openai::Client::from_env();
    let extractor = client.extractor::<SmokePerson>(openai::GPT_4O).build();

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
