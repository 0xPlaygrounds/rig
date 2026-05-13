//! OpenAI extractor smoke test.

use rig::providers::openai;

use crate::cassettes::ProviderCassette;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
async fn extractor_smoke() {
    let cassette = ProviderCassette::start(
        "openai",
        "extractor/extractor_smoke",
        "https://api.openai.com/v1",
    )
    .await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
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
    cassette.finish().await;
}
