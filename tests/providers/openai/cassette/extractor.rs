//! OpenAI extractor smoke test.

use rig::client::CompletionClient;
use rig::providers::openai;
use rig_agent::test_utils::validate_extraction_fields;

use super::super::support::with_openai_cassette;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
async fn extractor_smoke() {
    with_openai_cassette("extractor/extractor_smoke", |client| async move {
        let extractor = client.extractor::<SmokePerson>(openai::GPT_4O).build();

        let response = extractor
            .extract_with_usage(EXTRACTOR_TEXT)
            .await
            .expect("extractor request should succeed");

        validate_extraction_fields(
            "openai_extractor_smoke",
            response.data.first_name.as_deref(),
            response.data.last_name.as_deref(),
            response.data.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");

        let first_name = response.data.first_name.as_deref().unwrap_or_default();
        let last_name = response.data.last_name.as_deref().unwrap_or_default();
        let job = response.data.job.as_deref().unwrap_or_default();

        assert_nonempty_response(first_name);
        assert_nonempty_response(last_name);
        assert_nonempty_response(job);
        assert!(response.usage.total_tokens > 0, "usage should be populated");
    })
    .await;
}
