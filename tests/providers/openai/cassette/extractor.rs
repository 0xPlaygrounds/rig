//! OpenAI extractor smoke test.

use rig::prelude::*;
use rig::providers::openai;
use rig_agent::test_utils::validate_extraction_fields;

use super::super::support::with_openai_cassette;
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
async fn extractor_smoke() {
    with_openai_cassette("extractor/extractor_smoke", |client| async move {
        let agent = client.agent(openai::GPT_4O).build();
        let response = agent
            .extractor(EXTRACTOR_TEXT)
            .run_with_usage::<SmokePerson>()
            .await
            .expect("extractor request should succeed");

        validate_extraction_fields(
            "openai_extractor_smoke",
            response.value.first_name.as_deref(),
            response.value.last_name.as_deref(),
            response.value.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");

        let first_name = response.value.first_name.as_deref().unwrap_or_default();
        let last_name = response.value.last_name.as_deref().unwrap_or_default();
        let job = response.value.job.as_deref().unwrap_or_default();

        assert_nonempty_response(first_name);
        assert_nonempty_response(last_name);
        assert_nonempty_response(job);
        assert!(response.usage.total_tokens > 0, "usage should be populated");
    })
    .await;
}
