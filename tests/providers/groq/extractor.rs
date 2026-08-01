//! Groq extractor smoke test.

use rig::prelude::*;
use rig::providers::groq;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::EXTRACTOR_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn extractor_smoke() {
    let response = groq::Client::from_env()
        .expect("Groq client should build from env")
        .agent(EXTRACTOR_MODEL)
        .build()
        .extractor(EXTRACTOR_TEXT)
        .classic()
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
