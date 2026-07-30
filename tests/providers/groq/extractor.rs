//! Groq extractor smoke test.

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::Runtime;
use std::sync::Arc;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::EXTRACTOR_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn extractor_smoke() {
    let response = extract_with_options::<SmokePerson>(
        AgentConfig::new(),
        super::live(EXTRACTOR_MODEL),
        Arc::new(Runtime::new()),
        EXTRACTOR_TEXT,
        ExtractOptions::classic_extractor(),
    )
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
