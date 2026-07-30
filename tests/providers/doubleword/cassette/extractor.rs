//! Cassette-backed Doubleword structured extraction coverage.

use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::Runtime;
use rig_agent::test_utils::validate_extraction_fields;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{EXTRACTOR_TEXT, SmokePerson};

#[tokio::test]
async fn extractor_smoke() {
    with_doubleword_cassette("extractor/extractor_smoke", |env| async move {
        let response = extract_with_options::<SmokePerson>(
            AgentConfig::new(),
            env.provider(DEFAULT_MODEL),
            Arc::new(Runtime::new()),
            EXTRACTOR_TEXT,
            ExtractOptions::classic_extractor(),
        )
        .await
        .expect("extractor request should succeed");

        validate_extraction_fields(
            "doubleword_extractor_smoke",
            response.value.first_name.as_deref(),
            response.value.last_name.as_deref(),
            response.value.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");
    })
    .await;
}
