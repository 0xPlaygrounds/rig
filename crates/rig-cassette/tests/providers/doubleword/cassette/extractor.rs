//! Cassette-backed Doubleword structured extraction coverage.

use rig::prelude::*;
use rig_agent::test_utils::validate_extraction_fields;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{EXTRACTOR_TEXT, SmokePerson};

#[tokio::test]
async fn extractor_smoke() {
    with_doubleword_cassette("extractor/extractor_smoke", |client| async move {
        let response = client
            .extractor::<SmokePerson>(DEFAULT_MODEL)
            .build()
            .extract(EXTRACTOR_TEXT)
            .await
            .expect("extractor request should succeed");

        validate_extraction_fields(
            "doubleword_extractor_smoke",
            response.output.first_name.as_deref(),
            response.output.last_name.as_deref(),
            response.output.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");
    })
    .await;
}
