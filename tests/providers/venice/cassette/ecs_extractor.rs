//! Native extraction preserving the original provider assertions.
use super::super::{DEFAULT_MODEL, support::with_venice_cassette};
use crate::ecs_extractor::EcsExtractor;
use crate::support::{EXTRACTOR_TEXT, SmokePerson};
use rig::prelude::*;
use rig_agent::test_utils::validate_extraction_fields;
#[tokio::test]
async fn extractor_smoke() {
    with_venice_cassette("extractor/extractor_smoke", |client| async move {
        let response =
            EcsExtractor::<SmokePerson>::new(client.completion_model(DEFAULT_MODEL), None, None)
                .extract(EXTRACTOR_TEXT, &[])
                .await
                .expect("extractor request should succeed");
        validate_extraction_fields(
            "venice_extractor_smoke",
            response.output.first_name.as_deref(),
            response.output.last_name.as_deref(),
            response.output.job.as_deref(),
            response.usage,
        )
        .expect("portable extraction contract should hold");
    })
    .await;
}
