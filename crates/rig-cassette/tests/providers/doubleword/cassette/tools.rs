//! Doubleword runs of provider-neutral tool conformance scenarios.

use rig_agent::test_utils::optional_argument;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn tool_with_optional_argument() {
    with_doubleword_cassette("tools/optional_argument", |client| async move {
        optional_argument(client.completion(DEFAULT_MODEL), |builder| builder)
            .await
            .expect("optional-argument conformance scenario should succeed");
    })
    .await;
}
