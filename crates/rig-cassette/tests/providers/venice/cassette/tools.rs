//! Venice runs of provider-neutral tool conformance scenarios.

use rig_agent::test_utils::optional_argument;

use super::super::{TOOL_MODEL, support::with_venice_cassette};

#[tokio::test]
async fn tool_with_optional_argument() {
    with_venice_cassette("tools/optional_argument", |client| async move {
        optional_argument(client.completion(TOOL_MODEL), |builder| builder)
            .await
            .expect("optional-argument conformance scenario should succeed");
    })
    .await;
}
