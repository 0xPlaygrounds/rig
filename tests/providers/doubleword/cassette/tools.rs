//! Doubleword runs of provider-neutral tool conformance scenarios.

use rig::client::CompletionClient;
use rig_agent::test_utils::{optional_argument, sequential_tools};

use super::super::{DEFAULT_MODEL, TOOL_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn tool_with_optional_argument() {
    with_doubleword_cassette("tools/optional_argument", |client| async move {
        optional_argument(client.completion_model(DEFAULT_MODEL), |builder| builder)
            .await
            .expect("optional-argument conformance scenario should succeed");
    })
    .await;
}

#[tokio::test]
async fn two_tools_nonstreaming_chain() {
    with_doubleword_cassette("tools/two_tools_nonstreaming", |client| async move {
        sequential_tools(client.completion_model(TOOL_MODEL), |builder| builder)
            .await
            .expect("sequential-tool conformance scenario should succeed");
    })
    .await;
}
