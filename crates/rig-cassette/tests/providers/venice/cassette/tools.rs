//! Venice runs of provider-neutral tool conformance scenarios.

use rig_agent::test_utils::{optional_argument, sequential_tools};

use super::super::{TOOL_MODEL, support::with_venice_cassette};

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "venice/tools/optional_argument"
))]
#[tokio::test]
async fn tool_with_optional_argument() {
    with_venice_cassette("tools/optional_argument", |client| async move {
        optional_argument(client.completion(TOOL_MODEL), |builder| builder)
            .await
            .expect("optional-argument conformance scenario should succeed");
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "venice/tools/two_tools_nonstreaming"
))]
#[tokio::test]
async fn two_tools_nonstreaming_chain() {
    with_venice_cassette("tools/two_tools_nonstreaming", |client| async move {
        sequential_tools(client.completion(TOOL_MODEL), |builder| builder)
            .await
            .expect("sequential-tool conformance scenario should succeed");
    })
    .await;
}
