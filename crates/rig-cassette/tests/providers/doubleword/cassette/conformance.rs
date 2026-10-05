//! Portable model-contract scenarios recorded through Doubleword's live API.

use rig_agent::test_utils::{
    invalid_tool_recovery, streaming_structured_after_tool, tool_choice_modes, zero_argument_tool,
};

use super::super::{TOOL_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn zero_argument_tool_roundtrip() {
    with_doubleword_cassette("conformance/zero_argument_tool", |client| async move {
        zero_argument_tool(client.completion(TOOL_MODEL), |builder| builder)
            .await
            .expect("zero-argument tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn invalid_tool_call_recovers() {
    with_doubleword_cassette("conformance/invalid_tool_recovery", |client| async move {
        invalid_tool_recovery(client.completion(TOOL_MODEL), |builder| builder)
            .await
            .expect("invalid tool call recovery should succeed");
    })
    .await;
}

#[tokio::test]
async fn streaming_structured_output_after_tool() {
    with_doubleword_cassette(
        "conformance/streaming_structured_after_tool",
        |client| async move {
            streaming_structured_after_tool(client.completion(TOOL_MODEL), |builder| builder)
                .await
                .expect("streaming structured output after tool should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn tool_choice_modes_roundtrip() {
    with_doubleword_cassette("conformance/tool_choice_modes", |client| async move {
        tool_choice_modes(client.completion(TOOL_MODEL), |request| request)
            .await
            .expect("tool choice modes should succeed");
    })
    .await;
}
