//! Portable model-contract scenarios recorded through Venice's live API.

use rig_agent::test_utils::{
    streaming_structured_after_tool, structured_extraction, tool_choice_modes,
};

use super::super::{DEFAULT_MODEL, TOOL_MODEL, support::with_venice_cassette};

#[tokio::test]
async fn streaming_structured_output_after_tool() {
    with_venice_cassette(
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
async fn structured_extraction_roundtrip() {
    with_venice_cassette("conformance/structured_extraction", |client| async move {
        structured_extraction(client.completion(DEFAULT_MODEL), None)
            .await
            .expect("structured extraction should succeed");
    })
    .await;
}

#[tokio::test]
async fn tool_choice_modes_roundtrip() {
    with_venice_cassette("conformance/tool_choice_modes", |client| async move {
        tool_choice_modes(client.completion(TOOL_MODEL), |request| request)
            .await
            .expect("tool choice modes should succeed");
    })
    .await;
}
