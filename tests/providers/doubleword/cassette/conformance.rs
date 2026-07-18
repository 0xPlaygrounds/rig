//! Portable model-contract scenarios recorded through Doubleword's live API.

use rig::client::CompletionClient;
use rig_agent::test_utils::{
    cancellation_and_max_turns, hook_rewrites_and_request_patch, invalid_tool_recovery,
    parallel_tools, streaming_structured_after_tool, streaming_tool, structured_after_tool,
    structured_extraction, tool_choice_modes, tool_output_serialization, zero_argument_tool,
};

use super::super::{DEFAULT_MODEL, TOOL_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn zero_argument_tool_roundtrip() {
    with_doubleword_cassette("conformance/zero_argument_tool", |client| async move {
        zero_argument_tool(client.completion_model(TOOL_MODEL), |builder| builder)
            .await
            .expect("zero-argument tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn parallel_tool_calls_roundtrip() {
    with_doubleword_cassette("conformance/parallel_tools", |client| async move {
        parallel_tools(client.completion_model(TOOL_MODEL), |builder| builder, None)
            .await
            .expect("parallel tool calls should succeed");
    })
    .await;
}

#[tokio::test]
async fn cancellation_and_max_turn_diagnostics() {
    with_doubleword_cassette(
        "conformance/cancellation_and_max_turns",
        |client| async move {
            cancellation_and_max_turns(client.completion_model(TOOL_MODEL), |builder| builder)
                .await
                .expect("cancellation and max-turn diagnostics should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn tool_output_types_roundtrip() {
    with_doubleword_cassette(
        "conformance/tool_output_serialization",
        |client| async move {
            tool_output_serialization(client.completion_model(TOOL_MODEL), |builder| builder)
                .await
                .expect("tool output serialization should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn invalid_tool_call_recovers() {
    with_doubleword_cassette("conformance/invalid_tool_recovery", |client| async move {
        invalid_tool_recovery(client.completion_model(TOOL_MODEL), |builder| builder)
            .await
            .expect("invalid tool call recovery should succeed");
    })
    .await;
}

#[tokio::test]
async fn hooks_rewrite_tool_flow() {
    with_doubleword_cassette(
        "conformance/hook_rewrites_and_request_patch",
        |client| async move {
            hook_rewrites_and_request_patch(client.completion_model(TOOL_MODEL), |builder| builder)
                .await
                .expect("hook rewrite scenario should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_tool_roundtrip() {
    with_doubleword_cassette("conformance/streaming_tool", |client| async move {
        streaming_tool(client.completion_model(TOOL_MODEL), |builder| builder)
            .await
            .expect("streaming tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn structured_output_after_tool() {
    with_doubleword_cassette("conformance/structured_after_tool", |client| async move {
        structured_after_tool(client.completion_model(TOOL_MODEL), |builder| builder)
            .await
            .expect("structured output after tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn streaming_structured_output_after_tool() {
    with_doubleword_cassette(
        "conformance/streaming_structured_after_tool",
        |client| async move {
            streaming_structured_after_tool(client.completion_model(TOOL_MODEL), |builder| builder)
                .await
                .expect("streaming structured output after tool should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn structured_extraction_roundtrip() {
    with_doubleword_cassette("conformance/structured_extraction", |client| async move {
        structured_extraction(client.completion_model(DEFAULT_MODEL))
            .await
            .expect("structured extraction should succeed");
    })
    .await;
}

#[tokio::test]
async fn tool_choice_modes_roundtrip() {
    with_doubleword_cassette("conformance/tool_choice_modes", |client| async move {
        tool_choice_modes(client.completion_model(TOOL_MODEL))
            .await
            .expect("tool choice modes should succeed");
    })
    .await;
}
