//! Portable model-contract scenarios recorded through Doubleword's live API.

use rig_agent::test_utils::{
    ScenarioOverrides, cancellation_and_max_turns, hook_rewrites_and_request_patch,
    invalid_tool_recovery, invalid_tool_recovery_session, parallel_tools, parallel_tools_session,
    streaming_structured_after_tool, streaming_tool, streaming_tool_session, structured_after_tool,
    structured_extraction, tool_choice_modes, tool_output_serialization,
    tool_output_serialization_session, zero_argument_tool, zero_argument_tool_session,
};

use super::super::{DEFAULT_MODEL, TOOL_MODEL, support::with_doubleword_cassette};

#[tokio::test]
async fn zero_argument_tool_roundtrip() {
    with_doubleword_cassette("conformance/zero_argument_tool", |env| async move {
        zero_argument_tool(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("zero-argument tool should succeed");
    })
    .await;
}

/// Session-driver twin replaying the SAME cassette as
/// [`zero_argument_tool_roundtrip`]: proves the session driver issues
/// byte-identical requests to the classic agent.
#[tokio::test]
async fn zero_argument_tool_roundtrip_session() {
    with_doubleword_cassette("conformance/zero_argument_tool", |env| async move {
        zero_argument_tool_session(env.provider(TOOL_MODEL), ScenarioOverrides::new())
            .await
            .expect("session zero-argument tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn parallel_tool_calls_roundtrip() {
    with_doubleword_cassette("conformance/parallel_tools", |env| async move {
        parallel_tools(env.provider(TOOL_MODEL), |builder| builder, None)
            .await
            .expect("parallel tool calls should succeed");
    })
    .await;
}

/// Session-driver twin replaying the SAME cassette as
/// [`parallel_tool_calls_roundtrip`].
#[tokio::test]
async fn parallel_tool_calls_roundtrip_session() {
    with_doubleword_cassette("conformance/parallel_tools", |env| async move {
        parallel_tools_session(env.provider(TOOL_MODEL), ScenarioOverrides::new(), None)
            .await
            .expect("session parallel tool calls should succeed");
    })
    .await;
}

#[tokio::test]
async fn cancellation_and_max_turn_diagnostics() {
    with_doubleword_cassette("conformance/cancellation_and_max_turns", |env| async move {
        cancellation_and_max_turns(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("cancellation and max-turn diagnostics should succeed");
    })
    .await;
}

#[tokio::test]
async fn tool_output_types_roundtrip() {
    with_doubleword_cassette("conformance/tool_output_serialization", |env| async move {
        tool_output_serialization(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("tool output serialization should succeed");
    })
    .await;
}

/// Session-driver twin replaying the SAME cassette as
/// [`tool_output_types_roundtrip`].
#[tokio::test]
async fn tool_output_types_roundtrip_session() {
    with_doubleword_cassette("conformance/tool_output_serialization", |env| async move {
        tool_output_serialization_session(env.provider(TOOL_MODEL), ScenarioOverrides::new())
            .await
            .expect("session tool output serialization should succeed");
    })
    .await;
}

#[tokio::test]
async fn invalid_tool_call_recovers() {
    with_doubleword_cassette("conformance/invalid_tool_recovery", |env| async move {
        invalid_tool_recovery(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("invalid tool call recovery should succeed");
    })
    .await;
}

/// Session-driver twin replaying the SAME cassette as
/// [`invalid_tool_call_recovers`].
#[tokio::test]
async fn invalid_tool_call_recovers_session() {
    with_doubleword_cassette("conformance/invalid_tool_recovery", |env| async move {
        invalid_tool_recovery_session(env.provider(TOOL_MODEL), ScenarioOverrides::new())
            .await
            .expect("session invalid tool call recovery should succeed");
    })
    .await;
}

#[tokio::test]
async fn hooks_rewrite_tool_flow() {
    with_doubleword_cassette(
        "conformance/hook_rewrites_and_request_patch",
        |env| async move {
            hook_rewrites_and_request_patch(env.provider(TOOL_MODEL), |builder| builder)
                .await
                .expect("hook rewrite scenario should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_tool_roundtrip() {
    with_doubleword_cassette("conformance/streaming_tool", |env| async move {
        streaming_tool(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("streaming tool should succeed");
    })
    .await;
}

/// Session-driver twin replaying the SAME cassette as
/// [`streaming_tool_roundtrip`]: streaming parity through
/// `AgentStream::next_item_with_tools`.
#[tokio::test]
async fn streaming_tool_roundtrip_session() {
    with_doubleword_cassette("conformance/streaming_tool", |env| async move {
        streaming_tool_session(env.provider(TOOL_MODEL), ScenarioOverrides::new())
            .await
            .expect("session streaming tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn structured_output_after_tool() {
    with_doubleword_cassette("conformance/structured_after_tool", |env| async move {
        structured_after_tool(env.provider(TOOL_MODEL), |builder| builder)
            .await
            .expect("structured output after tool should succeed");
    })
    .await;
}

#[tokio::test]
async fn streaming_structured_output_after_tool() {
    with_doubleword_cassette(
        "conformance/streaming_structured_after_tool",
        |env| async move {
            streaming_structured_after_tool(env.provider(TOOL_MODEL), |builder| builder)
                .await
                .expect("streaming structured output after tool should succeed");
        },
    )
    .await;
}

#[tokio::test]
async fn structured_extraction_roundtrip() {
    with_doubleword_cassette("conformance/structured_extraction", |env| async move {
        structured_extraction(env.provider(DEFAULT_MODEL))
            .await
            .expect("structured extraction should succeed");
    })
    .await;
}

#[tokio::test]
async fn tool_choice_modes_roundtrip() {
    with_doubleword_cassette("conformance/tool_choice_modes", |env| async move {
        tool_choice_modes(
            env.provider(TOOL_MODEL),
            std::sync::Arc::new(rig_agent::provider::Runtime::new()),
        )
        .await
        .expect("tool choice modes should succeed");
    })
    .await;
}
