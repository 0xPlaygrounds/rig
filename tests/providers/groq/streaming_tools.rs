//! Groq streaming tools smoke test.

use rig::OneOrMany;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{AssistantContent, Message, ToolChoice};
use rig::prelude::*;
use rig::providers::groq;
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_raw_stream_contains_distinct_tool_calls_before_text,
    assert_raw_stream_text_contains, assert_raw_stream_tool_call_precedes_text,
    assert_stream_contains_zero_arg_tool_call_named, assert_tool_call_precedes_later_text,
    assert_two_tool_roundtrip_contract, collect_raw_stream_observation, collect_stream_observation,
    zero_arg_tool_definition,
};

use super::{
    STREAMING_TOOLS_MULTI_MODEL, STREAMING_TOOLS_ORDERED_MODEL, STREAMING_TOOLS_RAW_MODEL,
};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = groq::Client::from_env().expect("client should build");
    let model = client.completion_model(STREAMING_TOOLS_RAW_MODEL);
    let request = CompletionRequest {
        tools: vec![zero_arg_tool_definition("ping")],
        tool_choice: Some(ToolChoice::Required),
        ..CompletionRequest::from_prompt(REQUIRED_ZERO_ARG_TOOL_PROMPT)
    };
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    let client = groq::Client::from_env().expect("client should build");
    let model = client.completion_model(STREAMING_TOOLS_RAW_MODEL);
    let request = CompletionRequest {
        tools: vec![
            rig::tool::portable_tool_definition(&AlphaSignal),
            rig::tool::portable_tool_definition(&BetaSignal),
        ],
        ..CompletionRequest::with_history(
            Some(TWO_TOOL_STREAM_PREAMBLE),
            Vec::new(),
            TWO_TOOL_STREAM_PROMPT,
        )
    };

    let observation = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw stream should start"),
    )
    .await;

    assert_raw_stream_contains_distinct_tool_calls_before_text(
        &observation,
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    let client = groq::Client::from_env().expect("client should build");
    let agent = client
        .agent(STREAMING_TOOLS_MULTI_MODEL)
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .tool(BetaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(TWO_TOOL_STREAM_PROMPT)
        .max_turns(8)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_two_tool_roundtrip_contract(
        &observation,
        &["lookup_harbor_label", "lookup_orchard_label"],
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_tools_emit_tool_call_before_later_text() {
    let client = groq::Client::from_env().expect("client should build");
    let agent = client
        .agent(STREAMING_TOOLS_ORDERED_MODEL)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
        .max_turns(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    let client = groq::Client::from_env().expect("client should build");
    let model = client.completion_model(STREAMING_TOOLS_RAW_MODEL);
    let request = CompletionRequest {
        tools: vec![rig::tool::portable_tool_definition(&AlphaSignal)],
        ..CompletionRequest::with_history(
            Some(ORDERED_TOOL_STREAM_PREAMBLE),
            Vec::new(),
            ORDERED_TOOL_STREAM_PROMPT,
        )
    };

    let first_turn = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw stream should start"),
    )
    .await;

    assert_raw_stream_tool_call_precedes_text(&first_turn, "lookup_harbor_label");

    let tool_call = first_turn
        .tool_calls
        .iter()
        .find(|tool_call| tool_call.function.name == "lookup_harbor_label")
        .cloned()
        .expect("raw stream should yield lookup_harbor_label");
    let assistant_message = Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
    };
    let tool_result_message =
        Message::tool_result_with_call_id(tool_call.id, tool_call.call_id, ALPHA_SIGNAL_OUTPUT);
    let followup_request = CompletionRequest::with_history(
        Some("Use the provided tool result and answer directly."),
        vec![assistant_message, tool_result_message],
        "Now reply in one short sentence using the provided tool result. Do not call any tools.",
    );

    let second_turn = collect_raw_stream_observation(
        model
            .stream(followup_request)
            .await
            .expect("raw followup stream should start"),
    )
    .await;

    assert!(
        second_turn.tool_calls.is_empty(),
        "follow-up raw stream should not emit fresh tool calls, saw {:?}",
        second_turn
            .tool_calls
            .iter()
            .map(|tool_call| tool_call.function.name.as_str())
            .collect::<Vec<_>>()
    );
    assert_raw_stream_text_contains(&second_turn, &[ALPHA_SIGNAL_OUTPUT]);
}
