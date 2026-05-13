//! Gemini streaming tools coverage, including the migrated example path.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, assert_two_tool_roundtrip_contract,
    collect_stream_observation, zero_arg_tool_definition,
};

fn streaming_tool_params() -> serde_json::Value {
    serde_json::to_value(AdditionalParameters::default().with_config(GenerationConfig::default()))
        .expect("Gemini additional params should serialize")
}

#[tokio::test]
async fn streaming_tools_smoke() {
    let (cassette, client) =
        super::super::support::gemini_cassette("streaming_tools/streaming_tools_smoke").await;
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent
        .stream_prompt(STREAMING_TOOLS_PROMPT)
        .multi_turn(3)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    cassette.finish().await;
}

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let (cassette, client) = super::super::support::gemini_cassette(
        "streaming_tools/raw_stream_emits_required_zero_arg_tool_call",
    )
    .await;
    let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .additional_params(streaming_tool_params())
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;

    cassette.finish().await;
}

#[tokio::test]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    let (cassette, client) = super::super::support::gemini_cassette(
        "streaming_tools/streaming_tools_surface_two_distinct_tool_calls_before_final_answer",
    )
    .await;
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .tool(BetaSignal)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent
        .stream_prompt(TWO_TOOL_STREAM_PROMPT)
        .multi_turn(8)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_two_tool_roundtrip_contract(
        &observation,
        &["lookup_harbor_label", "lookup_orchard_label"],
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );

    cassette.finish().await;
}

#[tokio::test]
async fn streaming_tools_emit_tool_call_before_later_text() {
    let (cassette, client) = super::super::support::gemini_cassette(
        "streaming_tools/streaming_tools_emit_tool_call_before_later_text",
    )
    .await;
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .tool(AlphaSignal)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent
        .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );

    cassette.finish().await;
}

#[tokio::test]
async fn example_streaming_with_tools() {
    let (cassette, client) =
        super::super::support::gemini_cassette("streaming_tools/example_streaming_with_tools")
            .await;
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent
        .stream_prompt(STREAMING_TOOLS_PROMPT)
        .multi_turn(3)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    cassette.finish().await;
}
