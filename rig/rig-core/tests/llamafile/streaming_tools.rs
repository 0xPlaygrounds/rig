//! Llamafile streaming tools smoke test.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::llamafile;
use rig::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, assert_two_tool_roundtrip_contract,
    collect_stream_observation, zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = llamafile::Client::from_url("http://localhost:8080");
    let model = client.completion_model(llamafile::LLAMA_CPP);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    let client = llamafile::Client::from_url("http://localhost:8080");
    let agent = client
        .agent(llamafile::LLAMA_CPP)
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
        .tool_choice(ToolChoice::Required)
        .tool(AlphaSignal)
        .tool(BetaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(TWO_TOOL_STREAM_PROMPT)
        .multi_turn(8)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_two_tool_roundtrip_contract(
        &observation,
        &["alpha_signal", "beta_signal"],
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_tools_emit_tool_call_before_later_text() {
    let client = llamafile::Client::from_url("http://localhost:8080");
    let agent = client
        .agent(llamafile::LLAMA_CPP)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .tool_choice(ToolChoice::Required)
        .tool(AlphaSignal)
        .build();

    let mut stream = agent
        .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(&observation, "alpha_signal", &[ALPHA_SIGNAL_OUTPUT]);
}
