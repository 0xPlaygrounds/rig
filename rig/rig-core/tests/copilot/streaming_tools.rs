//! Copilot streaming tools coverage, including the migrated example path.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::streaming::StreamingPrompt;

use crate::copilot::{LIVE_MODEL, live_client};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    REQUIRED_ZERO_ARG_TOOL_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number,
    assert_stream_contains_zero_arg_tool_call_named, assert_two_tool_roundtrip_contract,
    collect_stream_final_response, collect_stream_observation, zero_arg_tool_definition,
};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming_tools_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn example_streaming_with_tools() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tools prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let model = live_client().completion_model(LIVE_MODEL);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    let agent = live_client()
        .agent(LIVE_MODEL)
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
