//! Gemini streaming tools coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::ToolChoice;
use rig_core::providers::gemini;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number,
    assert_stream_contains_zero_arg_tool_call_named, assert_tool_call_precedes_later_text,
    assert_two_tool_roundtrip_contract, collect_stream_final_response, collect_stream_observation,
    zero_arg_tool_definition,
};

fn streaming_tool_params() -> serde_json::Value {
    serde_json::to_value(AdditionalParameters::default().with_config(GenerationConfig::default()))
        .expect("Gemini additional params should serialize")
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tools_smoke() {
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    let client = gemini::Client::from_env().expect("client should build");
    let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
    let request = model
        .completion_request(REQUIRED_ZERO_ARG_TOOL_PROMPT)
        .tool(zero_arg_tool_definition("ping"))
        .tool_choice(ToolChoice::Required)
        .additional_params(streaming_tool_params())
        .build();
    let stream = model.stream(request).await.expect("stream should start");

    assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    let client = gemini::Client::from_env().expect("client should build");
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
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tools_emit_tool_call_before_later_text() {
    let client = gemini::Client::from_env().expect("client should build");
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
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn example_streaming_with_tools() {
    let agent = gemini::Client::from_env()
        .expect("client should build")
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question.",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .additional_params(streaming_tool_params())
        .build();

    let mut stream = agent.stream_prompt("Calculate 2 - 5").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
