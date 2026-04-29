//! Mistral streaming tools coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Message;
use rig_core::providers::mistral;
use rig_core::streaming::{StreamingChat, StreamingPrompt};

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, assert_tool_call_precedes_later_text,
    collect_stream_final_response, collect_stream_observation,
};

use super::TOOL_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn streaming_tools_smoke() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(TOOL_MODEL)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .max_tokens(256)
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
#[ignore = "requires MISTRAL_API_KEY"]
async fn example_streaming_with_tools() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(TOOL_MODEL)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Use the tools provided to answer the user's question and answer in a full sentence.",
        )
        .max_tokens(256)
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
#[ignore = "requires MISTRAL_API_KEY"]
async fn stream_prompt_tool_roundtrip_preserves_streaming_contract() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(TOOL_MODEL)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .max_tokens(256)
        .tool(AlphaSignal)
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
#[ignore = "requires MISTRAL_API_KEY"]
async fn stream_chat_tool_roundtrip_preserves_streaming_contract() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(TOOL_MODEL)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
        .max_tokens(256)
        .tool(AlphaSignal)
        .build();

    let mut stream = agent
        .stream_chat(ORDERED_TOOL_STREAM_PROMPT, Vec::<Message>::new())
        .multi_turn(5)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert_tool_call_precedes_later_text(
        &observation,
        "lookup_harbor_label",
        &[ALPHA_SIGNAL_OUTPUT],
    );
}
