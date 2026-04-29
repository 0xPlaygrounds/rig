//! OpenAI streaming tools coverage, including the migrated example path.

use rig_core::OneOrMany;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, Message};
use rig_core::providers::openai;
use rig_core::streaming::StreamingPrompt;
use rig_core::tool::Tool;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_tool_call_precedes_later_text,
    collect_raw_stream_observation, collect_stream_final_response, collect_stream_observation,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn streaming_tools_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
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
#[ignore = "requires OPENAI_API_KEY"]
async fn example_streaming_with_tools() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
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
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_stream_preserves_tool_result_flow() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
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
#[ignore = "requires OPENAI_API_KEY"]
async fn raw_responses_stream_preserves_tool_then_followup_text_ordering() {
    let client = openai::Client::from_env().expect("client should build");
    let model = client.completion_model(openai::GPT_4O);
    let request = model
        .completion_request(ORDERED_TOOL_STREAM_PROMPT)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .build();

    let first_turn = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("raw responses stream should start"),
    )
    .await;

    assert_raw_stream_tool_call_precedes_text(&first_turn, "lookup_harbor_label");

    let tool_call = first_turn
        .tool_calls
        .iter()
        .find(|tool_call| tool_call.function.name == "lookup_harbor_label")
        .cloned()
        .expect("raw responses stream should yield lookup_harbor_label");
    let assistant_message = Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
    };
    let tool_result_message =
        Message::tool_result_with_call_id(tool_call.id, tool_call.call_id, ALPHA_SIGNAL_OUTPUT);
    let followup_request = model
        .completion_request(
            "Now reply in one short sentence using the provided tool result. Do not call any tools.",
        )
        .preamble("Use the provided tool result and answer directly.".to_string())
        .message(assistant_message)
        .message(tool_result_message)
        .build();

    let second_turn = collect_raw_stream_observation(
        model
            .stream(followup_request)
            .await
            .expect("raw followup responses stream should start"),
    )
    .await;

    assert!(
        second_turn.tool_calls.is_empty(),
        "follow-up raw responses stream should not emit fresh tool calls, saw {:?}",
        second_turn
            .tool_calls
            .iter()
            .map(|tool_call| tool_call.function.name.as_str())
            .collect::<Vec<_>>()
    );
    assert_raw_stream_text_contains(&second_turn, &[ALPHA_SIGNAL_OUTPUT]);
}
