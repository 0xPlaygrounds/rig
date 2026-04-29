//! Llamafile streaming tools smoke test.

use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, Message, ToolChoice};
use rig_core::streaming::StreamingPrompt;
use rig_core::tool::Tool;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number,
    assert_raw_stream_contains_distinct_tool_calls_before_text, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_stream_contains_zero_arg_tool_call_named,
    assert_tool_call_precedes_later_text, assert_two_tool_roundtrip_contract,
    collect_raw_stream_observation, collect_stream_final_response, collect_stream_observation,
    zero_arg_tool_definition,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_tools_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = client
        .agent(support::model_name())
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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn example_streaming_with_tools() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = client
        .agent(support::model_name())
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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let model = client.completion_model(support::model_name());
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
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let model = client.completion_model(support::model_name());
    let request = model
        .completion_request(TWO_TOOL_STREAM_PROMPT)
        .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .tool(BetaSignal.definition(String::new()).await)
        .build();

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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_tools_surface_two_distinct_tool_calls_before_final_answer() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = client
        .agent(support::model_name())
        .preamble(TWO_TOOL_STREAM_PREAMBLE)
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
        &["lookup_harbor_label", "lookup_orchard_label"],
        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
    );
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn streaming_tools_emit_tool_call_before_later_text() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = client
        .agent(support::model_name())
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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let model = client.completion_model(support::model_name());
    let request = model
        .completion_request(ORDERED_TOOL_STREAM_PROMPT)
        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
        .tool(AlphaSignal.definition(String::new()).await)
        .build();

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
