//! Copilot streaming tools coverage, including the migrated example path.
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};

use crate::copilot::{LIVE_MODEL, with_copilot_cassette};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, Subtract,
    assert_mentions_expected_number, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_precedes_text, assert_stream_contains_zero_arg_tool_call_named,
    collect_raw_stream_observation, collect_stream_final_response, zero_arg_tool_definition,
};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn example_streaming_with_tools() {
    with_copilot_cassette("streaming_tools/example_streaming_with_tools", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(LIVE_MODEL))
            .preamble(
                "You are a calculator here to help the user perform arithmetic operations. \
                 Use the tools provided to answer the user's question and answer in a full sentence.",
            )
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .default_max_turns(2)
            .build();

        let mut stream = agent.prompt("Calculate 2 - 5").stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming tools prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    with_copilot_cassette(
        "streaming_tools/raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.completion(LIVE_MODEL);
            let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required);
            let stream = model.stream(request).expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}

#[tokio::test]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    with_copilot_cassette("streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls", |client| async move {
        let model = client.completion(LIVE_MODEL);
        let request = CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
            .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
            .tool(rig::tool::tool_definition(&AlphaSignal));

        let first_turn = collect_raw_stream_observation(
            model
                .stream(request)
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
        let assistant_message = Message::Assistant(rig::message::AssistantMessage::new(vec![AssistantContent::ToolCall(tool_call.clone())]));
        let tool_result_message =
            Message::User {
        content: vec![UserContent::tool_result(tool_call.id.clone(), tool_call.function.name.clone(), vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)])],
    };
        let followup_request = CompletionRequest::new(
                "Now reply in one short sentence using the provided tool result. Do not call any tools.",
            )
            .preamble("Use the provided tool result and answer directly.")
            .message(assistant_message)
            .message(tool_result_message);

        let second_turn = collect_raw_stream_observation(
            model
                .stream(followup_request)
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
    }).await;
}
