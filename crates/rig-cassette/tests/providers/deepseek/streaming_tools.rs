//! DeepSeek streaming tools smoke test.
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig::providers::deepseek::DEEPSEEK_V4_FLASH;

use super::support::with_deepseek_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_raw_stream_contains_distinct_tool_calls_before_text,
    assert_raw_stream_text_contains, assert_raw_stream_tool_call_precedes_text,
    assert_stream_contains_zero_arg_tool_call_named, collect_raw_stream_observation,
    zero_arg_tool_definition,
};
use rig::completion::{CompletionRequest, GenerationOptions, Reasoning};

fn non_thinking() -> GenerationOptions {
    GenerationOptions::default().reasoning(Reasoning::Off)
}

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    with_deepseek_cassette(
        "streaming_tools/raw_stream_emits_required_zero_arg_tool_call",
        |client| async move {
            let model = client.completion(DEEPSEEK_V4_FLASH);
            let request = CompletionRequest::new(REQUIRED_ZERO_ARG_TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .options(non_thinking());
            let stream = model.stream(request).expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}

#[tokio::test]
#[ignore = "deepseek-v4-flash now streams text before its tool calls, even for the original request bytes (3 live attempts, 2026-10-03)"]
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    with_deepseek_cassette(
        "streaming_tools/raw_stream_surfaces_two_distinct_tool_calls_before_text",
        |client| async move {
            let model = client.completion(DEEPSEEK_V4_FLASH);
            let request = CompletionRequest::new(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .options(non_thinking());

            let observation = collect_raw_stream_observation(
                model.stream(request).expect("raw stream should start"),
            )
            .await;

            assert_raw_stream_contains_distinct_tool_calls_before_text(
                &observation,
                &["lookup_harbor_label", "lookup_orchard_label"],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    with_deepseek_cassette(
        "streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls",
        |client| async move {
            let model = client.completion(DEEPSEEK_V4_FLASH);
            let request = CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .options(non_thinking());

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
            let tool_result_message = Message::User {
        content: vec![UserContent::tool_result(tool_call.id.clone(), tool_call.function.name.clone(), vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)])],
    };
            let followup_request = CompletionRequest::new(
                    "Now reply in one short sentence using the provided tool result. Do not call any tools.",
                )
                .preamble("Use the provided tool result and answer directly.")
                .message(assistant_message)
                .message(tool_result_message)
                .options(non_thinking());

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
        },
    )
    .await;
}
