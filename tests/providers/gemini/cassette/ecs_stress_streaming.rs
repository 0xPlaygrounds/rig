//! Native streaming stress, preserving original observations and response assertions.
use super::super::{
    hook_stress_support::{CHAIN_PREAMBLE, ResultRewrite},
    support::with_gemini_cassette,
    tools_support::{CountingAdd, CountingSubtract},
};
use super::ecs_stress_streaming_runtime::{self as runtime, EventTap};
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};
use rig::{prelude::*, providers::gemini};
#[tokio::test]
async fn streaming_text_only_emits_text_deltas_and_stream_finish() {
    let tap = EventTap::default();
    let probe = tap.clone();
    with_gemini_cassette(
        "hook_stress_streaming/streaming_text_only_emits_text_deltas_and_stream_finish",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a concise assistant. Answer directly in plain text.",
                "stress-agent",
                Some(0.0),
            );
            let final_text = runtime::prompt(
                &mut ecs,
                "In one short sentence, describe the color of a clear daytime sky.",
                2,
                true,
                vec![tap],
            )
            .await;
            assert_nonempty_response(&final_text);
            assert_eq!(probe.is_streaming(), Some(true));
            assert!(
                probe.count("TextDelta") >= 1,
                "a streamed text turn must emit TextDelta events"
            );
            assert!(
                probe.count("CompletionResponse") >= 1,
                "a streamed text turn must emit CompletionResponse once the stream is assembled"
            );
            assert!(
                probe.count("ModelTurnFinished") >= 1,
                "ModelTurnFinished must fire on the streaming surface"
            );
            assert_eq!(
                probe.count("CompletionResponse"),
                probe.count("ModelTurnFinished"),
                "CompletionResponse and ModelTurnFinished fire once per accepted turn"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_tool_turns_fire_model_turn_finished() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = EventTap::default();
    let probe = tap.clone();
    with_gemini_cassette(
        "hook_stress_streaming/streaming_tool_turns_fire_model_turn_finished",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            ecs.tool(subtract);
            let final_text = runtime::prompt(
                &mut ecs,
                "First add 40 and 2 with the add tool. Then subtract 10 from that sum with the \
                     subtract tool. Report the final number.",
                6,
                true,
                vec![tap],
            )
            .await;
            assert_nonempty_response(&final_text);
            assert_eq!(probe.is_streaming(), Some(true));
            assert!(
                probe.count("ToolCall") >= 1,
                "the streamed run should call tools"
            );
            assert!(
                probe.count("ModelTurnFinished") >= 2,
                "ModelTurnFinished must fire once per accepted turn on the streaming surface, \
                 including tool turns"
            );
            assert_eq!(
                probe.count("CompletionResponse"),
                probe.count("ModelTurnFinished"),
                "CompletionResponse fires once per accepted turn on the streaming surface, \
                 tool-only turns included"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_result_redaction_reaches_final_response() {
    let add = CountingAdd::default();
    with_gemini_cassette(
        "hook_stress_streaming/streaming_result_redaction_reaches_final_response",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use the add tool, then report the exact tool \
                     result text verbatim.",
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            runtime::rewrite_result(
                &mut ecs,
                "add",
                ResultRewrite::Replace("STREAM-REDACTED-Q3"),
            );
            let final_text = runtime::prompt(
                &mut ecs,
                "Use the add tool to add 5 and 5, then report the exact tool result.",
                4,
                true,
                vec![],
            )
            .await;
            assert!(
                final_text.contains("STREAM-REDACTED-Q3"),
                "the redacted result must reach the streamed final response: {final_text:?}"
            );
            assert!(
                !final_text.contains("10"),
                "the raw tool result must not reach the model: {final_text:?}"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_active_tools_narrowing_filters_a_tool() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    with_gemini_cassette(
        "hook_stress_streaming/streaming_active_tools_narrowing_filters_a_tool",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. Use a provided tool for any arithmetic you \
                     can; if a tool is unavailable, say so and continue.",
                "stress-agent",
                None,
            );
            ecs.tool(add);
            ecs.tool(subtract);
            runtime::patch(
                &mut ecs,
                rig_ecs::agent::RequestPatch {
                    active_tools: Some((["add"]).into_iter().map(str::to_owned).collect()),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            );
            let final_text = runtime::prompt(
                &mut ecs,
                "Compute 12 + 8, then compute 30 - 7. Report whichever you can.",
                5,
                true,
                vec![],
            )
            .await;
            assert_nonempty_response(&final_text);
            assert!(
                add_calls.count() >= 1,
                "add stays advertised and should run"
            );
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract is filtered out of active_tools on the streaming surface too"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_skip_leaves_tool_unexecuted() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_calls = subtract.counter.clone();
    with_gemini_cassette(
        "hook_stress_streaming/streaming_skip_leaves_tool_unexecuted",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                "You are a calculator assistant. You MUST use the provided tools. If a tool \
                     reports it is unavailable, acknowledge that and report any results you have.",
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add);
            ecs.tool(subtract);
            runtime::skip(
                &mut ecs,
                "subtract",
                "the subtract tool is offline; continue without it",
            );
            let final_text = runtime::prompt(
                &mut ecs,
                "Add 14 and 6, and subtract 9 from 40. Report what you can.",
                5,
                true,
                vec![],
            )
            .await;
            assert_nonempty_response(&final_text);
            assert_eq!(
                subtract_calls.count(),
                0,
                "a skipped tool must never execute on the streaming surface"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn blocking_and_streaming_produce_same_final_answer() {
    const PROMPT: &str = "First add 10 and 5 with the add tool. Then subtract 3 from that sum with \
         the subtract tool. Report the final number.";
    const EXPECTED: i32 = 12;
    let add_b = CountingAdd::default();
    let sub_b = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add_b);
            ecs.tool(sub_b);
            let response = runtime::prompt(&mut ecs, PROMPT, 6, false, vec![]).await;
            assert_mentions_expected_number(&response, EXPECTED);
        },
    )
    .await;
    let add_s = CountingAdd::default();
    let sub_s = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_streaming",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                CHAIN_PREAMBLE,
                "stress-agent",
                Some(0.0),
            );
            ecs.tool(add_s);
            ecs.tool(sub_s);
            let final_text = runtime::prompt(&mut ecs, PROMPT, 6, true, vec![]).await;
            assert_mentions_expected_number(&final_text, EXPECTED);
        },
    )
    .await;
}
