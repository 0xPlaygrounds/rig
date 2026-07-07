//! Hook-system stress suite: streaming lifecycle and blocking-vs-streaming
//! parity — `TextDelta` / `StreamResponseFinish` / `ModelTurnFinished` on the
//! streaming surface, `RewriteResult` redaction reaching the `FinalResponse`,
//! `Skip` on the streaming driver, and the same workflow producing the same
//! answer on both surfaces. Recorded against real Gemini.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::gemini;
use rig::streaming::StreamingPrompt;

use super::super::hook_stress_support::{
    CHAIN_PREAMBLE, EventTap, ResultRewrite, RewriteToolResult,
};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, SkipToolHook};
use crate::support::{
    assert_mentions_expected_number, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_text_only_emits_text_deltas_and_stream_finish() {
    let tap = EventTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_text_only_emits_text_deltas_and_stream_finish",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a concise assistant. Answer directly in plain text.")
                .temperature(0.0)
                .build();

            let mut stream = agent
                .stream_prompt("In one short sentence, describe the color of a clear daytime sky.")
                .add_hook(tap)
                .multi_turn(2)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_nonempty_response(&final_text);

            assert_eq!(probe.is_streaming(), Some(true));
            assert!(
                probe.count("TextDelta") >= 1,
                "a streamed text turn must emit TextDelta events"
            );
            assert!(
                probe.count("StreamResponseFinish") >= 1,
                "a streamed text turn must emit StreamResponseFinish"
            );
            assert!(
                probe.count("ModelTurnFinished") >= 1,
                "ModelTurnFinished must fire on the streaming surface"
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
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let mut stream = agent
                .stream_prompt(
                    "First add 40 and 2 with the add tool. Then subtract 10 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .add_hook(tap)
                .multi_turn(6)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
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
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use the add tool, then report the exact tool \
                     result text verbatim.",
                )
                .temperature(0.0)
                .tool(add)
                .build();

            let mut stream = agent
                .stream_prompt(
                    "Use the add tool to add 5 and 5, then report the exact tool result.",
                )
                .add_hook(RewriteToolResult {
                    tool: "add",
                    rewrite: ResultRewrite::Replace("STREAM-REDACTED-Q3"),
                })
                .multi_turn(4)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
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
async fn streaming_skip_leaves_tool_unexecuted() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_skip_leaves_tool_unexecuted",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools. If a tool \
                     reports it is unavailable, acknowledge that and report any results you have.",
                )
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            let mut stream = agent
                .stream_prompt("Add 14 and 6, and subtract 9 from 40. Report what you can.")
                .add_hook(SkipToolHook {
                    tool_name: "subtract",
                    reason: "the subtract tool is offline; continue without it",
                })
                .multi_turn(5)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
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

    // Blocking surface.
    let add_b = CountingAdd::default();
    let sub_b = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add_b)
                .tool(sub_b)
                .build();
            let response = agent
                .prompt(PROMPT)
                .max_turns(6)
                .await
                .expect("blocking parity run should succeed");
            assert_mentions_expected_number(&response, EXPECTED);
        },
    )
    .await;

    // Streaming surface — same workflow, same expected answer.
    let add_s = CountingAdd::default();
    let sub_s = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_streaming",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add_s)
                .tool(sub_s)
                .build();
            let mut stream = agent.stream_prompt(PROMPT).multi_turn(6).await;
            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_mentions_expected_number(&final_text, EXPECTED);
        },
    )
    .await;
}
