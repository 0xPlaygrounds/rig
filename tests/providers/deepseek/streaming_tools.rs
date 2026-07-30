//! DeepSeek streaming tools smoke test.

use rig::OneOrMany;
use rig::completion::CompletionRequest;
use rig::http_runtime::HttpRuntime;
use rig::message::{AssistantContent, Message, ToolChoice};
use rig::prelude::*;
use rig::providers::deepseek;
use rig::providers::deepseek::DEEPSEEK_V4_FLASH;

use super::support::with_deepseek_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, REQUIRED_ZERO_ARG_TOOL_PROMPT,
    Subtract, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number,
    assert_raw_stream_contains_distinct_tool_calls_before_text, assert_raw_stream_text_contains,
    assert_raw_stream_tool_call_arguments_are_objects, assert_raw_stream_tool_call_precedes_text,
    assert_stream_contains_zero_arg_tool_call_named, assert_tool_call_precedes_later_text,
    assert_two_tool_roundtrip_contract, collect_raw_stream_observation,
    collect_stream_final_response, collect_stream_observation, zero_arg_tool_definition,
};

fn non_thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "disabled" }
    })
}

#[tokio::test]
async fn streaming_chat_with_tools() {
    with_deepseek_cassette(
        "streaming_tools/streaming_chat_with_tools",
        |env| async move {
            let agent = AgentBuilder::new(env.provider(DEEPSEEK_V4_FLASH))
                .preamble(
                    "You are a calculator here to help the user perform arithmetic operations.",
                )
                .max_tokens(1024)
                .tool(Adder)
                .tool(Subtract)
                .additional_params(non_thinking_params())
                .default_max_turns(2)
                .build();

            let history: &[Message] = &[];
            let mut stream = Box::pin(
                agent
                    .runner("Calculate 2 - 5")
                    .history(history)
                    .stream_run(),
            );
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming chat should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn raw_stream_emits_required_zero_arg_tool_call() {
    with_deepseek_cassette(
        "streaming_tools/raw_stream_emits_required_zero_arg_tool_call",
        |env| async move {
            let model_cfg = env.config(DEEPSEEK_V4_FLASH);
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                tools: vec![zero_arg_tool_definition("ping")],
                tool_choice: Some(ToolChoice::Required),
                additional_params: Some(non_thinking_params()),
                ..CompletionRequest::from_prompt(REQUIRED_ZERO_ARG_TOOL_PROMPT)
            };
            let stream = deepseek::functions::open_stream(&model_cfg, &rt, request)
                .await
                .expect("stream should start");

            assert_stream_contains_zero_arg_tool_call_named(stream, "ping", true).await;
        },
    )
    .await;
}

#[tokio::test]
async fn raw_stream_surfaces_two_distinct_tool_calls_before_text() {
    with_deepseek_cassette(
        "streaming_tools/raw_stream_surfaces_two_distinct_tool_calls_before_text",
        |env| async move {
            let model_cfg = env.config(DEEPSEEK_V4_FLASH);
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                tools: vec![
                    rig::tool::portable_tool_definition(&AlphaSignal),
                    rig::tool::portable_tool_definition(&BetaSignal),
                ],
                additional_params: Some(non_thinking_params()),
                ..CompletionRequest::with_history(
                    Some(TWO_TOOL_STREAM_PREAMBLE),
                    Vec::new(),
                    TWO_TOOL_STREAM_PROMPT,
                )
            };

            let observation = collect_raw_stream_observation(
                deepseek::functions::open_stream(&model_cfg, &rt, request)
                    .await
                    .expect("raw stream should start"),
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

/// Live end-to-end guard for the #1958 invariant: every tool call surfaced by
/// the streaming aggregator carries a JSON **object** as its arguments, never a
/// bare string. Recorded against real DeepSeek traffic with two tool calls in a
/// single streamed turn (which exercises the same-turn multi-tool accumulation
/// path). DeepSeek assigns distinct indices, so this complements — rather than
/// replaces — the in-crate unit tests that drive the same-index eviction path
/// directly (a quirk only some API gateways emit and not reproducible live).
#[tokio::test]
async fn raw_stream_tool_call_arguments_are_objects() {
    with_deepseek_cassette(
        "streaming_tools/raw_stream_tool_call_arguments_are_objects",
        |env| async move {
            let model_cfg = env.config(DEEPSEEK_V4_FLASH);
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                tools: vec![
                    rig::tool::portable_tool_definition(&AlphaSignal),
                    rig::tool::portable_tool_definition(&BetaSignal),
                ],
                additional_params: Some(non_thinking_params()),
                ..CompletionRequest::with_history(
                    Some(TWO_TOOL_STREAM_PREAMBLE),
                    Vec::new(),
                    TWO_TOOL_STREAM_PROMPT,
                )
            };

            let observation = collect_raw_stream_observation(
                deepseek::functions::open_stream(&model_cfg, &rt, request)
                    .await
                    .expect("raw stream should start"),
            )
            .await;

            assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &["lookup_harbor_label", "lookup_orchard_label"],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_chat_surfaces_two_distinct_tool_calls_before_final_answer() {
    with_deepseek_cassette(
        "streaming_tools/streaming_chat_surfaces_two_distinct_tool_calls_before_final_answer",
        |env| async move {
            let agent = AgentBuilder::new(env.provider(DEEPSEEK_V4_FLASH))
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .additional_params(non_thinking_params())
                .build();

            let history: &[Message] = &[];
            let mut stream = Box::pin(
                agent
                    .runner(TWO_TOOL_STREAM_PROMPT)
                    .history(history)
                    .max_turns(8)
                    .stream_run(),
            );
            let observation = collect_stream_observation(&mut stream).await;

            assert_two_tool_roundtrip_contract(
                &observation,
                &["lookup_harbor_label", "lookup_orchard_label"],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_chat_emits_tool_call_before_later_text() {
    with_deepseek_cassette(
        "streaming_tools/streaming_chat_emits_tool_call_before_later_text",
        |env| async move {
            let agent = AgentBuilder::new(env.provider(DEEPSEEK_V4_FLASH))
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .additional_params(non_thinking_params())
                .build();

            let history: &[Message] = &[];
            let mut stream = Box::pin(
                agent
                    .runner(ORDERED_TOOL_STREAM_PROMPT)
                    .history(history)
                    .max_turns(5)
                    .stream_run(),
            );
            let observation = collect_stream_observation(&mut stream).await;

            assert_tool_call_precedes_later_text(
                &observation,
                "lookup_harbor_label",
                &[ALPHA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn raw_followup_uses_tool_result_without_new_tool_calls() {
    with_deepseek_cassette(
        "streaming_tools/raw_followup_uses_tool_result_without_new_tool_calls",
        |env| async move {
            let model_cfg = env.config(DEEPSEEK_V4_FLASH);
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                tools: vec![rig::tool::portable_tool_definition(&AlphaSignal)],
                additional_params: Some(non_thinking_params()),
                ..CompletionRequest::with_history(
                    Some(ORDERED_TOOL_STREAM_PREAMBLE),
                    Vec::new(),
                    ORDERED_TOOL_STREAM_PROMPT,
                )
            };

            let first_turn = collect_raw_stream_observation(
                deepseek::functions::open_stream(&model_cfg, &rt, request)
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
            let tool_result_message = Message::tool_result_with_call_id(
                tool_call.id,
                tool_call.call_id,
                ALPHA_SIGNAL_OUTPUT,
            );
            let followup_request = CompletionRequest {
                additional_params: Some(non_thinking_params()),
                ..CompletionRequest::with_history(
                    Some("Use the provided tool result and answer directly."),
                    vec![assistant_message, tool_result_message],
                    "Now reply in one short sentence using the provided tool result. Do not call any tools.",
                )
            };

            let second_turn = collect_raw_stream_observation(
                deepseek::functions::open_stream(&model_cfg, &rt, followup_request)
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
        },
    )
    .await;
}
