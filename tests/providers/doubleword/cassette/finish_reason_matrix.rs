//! Finish-reason matrix for Doubleword's blocking and streaming surfaces.
//!
//! Natural stops, token-budget truncation and tool-call turns are crossed with
//! both transports. The tests assert rig's normalized terminal reason and the
//! provider's recorded wire reason independently; this catches either a wire
//! dialect change or a normalization regression.
//!
//! | reason | blocking | streaming | control |
//! |---|---|---|---|
//! | natural stop | `stop` | `stop` | `reasoning_effort: none` |
//! | budget exhausted | `length` | `length` | `max_tokens: 1` |
//! | required tool | `tool_calls` | `tool_calls` | zero-argument `ping` |
//!
//! The stop control also records a live dialect rule discovered while probing:
//! Doubleword rejects `chat_template_kwargs.enable_thinking` and directs
//! callers to `reasoning_effort`; `"none"` disables reasoning on Qwen 3.5.

use rig::completion::{CompletionModel, FinishReason, NormalizeCompletionResponse};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::doubleword;
use serde_json::json;

use super::super::support::{recorded_chat_calls, with_doubleword_cassette};
use crate::support::{collect_text_and_terminal, zero_arg_tool_definition};

const STOP_PROMPT: &str = "Reply with exactly: done";
const LENGTH_PROMPT: &str = "Explain every step of how a compiler optimizes a large program.";
const TOOL_PROMPT: &str = "Call ping now.";

fn recorded_finish_reason(scenario: &str, streaming: bool) -> String {
    let calls = recorded_chat_calls(scenario);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.status, 200);

    if streaming {
        call.stream_chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["finish_reason"].as_str())
            .next_back()
            .expect("stream should record a finish reason")
            .to_owned()
    } else {
        call.response_json.as_ref().expect("blocking JSON response")["choices"][0]["finish_reason"]
            .as_str()
            .expect("blocking response should record a finish reason")
            .to_owned()
    }
}

async fn blocking_stop(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let raw = model
        .raw_completion(
            model
                .completion_request(STOP_PROMPT)
                .additional_params(json!({ "reasoning_effort": "none" }))
                .max_tokens(64)
                .build(),
        )
        .await
        .expect("blocking stop probe");
    let normalized = raw.normalize("doubleword").expect("normalize stop turn");
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Stop));
}

async fn blocking_length(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let raw = model
        .raw_completion(
            model
                .completion_request(LENGTH_PROMPT)
                .max_tokens(1)
                .build(),
        )
        .await
        .expect("blocking length probe should still decode");
    let normalized = raw
        .normalize("doubleword")
        .expect("a contentless truncated turn is still a completion");
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
}

async fn blocking_tool_calls_body(client: doubleword::Client) {
    let model = client.completion_model(doubleword::QWEN3_5_397B_A17B);
    let raw = model
        .raw_completion(
            model
                .completion_request(TOOL_PROMPT)
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .max_tokens(256)
                .build(),
        )
        .await
        .expect("blocking tool-call probe");
    let normalized = raw
        .normalize("doubleword")
        .expect("normalize tool-call turn");
    assert_eq!(normalized.finish_reason(), Some(FinishReason::ToolCalls));
    assert!(
        normalized
            .choice
            .iter()
            .any(|part| matches!(part, AssistantContent::ToolCall(_)))
    );
}

async fn streaming_reason(
    client: doubleword::Client,
    prompt: &'static str,
    max_tokens: u64,
    stop_probe: bool,
) -> FinishReason {
    let model = client.completion_model(doubleword::QWEN3_5_9B);
    let mut builder = model.completion_request(prompt).max_tokens(max_tokens);
    if stop_probe {
        builder = builder.additional_params(json!({ "reasoning_effort": "none" }));
    }
    let stream = model.stream(builder.build()).await.expect("stream probe");
    let (_, terminal) = collect_text_and_terminal(stream).await;
    terminal
        .expect("stream should carry a terminal record")
        .finish_reason
        .expect("stream should normalize its finish reason")
}

#[tokio::test]
async fn blocking_natural_stop() {
    const SCENARIO: &str = "finish_reason_matrix/blocking_natural_stop";
    with_doubleword_cassette("finish_reason_matrix/blocking_natural_stop", blocking_stop).await;
    assert_eq!(recorded_finish_reason(SCENARIO, false), "stop");
}

#[tokio::test]
async fn streaming_natural_stop() {
    const SCENARIO: &str = "finish_reason_matrix/streaming_natural_stop";
    with_doubleword_cassette(
        "finish_reason_matrix/streaming_natural_stop",
        |client| async move {
            assert_eq!(
                streaming_reason(client, STOP_PROMPT, 64, true).await,
                FinishReason::Stop
            );
        },
    )
    .await;
    assert_eq!(recorded_finish_reason(SCENARIO, true), "stop");
}

#[tokio::test]
async fn blocking_token_limit() {
    const SCENARIO: &str = "finish_reason_matrix/blocking_token_limit";
    with_doubleword_cassette("finish_reason_matrix/blocking_token_limit", blocking_length).await;
    assert_eq!(recorded_finish_reason(SCENARIO, false), "length");
}

#[tokio::test]
async fn streaming_token_limit() {
    const SCENARIO: &str = "finish_reason_matrix/streaming_token_limit";
    with_doubleword_cassette(
        "finish_reason_matrix/streaming_token_limit",
        |client| async move {
            assert_eq!(
                streaming_reason(client, LENGTH_PROMPT, 1, false).await,
                FinishReason::Length
            );
        },
    )
    .await;
    assert_eq!(recorded_finish_reason(SCENARIO, true), "length");
}

#[tokio::test]
async fn blocking_tool_calls() {
    const SCENARIO: &str = "finish_reason_matrix/blocking_tool_calls";
    with_doubleword_cassette(
        "finish_reason_matrix/blocking_tool_calls",
        blocking_tool_calls_body,
    )
    .await;
    assert_eq!(recorded_finish_reason(SCENARIO, false), "tool_calls");
}

#[tokio::test]
async fn streaming_tool_calls() {
    const SCENARIO: &str = "finish_reason_matrix/streaming_tool_calls";
    with_doubleword_cassette(
        "finish_reason_matrix/streaming_tool_calls",
        |client| async move {
            let model = client.completion_model(doubleword::QWEN3_5_397B_A17B);
            let stream = model
                .stream(
                    model
                        .completion_request(TOOL_PROMPT)
                        .tool(zero_arg_tool_definition("ping"))
                        .tool_choice(ToolChoice::Required)
                        .max_tokens(256)
                        .build(),
                )
                .await
                .expect("tool stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            assert_eq!(
                terminal.expect("terminal record").finish_reason,
                Some(FinishReason::ToolCalls)
            );
        },
    )
    .await;
    assert_eq!(recorded_finish_reason(SCENARIO, true), "tool_calls");
}
