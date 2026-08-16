//! Edge matrix for rig#2354: a `max_tokens`-truncated tool call destroyed the
//! whole blocking DeepSeek response.
//!
//! DeepSeek emits the tool call anyway when the budget runs out mid-arguments:
//! the turn comes back with `finish_reason: "length"` and
//! `tool_calls[].function.arguments` cut off partway through the JSON object.
//! `deepseek::Function` parsed that strictly, so the *whole* `CompletionResponse`
//! failed to decode and the text, usage, id, model and finish reason went with
//! it — while the streaming path kept the turn and dropped the unusable call
//! ([`UnparseableToolInput::Drop`]). The two transports disagreed about
//! identical wire bytes.
//!
//! Live budget sweep against `deepseek-v4-flash` (thinking disabled), one tool
//! whose required `summary` argument must be long:
//!
//! | budget | recorded `arguments` | `finish_reason` | class |
//! |---|---|---|---|
//! | 12 | *(no tool call at all)* | `length` | control: contentless truncation |
//! | 16 | `""` | `length` | boundary: cut before first argument token |
//! | 20 | `""` | `length` | boundary: cut before first argument token |
//! | 24 | `{"summary": ` | `length` | **truncated** |
//! | 32 | `{"summary": "Log this incident: the` | `length` | **truncated** |
//! | 48 | `…raced the artifact uploader, then the ret` | `length` | **truncated** |
//! | 64 | `…had to drain three` | `length` | **truncated** |
//! | 96 | complete object | `tool_calls` | control: untouched |
//!
//! Every budget is recorded on both transports, plus the shapes that share the
//! same decode: parallel calls where only the second is cut, a turn that spoke
//! before it was cut, a reasoner turn cut mid-arguments, and the agent loop.
//! See the module doc table in the PR body for per-cell status.

use anyhow::Result;
use rig::completion::{CompletionModel, NormalizeCompletionResponse, ToolDefinition};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{
    collect_raw_stream_outcome, recorded_response, recorded_stream_chunks,
    with_deepseek_truncation_cassette_result,
};

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

/// The tool's one required argument must be long enough that a small budget
/// lands inside the JSON string rather than after it.
const TOOL_PREAMBLE: &str = "You must call the file_report tool. The summary argument must be a verbatim, complete restatement of the user's entire request, word for word, at least 120 words long.";
const TEXT_FIRST_PREAMBLE: &str = "First write exactly one short sentence acknowledging the request, then call the file_report tool whose summary argument is a verbatim, complete restatement of the user request, word for word, at least 120 words long.";
const PARALLEL_PREAMBLE: &str = "You must call page_oncall with team set to platform, and then file_report whose summary argument is a verbatim, complete restatement of the user request, word for word, at least 120 words long. Emit both calls in the same turn.";
const INCIDENT_PROMPT: &str = "Log this incident: the nightly build broke because the cache warmer raced the artifact uploader, then the retry storm saturated the queue, and the on-call engineer had to drain three regions by hand while the dashboards lagged behind by nine minutes.";
/// The reasoner cells need a turn whose *thinking* is trivial and whose
/// *arguments* are long, so the budget reliably lands inside the JSON string
/// rather than inside the reasoning. A verbatim-copy instruction does that:
/// there is nothing to reason about and a great deal to type.
const REASONER_TOOL_PREAMBLE: &str = "Call the file_report tool exactly once. Set its summary argument to the user's text, copied out verbatim and in full. Do not summarise, do not shorten, do not think about it.";
const REASONER_INCIDENT_PROMPT: &str = "Copy this into file_report: the nightly build broke because the cache warmer raced the artifact uploader; the retry storm then saturated the queue; the on-call engineer drained three regions by hand; the dashboards lagged nine minutes behind; the checksum verifier timed out twice; the release channel notification never fired; the rollback took forty minutes; the incident channel filled with duplicate alerts; the paging policy escalated to the wrong rotation; and the postmortem template was missing three required sections.";

fn non_thinking_params() -> Value {
    json!({ "thinking": { "type": "disabled" } })
}

fn thinking_params() -> Value {
    json!({ "thinking": { "type": "enabled" } })
}

fn file_report_tool() -> ToolDefinition {
    ToolDefinition {
        name: "file_report".to_owned(),
        description: "File an incident report.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A long verbatim summary of the incident.",
                },
            },
            "required": ["summary"],
        }),
    }
}

fn page_oncall_tool() -> ToolDefinition {
    ToolDefinition {
        name: "page_oncall".to_owned(),
        description: "Page the on-call engineer.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "team": { "type": "string" } },
            "required": ["team"],
        }),
    }
}

fn request(
    model: &deepseek::CompletionModel,
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Value,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    request_for(model, INCIDENT_PROMPT, preamble, tools, params, max_tokens)
}

fn request_for(
    model: &deepseek::CompletionModel,
    prompt: &str,
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Value,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    let mut builder = model
        .completion_request(prompt)
        .preamble(preamble.to_owned())
        .additional_params(params)
        .max_tokens(max_tokens);
    for tool in tools {
        builder = builder.tool(tool);
    }
    builder.build()
}

fn tool_calls(choice: &[AssistantContent]) -> Vec<&rig::message::ToolCall> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// ================================================================
// Premise assertions, derived from each cell's own recorded bytes
// ================================================================

/// The `arguments` strings the recorded blocking turn carried, in wire order.
fn recorded_blocking_arguments(scenario: &str) -> Vec<String> {
    let response = recorded_response(scenario);
    response["choices"][0]["message"]["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    call["function"]["arguments"]
                        .as_str()
                        .unwrap_or("")
                        .to_owned()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn recorded_blocking_finish_reason(scenario: &str) -> String {
    recorded_response(scenario)["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// The `arguments` fragments the recorded stream delivered, concatenated per
/// `tool_calls[].index`.
fn recorded_stream_arguments(scenario: &str) -> Vec<String> {
    let mut accumulated: Vec<String> = Vec::new();
    for chunk in recorded_stream_chunks(scenario) {
        let Some(calls) = chunk["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for call in calls {
            let index = call["index"].as_u64().unwrap_or(0) as usize;
            if accumulated.len() <= index {
                accumulated.resize(index + 1, String::new());
            }
            if let Some(fragment) = call["function"]["arguments"].as_str() {
                accumulated[index].push_str(fragment);
            }
        }
    }
    accumulated
}

fn recorded_stream_finish_reason(scenario: &str) -> String {
    recorded_stream_chunks(scenario)
        .into_iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["finish_reason"]
                .as_str()
                .map(str::to_owned)
        })
        .next_back()
        .unwrap_or_default()
}

fn assert_unparseable(arguments: &str, scenario: &str) {
    assert!(
        !arguments.trim().is_empty(),
        "{scenario}: premise requires a non-empty truncated argument string, got {arguments:?}"
    );
    assert!(
        serde_json::from_str::<Value>(arguments).is_err(),
        "{scenario}: premise requires arguments that do not parse as JSON, got {arguments:?}"
    );
}

fn assert_parseable(arguments: &str, scenario: &str) {
    assert!(
        serde_json::from_str::<Value>(arguments).is_ok(),
        "{scenario}: control cell requires arguments that parse as JSON, got {arguments:?}"
    );
}

// ================================================================
// Shared cell bodies
// ================================================================

/// Blocking: the turn survives, reports `Length`, keeps usage/id/model, and
/// carries no tool call at all — the truncated one is dropped exactly as the
/// streaming path drops it.
async fn assert_blocking_truncation_survives(
    client: &deepseek::Client,
    max_tokens: u64,
) -> Result<()> {
    let model = client.completion_model(MODEL);
    let raw = model
        .raw_completion(request(
            &model,
            TOOL_PREAMBLE,
            vec![file_report_tool()],
            non_thinking_params(),
            max_tokens,
        ))
        .await?;

    assert_eq!(
        raw.choices[0].finish_reason, "length",
        "premise: the recorded turn must have been cut by the budget"
    );
    let deepseek::Message::Assistant {
        tool_calls: wire_calls,
        ..
    } = &raw.choices[0].message;
    assert!(
        wire_calls.is_empty(),
        "the unusable call is dropped at decode rather than surfaced: {wire_calls:?}"
    );

    let normalized = raw.clone().normalize("deepseek")?;
    assert_eq!(
        normalized.finish_reason(),
        Some(rig::completion::FinishReason::Length),
        "the surviving turn reports the truncation"
    );
    assert!(
        tool_calls(&normalized.choice).is_empty(),
        "an unusable call must not reach the caller: {:?}",
        normalized.choice
    );
    assert!(
        normalized.usage.total_tokens > 0 && normalized.usage.input_tokens > 0,
        "usage survives the truncated call: {:?}",
        normalized.usage
    );
    assert!(
        normalized.response_id.is_some(),
        "the response id survives the truncated call"
    );
    assert!(
        normalized.model.is_some(),
        "the model name survives the truncated call"
    );
    Ok(())
}

/// Streaming twin: the stream already dropped the unusable call; this pins that
/// it still does, and that its terminal record reports the same `Length`.
async fn assert_streaming_truncation_survives(
    client: &deepseek::Client,
    max_tokens: u64,
) -> Result<()> {
    let model = client.completion_model(MODEL);
    let outcome = collect_raw_stream_outcome(
        model
            .stream(request(
                &model,
                TOOL_PREAMBLE,
                vec![file_report_tool()],
                non_thinking_params(),
                max_tokens,
            ))
            .await?,
    )
    .await;

    assert!(
        outcome.errors.is_empty(),
        "stream errors: {:?}",
        outcome.errors
    );
    assert!(
        outcome.tool_calls.is_empty(),
        "the stream must drop the truncated call: {:?}",
        outcome.tool_call_names()
    );
    assert_eq!(
        outcome.finish_reason(),
        Some(rig::completion::FinishReason::Length),
        "the streamed terminal reports the truncation"
    );
    let usage = outcome
        .final_record
        .as_ref()
        .map(|record| record.usage)
        .unwrap_or_default();
    assert!(
        usage.total_tokens > 0,
        "streamed usage survives the truncated call: {usage:?}"
    );
    Ok(())
}

// ================================================================
// A. Blocking budget sweep
// ================================================================

#[tokio::test]
async fn blocking_budget_12_truncates_before_any_tool_call() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_12_truncates_before_any_tool_call";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_12_truncates_before_any_tool_call",
        |client| async move {
            let model = client.completion_model(MODEL);
            let raw = model
                .raw_completion(request(
                    &model,
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    12,
                ))
                .await?;
            let normalized = raw.normalize("deepseek")?;
            assert_eq!(
                normalized.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            assert!(
                tool_calls(&normalized.choice).is_empty(),
                "no call was emitted at all: {:?}",
                normalized.choice
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_budget_12_truncates_before_any_tool_call should replay from its cassette");

    assert!(
        recorded_blocking_arguments(SCENARIO).is_empty(),
        "control premise: the recorded turn carries no tool call"
    );
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn blocking_budget_16_empty_arguments_are_dropped_on_length() {
    const SCENARIO: &str =
        "truncation_matrix/blocking_budget_16_empty_arguments_are_dropped_on_length";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_16_empty_arguments_are_dropped_on_length",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request(
                    &model,
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    16,
                ))
                .await?
                .normalize("deepseek")?;
            let calls = tool_calls(&normalized.choice);
            assert!(
                calls.is_empty(),
                "`length` identifies the empty argument slot as an incomplete call"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "blocking_budget_16_empty_arguments_are_dropped_on_length should replay from its cassette",
    );

    let arguments = recorded_blocking_arguments(SCENARIO);
    assert_eq!(
        arguments,
        vec![String::new()],
        "boundary premise: empty arguments"
    );
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn blocking_budget_20_empty_arguments_are_dropped_on_length() {
    const SCENARIO: &str =
        "truncation_matrix/blocking_budget_20_empty_arguments_are_dropped_on_length";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_20_empty_arguments_are_dropped_on_length",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request(
                    &model,
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    20,
                ))
                .await?
                .normalize("deepseek")?;
            let calls = tool_calls(&normalized.choice);
            assert!(calls.is_empty());
            assert_eq!(
                normalized.finish_reason(),
                Some(rig::completion::FinishReason::Length),
                "the boundary is a `length` turn, not a natural stop"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "blocking_budget_20_empty_arguments_are_dropped_on_length should replay from its cassette",
    );

    assert_eq!(recorded_blocking_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn blocking_budget_24_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_24_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_24_truncated_arguments_keep_the_turn",
        |client| async move { assert_blocking_truncation_survives(&client, 24).await },
    )
    .await
    .expect("blocking_budget_24_truncated_arguments_keep_the_turn should replay from its cassette");

    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn blocking_budget_32_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_32_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_32_truncated_arguments_keep_the_turn",
        |client| async move { assert_blocking_truncation_survives(&client, 32).await },
    )
    .await
    .expect("blocking_budget_32_truncated_arguments_keep_the_turn should replay from its cassette");

    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn blocking_budget_48_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_48_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_48_truncated_arguments_keep_the_turn",
        |client| async move { assert_blocking_truncation_survives(&client, 48).await },
    )
    .await
    .expect("blocking_budget_48_truncated_arguments_keep_the_turn should replay from its cassette");

    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn blocking_budget_64_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_64_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_64_truncated_arguments_keep_the_turn",
        |client| async move { assert_blocking_truncation_survives(&client, 64).await },
    )
    .await
    .expect("blocking_budget_64_truncated_arguments_keep_the_turn should replay from its cassette");

    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn blocking_budget_96_complete_arguments_are_untouched() {
    const SCENARIO: &str = "truncation_matrix/blocking_budget_96_complete_arguments_are_untouched";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_96_complete_arguments_are_untouched",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request(
                    &model,
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    96,
                ))
                .await?
                .normalize("deepseek")?;
            let calls = tool_calls(&normalized.choice);
            assert_eq!(calls.len(), 1, "the complete call still reaches the caller");
            assert!(
                calls[0].function.arguments["summary"].is_string(),
                "the tolerant parse must not weaken a complete payload: {:?}",
                calls[0].function.arguments
            );
            assert_eq!(
                normalized.finish_reason(),
                Some(rig::completion::FinishReason::ToolCalls)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_budget_96_complete_arguments_are_untouched should replay from its cassette");

    assert_parseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "tool_calls");
}

// ================================================================
// B. Streaming budget sweep (the parity twin)
// ================================================================

#[tokio::test]
async fn streaming_budget_12_truncates_before_any_tool_call() {
    const SCENARIO: &str = "truncation_matrix/streaming_budget_12_truncates_before_any_tool_call";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_12_truncates_before_any_tool_call",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request(
                        &model,
                        TOOL_PREAMBLE,
                        vec![file_report_tool()],
                        non_thinking_params(),
                        12,
                    ))
                    .await?,
            )
            .await;
            assert!(outcome.tool_calls.is_empty());
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_budget_12_truncates_before_any_tool_call should replay from its cassette");

    assert!(recorded_stream_arguments(SCENARIO).is_empty());
    assert_eq!(recorded_stream_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn streaming_budget_16_empty_arguments_are_dropped_on_length() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_budget_16_empty_arguments_are_dropped_on_length";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_16_empty_arguments_are_dropped_on_length",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request(
                        &model,
                        TOOL_PREAMBLE,
                        vec![file_report_tool()],
                        non_thinking_params(),
                        16,
                    ))
                    .await?,
            )
            .await;
            assert!(
                outcome.tool_calls.is_empty(),
                "`length` prevents an incomplete zero-byte call from reaching a tool"
            );
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "streaming_budget_16_empty_arguments_are_dropped_on_length should replay from its cassette",
    );

    assert_eq!(recorded_stream_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_stream_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn streaming_budget_24_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_budget_24_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_24_truncated_arguments_keep_the_turn",
        |client| async move { assert_streaming_truncation_survives(&client, 24).await },
    )
    .await
    .expect(
        "streaming_budget_24_truncated_arguments_keep_the_turn should replay from its cassette",
    );

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_budget_32_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_budget_32_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_32_truncated_arguments_keep_the_turn",
        |client| async move { assert_streaming_truncation_survives(&client, 32).await },
    )
    .await
    .expect(
        "streaming_budget_32_truncated_arguments_keep_the_turn should replay from its cassette",
    );

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_budget_48_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_budget_48_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_48_truncated_arguments_keep_the_turn",
        |client| async move { assert_streaming_truncation_survives(&client, 48).await },
    )
    .await
    .expect(
        "streaming_budget_48_truncated_arguments_keep_the_turn should replay from its cassette",
    );

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_budget_64_truncated_arguments_keep_the_turn() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_budget_64_truncated_arguments_keep_the_turn";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_64_truncated_arguments_keep_the_turn",
        |client| async move { assert_streaming_truncation_survives(&client, 64).await },
    )
    .await
    .expect(
        "streaming_budget_64_truncated_arguments_keep_the_turn should replay from its cassette",
    );

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_budget_96_complete_arguments_are_untouched() {
    const SCENARIO: &str = "truncation_matrix/streaming_budget_96_complete_arguments_are_untouched";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_budget_96_complete_arguments_are_untouched",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request(
                        &model,
                        TOOL_PREAMBLE,
                        vec![file_report_tool()],
                        non_thinking_params(),
                        96,
                    ))
                    .await?,
            )
            .await;
            assert_eq!(outcome.tool_call_names(), vec!["file_report"]);
            assert!(outcome.tool_calls[0].function.arguments["summary"].is_string());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_budget_96_complete_arguments_are_untouched should replay from its cassette");

    assert_parseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

// ================================================================
// C. Parallel calls: only the truncated one is lost
// ================================================================

#[tokio::test]
async fn blocking_parallel_calls_keep_the_complete_one() {
    const SCENARIO: &str = "truncation_matrix/blocking_parallel_calls_keep_the_complete_one";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_parallel_calls_keep_the_complete_one",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request(
                    &model,
                    PARALLEL_PREAMBLE,
                    vec![page_oncall_tool(), file_report_tool()],
                    json!({ "thinking": { "type": "disabled" }, "parallel_tool_calls": true }),
                    56,
                ))
                .await?
                .normalize("deepseek")?;

            let calls = tool_calls(&normalized.choice);
            assert_eq!(
                calls
                    .iter()
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["page_oncall"],
                "the complete call survives; only the truncated one is dropped: {:?}",
                normalized.choice
            );
            assert_eq!(calls[0].function.arguments, json!({ "team": "platform" }));
            assert_eq!(
                normalized.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_parallel_calls_keep_the_complete_one should replay from its cassette");

    let arguments = recorded_blocking_arguments(SCENARIO);
    assert_eq!(arguments.len(), 2, "premise: two calls were emitted");
    assert_parseable(&arguments[0], SCENARIO);
    assert_unparseable(&arguments[1], SCENARIO);
}

#[tokio::test]
async fn streaming_parallel_calls_keep_the_complete_one() {
    const SCENARIO: &str = "truncation_matrix/streaming_parallel_calls_keep_the_complete_one";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_parallel_calls_keep_the_complete_one",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request(
                        &model,
                        PARALLEL_PREAMBLE,
                        vec![page_oncall_tool(), file_report_tool()],
                        json!({ "thinking": { "type": "disabled" }, "parallel_tool_calls": true }),
                        56,
                    ))
                    .await?,
            )
            .await;
            assert_eq!(outcome.tool_call_names(), vec!["page_oncall"]);
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_parallel_calls_keep_the_complete_one should replay from its cassette");

    let arguments = recorded_stream_arguments(SCENARIO);
    assert_eq!(arguments.len(), 2, "premise: two calls were streamed");
    assert_parseable(&arguments[0], SCENARIO);
    assert_unparseable(&arguments[1], SCENARIO);
}

// ================================================================
// D. The turn spoke before it was cut: the text must survive too
// ================================================================

#[tokio::test]
async fn blocking_text_before_a_truncated_call_survives() {
    const SCENARIO: &str = "truncation_matrix/blocking_text_before_a_truncated_call_survives";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_text_before_a_truncated_call_survives",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request(
                    &model,
                    TEXT_FIRST_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    40,
                ))
                .await?
                .normalize("deepseek")?;

            assert!(
                !text(&normalized.choice).trim().is_empty(),
                "the assistant text the truncated call took down with it: {:?}",
                normalized.choice
            );
            assert!(tool_calls(&normalized.choice).is_empty());
            assert_eq!(
                normalized.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_text_before_a_truncated_call_survives should replay from its cassette");

    let response = recorded_response(SCENARIO);
    assert!(
        !response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .is_empty(),
        "premise: the recorded turn carried text beside the truncated call"
    );
    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_text_before_a_truncated_call_survives() {
    const SCENARIO: &str = "truncation_matrix/streaming_text_before_a_truncated_call_survives";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_text_before_a_truncated_call_survives",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request(
                        &model,
                        TEXT_FIRST_PREAMBLE,
                        vec![file_report_tool()],
                        non_thinking_params(),
                        40,
                    ))
                    .await?,
            )
            .await;
            assert!(!outcome.text.trim().is_empty(), "streamed text survives");
            assert!(outcome.tool_calls.is_empty());
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_text_before_a_truncated_call_survives should replay from its cassette");

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

// ================================================================
// E. Reasoner turns share the decode
// ================================================================

#[tokio::test]
async fn blocking_reasoner_truncated_call_keeps_the_reasoning_block() {
    const SCENARIO: &str =
        "truncation_matrix/blocking_reasoner_truncated_call_keeps_the_reasoning_block";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_reasoner_truncated_call_keeps_the_reasoning_block",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(request_for(
                    &model,
                    REASONER_INCIDENT_PROMPT,
                    REASONER_TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    thinking_params(),
                    112,
                ))
                .await?
                .normalize("deepseek")?;

            assert!(
                normalized
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "the reasoning block the truncated call took down with it: {:?}",
                normalized.choice
            );
            assert!(tool_calls(&normalized.choice).is_empty());
            assert!(normalized.usage.reasoning_tokens > 0);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_reasoner_truncated_call_keeps_the_reasoning_block should replay from its cassette");

    let response = recorded_response(SCENARIO);
    assert!(
        response["choices"][0]["message"]["reasoning_content"].is_string(),
        "premise: the recorded turn carried reasoning beside the truncated call"
    );
    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn streaming_reasoner_truncated_call_keeps_the_reasoning_block() {
    const SCENARIO: &str =
        "truncation_matrix/streaming_reasoner_truncated_call_keeps_the_reasoning_block";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_reasoner_truncated_call_keeps_the_reasoning_block",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request_for(
                        &model,
                        REASONER_INCIDENT_PROMPT,
                        REASONER_TOOL_PREAMBLE,
                        vec![file_report_tool()],
                        thinking_params(),
                        112,
                    ))
                    .await?,
            )
            .await;
            assert!(!outcome.reasoning.trim().is_empty());
            assert!(outcome.tool_calls.is_empty());
            assert_eq!(
                outcome.finish_reason(),
                Some(rig::completion::FinishReason::Length)
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_reasoner_truncated_call_keeps_the_reasoning_block should replay from its cassette");

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

// ================================================================
// F. Agent level: the loop sees a `Length` turn, not a failed request
// ================================================================

#[derive(Clone)]
struct FileReport {
    invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct FileReportArgs {
    summary: String,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct EmptyFileReportArgs {}

#[derive(Debug, thiserror::Error)]
#[error("file_report failed")]
struct FileReportError;

impl rig::tool::Tool for FileReport {
    const NAME: &'static str = "file_report";
    type Error = FileReportError;
    type Args = FileReportArgs;
    type Output = String;

    fn description(&self) -> String {
        "File an incident report.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A long verbatim summary of the incident.",
                },
            },
            "required": ["summary"],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.invocations
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(format!("filed: {}", args.summary))
    }
}

/// A real zero-argument side-effect tool for the empty-wire boundary. This is
/// intentionally separate from `FileReport`: `{}` cannot reach that tool's
/// `call` method because its required `summary` fails argument decoding, which
/// would make an invocation-count assertion pass for the wrong reason.
#[derive(Clone)]
struct ZeroArgumentFileReport {
    invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl rig::tool::Tool for ZeroArgumentFileReport {
    const NAME: &'static str = "file_report";
    type Error = FileReportError;
    type Args = EmptyFileReportArgs;
    type Output = String;

    fn description(&self) -> String {
        "File the incident now; this action takes no arguments.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.invocations
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok("filed".to_owned())
    }
}

#[tokio::test]
async fn agent_blocking_truncated_call_is_not_invoked() {
    const SCENARIO: &str = "truncation_matrix/agent_blocking_truncated_call_is_not_invoked";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_blocking_truncated_call_is_not_invoked",
        |client| async move {
            let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(TOOL_PREAMBLE)
                .tool(FileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(32)
                .default_max_turns(1)
                .build();

            // The turn is truncated, so the agent completes on the `Length`
            // terminal rather than dispatching a call it never fully received.
            //
            // Asserting the *outcome*, not just the invocation count: on
            // `origin/main` the response never decodes, so `prompt` fails and
            // the count is trivially zero. The cell only tests the fix if the
            // turn is required to have reached the loop at all.
            let outcome = rig::completion::Prompt::prompt(&agent, INCIDENT_PROMPT).await;
            let error = match outcome {
                Ok(_) => None,
                Err(error) => Some(error.to_string()),
            };
            if let Some(error) = &error {
                assert!(
                    !error.contains("ProviderResponseError"),
                    "the truncated turn must reach the agent loop, not fail the request: {error}"
                );
            }
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "a truncated call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("agent_blocking_truncated_call_is_not_invoked should replay from its cassette");

    assert_unparseable(&recorded_blocking_arguments(SCENARIO)[0], SCENARIO);
}

#[tokio::test]
async fn agent_streaming_truncated_call_is_not_invoked() {
    const SCENARIO: &str = "truncation_matrix/agent_streaming_truncated_call_is_not_invoked";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_streaming_truncated_call_is_not_invoked",
        |client| async move {
            let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(TOOL_PREAMBLE)
                .tool(FileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(32)
                .build();

            let mut stream = rig::streaming::StreamingChat::stream_chat(
                &agent,
                INCIDENT_PROMPT,
                Vec::<rig::completion::Message>::new(),
            )
            .max_turns(1)
            .await;
            let observation = crate::support::collect_stream_observation(&mut stream).await;

            assert!(
                observation.tool_calls.is_empty(),
                "the streamed truncated call must not surface: {:?}",
                observation.tool_calls
            );
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "a truncated call must never be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("agent_streaming_truncated_call_is_not_invoked should replay from its cassette");

    assert_unparseable(&recorded_stream_arguments(SCENARIO)[0], SCENARIO);
}

/// The empty-arguments boundary is safety-sensitive at agent level: without
/// consulting the outer `length` reason, `{}` would be dispatched to a
/// zero-argument side-effect tool even though generation ended before the
/// first argument token.
#[tokio::test]
async fn agent_blocking_empty_arguments_on_length_are_not_invoked() {
    const SCENARIO: &str =
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked",
        |client| async move {
            let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(TOOL_PREAMBLE)
                .tool(ZeroArgumentFileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(16)
                .default_max_turns(1)
                .build();

            let outcome = rig::completion::Prompt::prompt(&agent, INCIDENT_PROMPT).await;
            if let Err(error) = outcome {
                assert!(
                    !error.to_string().contains("ProviderResponseError"),
                    "the truncated turn must reach the agent loop: {error}"
                );
            }
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "the incomplete empty-argument call must not be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking empty-argument safety cell should replay");

    assert_eq!(recorded_blocking_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_blocking_finish_reason(SCENARIO), "length");
}

#[tokio::test]
async fn agent_streaming_empty_arguments_on_length_are_not_invoked() {
    const SCENARIO: &str =
        "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_streaming_empty_arguments_on_length_are_not_invoked",
        |client| async move {
            let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(TOOL_PREAMBLE)
                .tool(ZeroArgumentFileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(16)
                .build();

            let mut stream = rig::streaming::StreamingChat::stream_chat(
                &agent,
                INCIDENT_PROMPT,
                Vec::<rig::completion::Message>::new(),
            )
            .max_turns(1)
            .await;
            let observation = crate::support::collect_stream_observation(&mut stream).await;
            assert!(observation.tool_calls.is_empty());
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "the incomplete empty-argument call must not be dispatched"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming empty-argument safety cell should replay");

    assert_eq!(recorded_stream_arguments(SCENARIO), vec![String::new()]);
    assert_eq!(recorded_stream_finish_reason(SCENARIO), "length");
}

// ================================================================
// G. Wire-type decode, no recording needed
// ================================================================

/// The decode itself, exercised on this side of the crate boundary against the
/// exact bytes DeepSeek returned at the 24-token budget. A live recording
/// cannot force a *shape* the model does not happen to produce, and this cell
/// is about the type rather than the turn, so it is a unit cell in the matrix.
#[test]
fn a_truncated_call_is_dropped_at_decode_and_the_turn_survives() {
    let body = r#"{
        "id": "chatcmpl-truncated",
        "object": "chat.completion",
        "model": "deepseek-v4-flash",
        "choices": [{
            "index": 0,
            "logprobs": null,
            "finish_reason": "length",
            "message": {
                "role": "assistant",
                "content": "Acknowledged.",
                "tool_calls": [
                    {"index": 0, "id": "call_0", "type": "function", "function": {"name": "page_oncall", "arguments": "{\"team\": \"platform\"}"}},
                    {"index": 1, "id": "call_1", "type": "function", "function": {"name": "file_report", "arguments": "{\"summary\": "}}
                ]
            }
        }],
        "usage": {"prompt_tokens": 372, "completion_tokens": 24, "total_tokens": 396, "prompt_cache_hit_tokens": 256, "prompt_cache_miss_tokens": 116}
    }"#;

    let response: deepseek::CompletionResponse =
        serde_json::from_str(body).expect("a truncated turn must still decode");
    let deepseek::Message::Assistant {
        tool_calls: wire_calls,
        ..
    } = &response.choices[0].message;
    assert_eq!(
        wire_calls.len(),
        1,
        "only the truncated call is dropped: {wire_calls:?}"
    );
    assert_eq!(wire_calls[0].function.name, "page_oncall");
    assert_eq!(
        wire_calls[0].function.arguments,
        json!({"team": "platform"})
    );

    let normalized = response.normalize("deepseek").expect("normalize");
    assert_eq!(
        normalized.finish_reason(),
        Some(rig::completion::FinishReason::Length)
    );
    assert_eq!(
        tool_calls_names(&normalized.choice),
        vec!["page_oncall"],
        "only the truncated call is dropped: {:?}",
        normalized.choice
    );
    assert_eq!(normalized.usage.total_tokens, 396);
    assert_eq!(normalized.usage.cached_input_tokens, 256);
}

fn tool_calls_names(choice: &[AssistantContent]) -> Vec<&str> {
    tool_calls(choice)
        .into_iter()
        .map(|call| call.function.name.as_str())
        .collect()
}
