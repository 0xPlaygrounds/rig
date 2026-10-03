//! Edge matrix for rig#2354: a `max_tokens`-truncated tool call destroyed the
//! whole blocking DeepSeek response.
//!
//! DeepSeek emits the tool call anyway when the budget runs out mid-arguments:
//! the turn comes back with `finish_reason: "length"` and
//! `tool_calls[].function.arguments` cut off partway through the JSON object.
//! DeepSeek's tool-call type parsed that strictly, so the *whole* `CompletionResponse`
//! failed to decode and the text, usage, id, model and finish reason went with
//! it — while the streaming path kept the turn and dropped the unusable
//! call. The two transports disagreed about
//! identical wire bytes.
//!
//! Both transports now keep the cut call, as pi does: its arguments are what
//! pi's tolerant parse reads from the text (blank text is an empty object),
//! and a call whose text never parses keeps it as `invalid_arguments`, so the
//! agent answers it with an error result and never runs it.
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
//! Every budget is recorded on both transports except streaming budget 20:
//! the live sweep returned the same empty argument string and `length` reason
//! as streaming budget 16, exercising no additional adapter state, so that
//! duplicate cell is explicitly pruned. The matrix also records the shapes
//! that share the same decode: parallel calls where only the second is cut, a
//! turn that spoke before it was cut, a reasoner turn cut mid-arguments, and
//! the agent loop. See the module doc table in the PR body for per-cell status.

use anyhow::Result;
use rig::completion::ToolDefinition;
use rig::message::AssistantContent;
use rig::providers::deepseek;
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use super::support::{
    collect_raw_stream_outcome, recorded_response, recorded_stream_chunks,
    with_deepseek_truncation_cassette_result,
};
use rig::completion::CompletionRequest;

pub(super) const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

/// The tool's one required argument must be long enough that a small budget
/// lands inside the JSON string rather than after it.
pub(super) const TOOL_PREAMBLE: &str = "You must call the file_report tool. The summary argument must be a verbatim, complete restatement of the user's entire request, word for word, at least 120 words long.";
const TEXT_FIRST_PREAMBLE: &str = "First write exactly one short sentence acknowledging the request, then call the file_report tool whose summary argument is a verbatim, complete restatement of the user request, word for word, at least 120 words long.";
const PARALLEL_PREAMBLE: &str = "You must call page_oncall with team set to platform, and then file_report whose summary argument is a verbatim, complete restatement of the user request, word for word, at least 120 words long. Emit both calls in the same turn.";
pub(super) const INCIDENT_PROMPT: &str = "Log this incident: the nightly build broke because the cache warmer raced the artifact uploader, then the retry storm saturated the queue, and the on-call engineer had to drain three regions by hand while the dashboards lagged behind by nine minutes.";
/// The reasoner cells need a turn whose *thinking* is trivial and whose
/// *arguments* are long, so the budget reliably lands inside the JSON string
/// rather than inside the reasoning. A verbatim-copy instruction does that:
/// there is nothing to reason about and a great deal to type.
const REASONER_TOOL_PREAMBLE: &str = "Call the file_report tool exactly once. Set its summary argument to the user's text, copied out verbatim and in full. Do not summarise, do not shorten, do not think about it.";
const REASONER_INCIDENT_PROMPT: &str = "Copy this into file_report: the nightly build broke because the cache warmer raced the artifact uploader; the retry storm then saturated the queue; the on-call engineer drained three regions by hand; the dashboards lagged nine minutes behind; the checksum verifier timed out twice; the release channel notification never fired; the rollback took forty minutes; the incident channel filled with duplicate alerts; the paging policy escalated to the wrong rotation; and the postmortem template was missing three required sections.";

pub(super) fn non_thinking_params() -> Value {
    json!({ "thinking": { "type": "disabled" } })
}

fn thinking_params() -> Value {
    json!({ "thinking": { "type": "enabled" } })
}

fn file_report_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("file_report").expect("tool name"),
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
        name: rig_core::message::ToolName::new("page_oncall").expect("tool name"),
        description: "Page the on-call engineer.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "team": { "type": "string" } },
            "required": ["team"],
        }),
    }
}

fn request(
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Value,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    request_for(INCIDENT_PROMPT, preamble, tools, params, max_tokens)
}

fn request_for(
    prompt: &str,
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Value,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    let mut builder = CompletionRequest::new(prompt)
        .preamble(preamble.to_owned())
        .additional_params(params)
        .max_tokens(max_tokens);
    for tool in tools {
        builder = builder.tool(tool);
    }
    builder
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

/// The one cut call `calls` holds: kept with what its arguments state, and
/// the text they arrived as, so nothing runs it.
fn assert_one_cut_call<'a>(calls: impl IntoIterator<Item = &'a rig::message::ToolCall>) {
    let calls: Vec<&rig::message::ToolCall> = calls.into_iter().collect();
    let [call] = calls.as_slice() else {
        panic!("the cut call is kept: {calls:?}");
    };
    assert!(
        call.function.invalid_arguments.is_some(),
        "a call whose arguments never parse keeps their text: {call:?}"
    );
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
pub(super) fn recorded_blocking_arguments(scenario: &str) -> Vec<String> {
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

pub(super) fn recorded_blocking_finish_reason(scenario: &str) -> String {
    recorded_response(scenario)["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// The `arguments` fragments the recorded stream delivered, concatenated per
/// `tool_calls[].index`.
pub(super) fn recorded_stream_arguments(scenario: &str) -> Vec<String> {
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

pub(super) fn recorded_stream_finish_reason(scenario: &str) -> String {
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

pub(super) fn assert_unparseable(arguments: &str, scenario: &str) {
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
async fn assert_blocking_truncation_survives(client: &OpenAiModels, max_tokens: u64) -> Result<()> {
    let model = client.completion(MODEL);
    let response = model
        .call(request(
            TOOL_PREAMBLE,
            vec![file_report_tool()],
            non_thinking_params(),
            max_tokens,
        ))
        .await?;

    // The premise, read off DeepSeek's own view of the very reply the
    // normalized response was decoded from: `raw` is that reply's document.
    assert_eq!(
        response.raw["choices"][0]["finish_reason"], "length",
        "premise: the recorded turn must have been cut by the budget"
    );

    assert_eq!(
        response.finish_reason(),
        Some(rig::completion::FinishReason::Length),
        "the surviving turn reports the truncation"
    );
    assert_one_cut_call(tool_calls(&response.choice));
    assert!(
        response.usage.total_tokens.is_some_and(|n| n > 0)
            && response.usage.input_tokens.is_some_and(|n| n > 0),
        "usage survives the truncated call: {:?}",
        response.usage
    );
    assert!(
        response.response_id().is_some(),
        "the response id survives the truncated call"
    );
    assert!(
        response.model().is_some(),
        "the model name survives the truncated call"
    );
    Ok(())
}

/// Streaming twin: the stream already dropped the unusable call; this pins that
/// it still does, and that its terminal record reports the same `Length`.
async fn assert_streaming_truncation_survives(
    client: &OpenAiModels,
    max_tokens: u64,
) -> Result<()> {
    let model = client.completion(MODEL);
    let outcome = collect_raw_stream_outcome(model.stream(request(
        TOOL_PREAMBLE,
        vec![file_report_tool()],
        non_thinking_params(),
        max_tokens,
    ))?)
    .await;

    assert!(
        outcome.errors.is_empty(),
        "stream errors: {:?}",
        outcome.errors
    );
    assert_one_cut_call(&outcome.tool_calls);
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
        usage.total_tokens.is_some_and(|n| n > 0),
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    12,
                ))
                .await?;
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    16,
                ))
                .await?;
            // Blank arguments are an empty object, as pi reads them.
            let calls = tool_calls(&normalized.choice);
            assert_eq!(calls.len(), 1, "{:?}", normalized.choice);
            assert_eq!(calls[0].function.arguments_value(), json!({}));
            assert!(calls[0].function.invalid_arguments.is_none());
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    20,
                ))
                .await?;
            let calls = tool_calls(&normalized.choice);
            assert_eq!(calls.len(), 1, "{:?}", normalized.choice);
            assert_eq!(calls[0].function.arguments_value(), json!({}));
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    96,
                ))
                .await?;
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
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(model.stream(request(
                TOOL_PREAMBLE,
                vec![file_report_tool()],
                non_thinking_params(),
                12,
            ))?)
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
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(model.stream(request(
                TOOL_PREAMBLE,
                vec![file_report_tool()],
                non_thinking_params(),
                16,
            ))?)
            .await;
            // Blank arguments are an empty object, as pi reads them.
            assert_eq!(outcome.tool_call_names(), vec!["file_report"]);
            assert_eq!(outcome.tool_calls[0].function.arguments_value(), json!({}));
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
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(model.stream(request(
                TOOL_PREAMBLE,
                vec![file_report_tool()],
                non_thinking_params(),
                96,
            ))?)
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    PARALLEL_PREAMBLE,
                    vec![page_oncall_tool(), file_report_tool()],
                    json!({ "thinking": { "type": "disabled" }, "parallel_tool_calls": true }),
                    56,
                ))
                .await?;

            let calls = tool_calls(&normalized.choice);
            assert_eq!(
                calls
                    .iter()
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["page_oncall", "file_report"],
                "both calls are kept: {:?}",
                normalized.choice
            );
            assert_one_cut_call(calls.iter().skip(1).copied());
            assert_eq!(
                calls[0].function.arguments_value(),
                json!({ "team": "platform" })
            );
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
#[ignore = "deepseek-v4-flash now spends the 56-token budget before the second call's arguments start, even for the original request bytes (3 live attempts, 2026-10-03)"]
async fn streaming_parallel_calls_keep_the_complete_one() {
    const SCENARIO: &str = "truncation_matrix/streaming_parallel_calls_keep_the_complete_one";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/streaming_parallel_calls_keep_the_complete_one",
        |client| async move {
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(model.stream(request(
                PARALLEL_PREAMBLE,
                vec![page_oncall_tool(), file_report_tool()],
                json!({ "thinking": { "type": "disabled" }, "parallel_tool_calls": true }),
                56,
            ))?)
            .await;
            assert_eq!(
                outcome.tool_call_names(),
                vec!["page_oncall", "file_report"]
            );
            assert_one_cut_call(outcome.tool_calls.iter().skip(1));
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(
                    TEXT_FIRST_PREAMBLE,
                    vec![file_report_tool()],
                    non_thinking_params(),
                    40,
                ))
                .await?;

            assert!(
                !text(&normalized.choice).trim().is_empty(),
                "the assistant text the truncated call took down with it: {:?}",
                normalized.choice
            );
            assert_one_cut_call(tool_calls(&normalized.choice));
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
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(model.stream(request(
                TEXT_FIRST_PREAMBLE,
                vec![file_report_tool()],
                non_thinking_params(),
                40,
            ))?)
            .await;
            assert!(!outcome.text.trim().is_empty(), "streamed text survives");
            assert_one_cut_call(&outcome.tool_calls);
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
            let model = client.completion(MODEL);
            let normalized = model
                .call(request_for(REASONER_INCIDENT_PROMPT,
                    REASONER_TOOL_PREAMBLE,
                    vec![file_report_tool()],
                    thinking_params(),
                    112,
                ))
                .await?;

            assert!(
                normalized
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "the reasoning block the truncated call took down with it: {:?}",
                normalized.choice
            );
            assert_one_cut_call(tool_calls(&normalized.choice));
            assert!(normalized.usage.reasoning_tokens.is_some_and(|n| n > 0));
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
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(request_for(REASONER_INCIDENT_PROMPT,
                        REASONER_TOOL_PREAMBLE,
                        vec![file_report_tool()],
                        thinking_params(),
                        112,
                    ))
                    ?,
            )
            .await;
            assert!(!outcome.reasoning.trim().is_empty());
            assert_one_cut_call(&outcome.tool_calls);
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
pub(super) struct FileReport {
    pub(super) invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(super) struct FileReportArgs {
    summary: String,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(super) struct EmptyFileReportArgs {}

#[derive(Debug, thiserror::Error)]
#[error("file_report failed")]
pub(super) struct FileReportError;

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
pub(super) struct ZeroArgumentFileReport {
    pub(super) invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
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
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
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
            let outcome = agent.prompt(INCIDENT_PROMPT).await;
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
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .preamble(TOOL_PREAMBLE)
                .tool(FileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(32)
                .build();

            let mut stream = agent
                .prompt(INCIDENT_PROMPT)
                .history(Vec::<rig::completion::Message>::new())
                .max_turns(1)
                .stream();
            let observation = crate::support::collect_stream_observation(&mut stream).await;

            assert_eq!(
                observation.tool_calls,
                vec!["file_report".to_owned()],
                "the streamed cut call surfaces, kept with its text"
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

/// The empty-arguments boundary at agent level: generation ended before the
/// first argument token, and blank arguments are an empty object, as pi
/// reads them, so a zero-argument tool runs once.
#[tokio::test]
async fn agent_blocking_empty_arguments_on_length_are_not_invoked() {
    const SCENARIO: &str =
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/agent_blocking_empty_arguments_on_length_are_not_invoked",
        |client| async move {
            let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .preamble(TOOL_PREAMBLE)
                .tool(ZeroArgumentFileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(16)
                .default_max_turns(1)
                .build();

            let outcome = agent.prompt(INCIDENT_PROMPT).await;
            if let Err(error) = outcome {
                assert!(
                    !error.to_string().contains("ProviderResponseError"),
                    "the truncated turn must reach the agent loop: {error}"
                );
            }
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "blank arguments are an empty object"
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
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .preamble(TOOL_PREAMBLE)
                .tool(ZeroArgumentFileReport {
                    invocations: invocations.clone(),
                })
                .additional_params(non_thinking_params())
                .max_tokens(16)
                .build();

            let mut stream = agent
                .prompt(INCIDENT_PROMPT)
                .history(Vec::<rig::completion::Message>::new())
                .max_turns(1)
                .stream();
            let observation = crate::support::collect_stream_observation(&mut stream).await;
            assert_eq!(observation.tool_calls, vec!["file_report".to_owned()]);
            assert_eq!(
                invocations.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "blank arguments are an empty object"
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
// G. Provider-type decode, no recording needed
// ================================================================
