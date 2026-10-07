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
//! turn that spoke before it was cut, and a reasoner turn cut mid-arguments.
//! See the module doc table in the PR body for per-cell status.

use rig::completion::ToolDefinition;
use rig::message::AssistantContent;
use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{
    collect_raw_stream_outcome, recorded_response, recorded_stream_chunks,
    with_deepseek_truncation_cassette_result,
};
use rig::completion::{CompletionRequest, GenerationOptions, Reasoning};

pub(super) const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

/// The tool's one required argument must be long enough that a small budget
/// lands inside the JSON string rather than after it.
pub(super) const TOOL_PREAMBLE: &str = "You must call the file_report tool. The summary argument must be a verbatim, complete restatement of the user's entire request, word for word, at least 120 words long.";
const PARALLEL_PREAMBLE: &str = "You must call page_oncall with team set to platform, and then file_report whose summary argument is a verbatim, complete restatement of the user request, word for word, at least 120 words long. Emit both calls in the same turn.";
pub(super) const INCIDENT_PROMPT: &str = "Log this incident: the nightly build broke because the cache warmer raced the artifact uploader, then the retry storm saturated the queue, and the on-call engineer had to drain three regions by hand while the dashboards lagged behind by nine minutes.";

/// Thinking off: `{"thinking":{"type":"disabled"}}` on DeepSeek.
pub(super) fn non_thinking() -> GenerationOptions {
    GenerationOptions::default().reasoning(Reasoning::Off)
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

/// A request with thinking off and the raw `params` (typed
/// `parallel_tool_calls` is refused on DeepSeek).
fn request(
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Option<Value>,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    request_for(INCIDENT_PROMPT, preamble, tools, params, max_tokens)
}

fn request_for(
    prompt: &str,
    preamble: &str,
    tools: Vec<ToolDefinition>,
    params: Option<Value>,
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    let mut builder = CompletionRequest::new(prompt)
        .preamble(preamble.to_owned())
        .options(non_thinking())
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

// ================================================================
// A. Blocking budget sweep
// ================================================================

#[tokio::test]
#[ignore = "deepseek-v4-flash now writes text first and spends the 16-token budget before any call, even for the original request bytes (3 live attempts, 2026-10-03)"]
async fn blocking_budget_16_empty_arguments_are_dropped_on_length() {
    const SCENARIO: &str =
        "truncation_matrix/blocking_budget_16_empty_arguments_are_dropped_on_length";
    with_deepseek_truncation_cassette_result(
        "truncation_matrix/blocking_budget_16_empty_arguments_are_dropped_on_length",
        |client| async move {
            let model = client.completion(MODEL);
            let normalized = model
                .call(request(TOOL_PREAMBLE, vec![file_report_tool()], None, 16))
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

// ================================================================
// B. Streaming budget sweep (the parity twin)
// ================================================================

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
                    Some(json!({ "parallel_tool_calls": true })),
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
                Some(json!({ "parallel_tool_calls": true })),
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

// ================================================================
// E. Reasoner turns share the decode
// ================================================================

// ================================================================
// G. Provider-type decode, no recording needed
// ================================================================
