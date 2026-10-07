//! Raw provider response capture on OpenAI's streaming seams
//! (`CompletionResponse::raw`).
//!
//! # What this pins
//!
//! The streamed twin of `raw_capture_matrix`: every terminal
//! `StreamedAssistantContent::Final` carries `raw`, the unary document of
//! the route's API as the driver's reassembler rebuilt it from the reply's
//! frames. There is no switch behind it; a terminal `raw` is `Value::Null`
//! only on a record built by hand, never on one a stream yielded. A cell
//! reads it as JSON, by the paths a unary body is read by. It exposes a
//! field the normalized `CompletionResponse` does not model, and its
//! `usage`, finish reason, `model` and identity are the ones the stream
//! reported.
//!
//! On Chat Completions the document is a `chat.completion`: the top-level
//! chunk fields (`service_tier`, `system_fingerprint`) where a unary body
//! states them, the deltas folded into `choices[].message`, and the
//! provider's usage object, extra counters included. On the Responses API,
//! the response object, verbatim (whose `status` and message id come from
//! the terminal `response.completed` event alone).
//!
//! Cell 6 is the streamed twin of the tool-call cell in `raw_capture_matrix`:
//! a forced Chat tool call, whose document spells `finish_reason` as
//! `"tool_calls"` and whose normalized twin reports `FinishReason::ToolCalls`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_round_trips_typed` | chat, streamed | `raw` is the rebuilt `chat.completion`; its fields ≡ terminal | recorded |
//! | 4 | `responses_stream_raw_exposes_status` | Responses, terminal-only field | `raw["status"]` = `response.completed` status | recorded |
//! | 6 | `chat_tool_call_stream_raw_round_trips_typed` | chat, forced tool call (`tool_choice: required`) | `raw` is the rebuilt `chat.completion`; `raw["choices"][0]["finish_reason"] == "tool_calls"` = last finish chunk; normalized terminal reports `ToolCalls` | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture after the wrapper returns: the recorded stream ends with a
//! terminal frame carrying usage — Chat because the request the provider
//! sends already asks for `stream_options.include_usage`, Responses because
//! `response.completed` carries the whole response object. Cell 6 further
//! requires a chunk whose delta carries `tool_calls` and a chunk finishing
//! with `"tool_calls"`.

use rig::completion::CompletionRequest;
use rig::completion::CompletionResponse;
use rig::completion::FinishReason;
use rig::completion::ToolDefinition;
use rig::message::ToolChoice;
use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::{sse_json_frames, with_openai_cassette_result};
use crate::raw_capture::chat;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal, responses};
use crate::support::normalized_without_raw;
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "Call ping exactly once with no arguments.";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
}

fn ping_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("ping").expect("tool name"),
        description: "Matrix tool ping".to_owned(),
        parameters: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
    }
}

/// The forced tool call `raw_completion_parity_matrix` records: `required`
/// leaves the model no text-only exit.
fn tool_request() -> CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .tool(ping_tool())
        .tool_choice(ToolChoice::Required)
        .temperature(0.0)
        .max_tokens(64)
}

/// Chat premise: the request asked for usage on the stream and the last
/// recorded frame carries it. Returns the frames.
fn chat_frames_with_terminal_usage(scenario: &str, request: &str, body: &str) -> Vec<Value> {
    let request: Value = serde_json::from_str(request).expect("recorded request should be JSON");
    assert_eq!(
        request["stream_options"]["include_usage"], true,
        "{scenario}: the chat stream request asks for terminal usage"
    );
    let frames = sse_json_frames(body);
    let last = frames
        .last()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream must carry frames"));
    assert!(
        last["usage"].is_object(),
        "{scenario}: the last recorded chat frame must carry usage — without a terminal \
         frame this cell asserts nothing about the terminal record"
    );
    frames
}

/// Responses premise: the last recorded frame is `response.completed` with
/// usage on its response object. Returns that response object.
fn responses_completed_frame(scenario: &str, body: &str) -> Value {
    let frames = sse_json_frames(body);
    let last = frames
        .last()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream must carry frames"));
    assert_eq!(
        last["type"], "response.completed",
        "{scenario}: the recorded stream must end with response.completed"
    );
    assert!(
        last["response"]["usage"].is_object(),
        "{scenario}: the terminal response object must carry usage"
    );
    last["response"].clone()
}

fn last_chunk_field(frames: &[Value], field: &str) -> Value {
    frames
        .iter()
        .filter_map(|frame| frame.get(field))
        .next_back()
        .cloned()
        .unwrap_or(Value::Null)
}

/// The `raw` a streamed terminal must carry — `Value::Null` is reserved for
/// records built by hand, which a terminal off a live stream never is.
fn captured_raw<'a>(scenario: &str, terminal: &'a CompletionResponse) -> &'a Value {
    assert!(
        !terminal.raw.is_null(),
        "{scenario}: a streamed terminal always carries `raw`"
    );
    &terminal.raw
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed",
        |client| capture_terminal(client.openai.chat(MODEL), request(), observed.clone()),
    )
    .await
    .expect("chat_stream_raw_round_trips_typed should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);

    // `raw` is there to read at all: it is the terminal response object.
    captured_raw(SCENARIO, &terminal);
    // It is the rebuilt `chat.completion`, read as JSON. It agrees with the
    // normalized terminal on identity, model, finish reason and the
    // accounting it normalized: two views of one record.
    let typed = chat::assert_terminal_round_trips(&terminal);
    // The captured value is *this* stream's terminal.
    assert_matches_recorded_token(
        typed["id"].as_str(),
        last_chunk_field(&frames, "id").as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed["model"].as_str(),
        last_chunk_field(&frames, "model").as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(
        typed["choices"][0]["finish_reason"],
        serde_json::json!("stop")
    );
    let recorded_usage = last_chunk_field(&frames, "usage");
    let usage = &typed["usage"];
    assert_eq!(
        usage["prompt_tokens"].as_u64(),
        recorded_usage["prompt_tokens"].as_u64(),
        "{SCENARIO}: terminal prompt tokens"
    );
    assert_eq!(
        usage["completion_tokens"].as_u64(),
        recorded_usage["completion_tokens"].as_u64(),
        "{SCENARIO}: terminal completion tokens"
    );
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_stream_raw_exposes_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/responses_stream_raw_exposes_status";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/responses_stream_raw_exposes_status",
        |client| capture_terminal(client.openai.completion(MODEL), request(), observed.clone()),
    )
    .await
    .expect("responses_stream_raw_exposes_status should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let completed = responses_completed_frame(SCENARIO, &bodies[0].1);
    let recorded_status = completed["status"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the terminal response object must carry a string `status`")
    });
    let recorded_message_id = completed["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "message"))
        .and_then(|item| item["id"].as_str())
        .unwrap_or_else(|| {
            panic!("{SCENARIO}: the terminal response object must carry a message output item")
        });

    let raw = captured_raw(SCENARIO, &terminal);
    assert_eq!(
        raw["status"].as_str(),
        Some(recorded_status),
        "{SCENARIO}: `status` is readable off the captured terminal and equals the \
         response.completed status"
    );
    let captured_message_id = raw["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "message"))
        .and_then(|item| item["id"].as_str());
    assert_matches_recorded_token(
        captured_message_id,
        Some(recorded_message_id),
        &format!("{SCENARIO}: `message_id` off the captured response vs the fixture"),
    );
    // `message_id` *is* normalized; the captured response and the
    // normalized one must agree on it.
    assert_eq!(
        captured_message_id,
        responses::message_item_id(&terminal),
        "{SCENARIO}: captured and normalized message ids agree"
    );
    assert_normalized_lacks(&normalized_without_raw(terminal.clone()), &["status"]);
}

// ---------------------------------------------------------------------------
// Reasoning and tool-call streams
// ---------------------------------------------------------------------------

/// A forced Chat tool-call stream: the rebuilt document agrees with the
/// normalized terminal, and `raw` spells `finish_reason` as OpenAI's own
/// `"tool_calls"`, the same word the last finishing chunk carried, while the
/// normalized terminal reports `FinishReason::ToolCalls`. Premise: a chunk's
/// delta carried `tool_calls`.
#[tokio::test]
async fn chat_tool_call_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed",
        |client| capture_terminal(client.openai.chat(MODEL), tool_request(), observed.clone()),
    )
    .await
    .expect("chat_tool_call_stream_raw_round_trips_typed should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);
    // Premise: the request forced the tool, some chunk streamed a tool-call
    // delta naming it, and the finishing chunk said `tool_calls`.
    let request: Value =
        serde_json::from_str(&bodies[0].0).expect("recorded request should be JSON");
    assert_eq!(
        request["tool_choice"], "required",
        "{SCENARIO}: forced tool call"
    );
    let streamed_tool_names: Vec<&str> = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .filter_map(|call| call["function"]["name"].as_str())
        .collect();
    assert_eq!(
        streamed_tool_names,
        vec!["ping"],
        "{SCENARIO}: the recorded chunks stream one call to ping"
    );
    let recorded_finish = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["finish_reason"].as_str())
        .next_back()
        .unwrap_or_else(|| panic!("{SCENARIO}: a recorded chunk must carry a finish reason"));
    assert_eq!(
        recorded_finish, "tool_calls",
        "{SCENARIO}: the recorded stream finished on a tool call"
    );

    let raw = captured_raw(SCENARIO, &terminal);
    // Two views of one reply. The wire already said `tool_calls`, so the
    // `Stop -> ToolCalls` reconciliation the normalized stream layers on has
    // nothing to change and the round trip's comparison stays exact.
    let typed = chat::assert_terminal_round_trips(&terminal);
    assert_matches_recorded_token(
        typed["id"].as_str(),
        last_chunk_field(&frames, "id").as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        raw["choices"][0]["finish_reason"], recorded_finish,
        "{SCENARIO}: raw keeps OpenAI's own finish-reason spelling"
    );
    assert_eq!(
        typed["choices"][0]["finish_reason"],
        serde_json::json!("tool_calls")
    );
    assert_eq!(
        terminal.finish_reason(),
        Some(FinishReason::ToolCalls),
        "{SCENARIO}: the normalized terminal reports the tool call"
    );
    let recorded_usage = last_chunk_field(&frames, "usage");
    let usage = &typed["usage"];
    assert_eq!(
        usage["prompt_tokens"].as_u64(),
        recorded_usage["prompt_tokens"].as_u64(),
        "{SCENARIO}: terminal prompt tokens"
    );
}
