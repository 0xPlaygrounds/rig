//! Raw provider response capture on OpenAI's streaming seams
//! (`StreamFinal::raw`).
//!
//! # What this pins
//!
//! The streamed twin of `raw_capture_matrix`: every terminal
//! `StreamedAssistantContent::Final` carries `raw` — the route's
//! provider-native terminal record, serialized at the decoder's
//! `final_record` seam. There is no switch behind it; a terminal `raw` is
//! `Value::Null` only on a record built by hand, never on one a stream
//! yielded. Unlike a unary `raw` — which is the provider's verbatim reply
//! document — a streamed `raw` is the record the decoder assembled from the
//! reply's frames, because no single frame is the terminal. So it
//! round-trips into that terminal type and re-serializes equal, it exposes a
//! terminal-only field the normalized `StreamFinal` does not model, and —
//! because capture is unconditional and must stay an escape hatch — running
//! the decoder's own mapper over the captured record reproduces the `usage`,
//! `finish_reason`, `model` and identity the stream reported.
//!
//! Terminal types: Chat Completions' `openai::wire::StreamingCompletionResponse`
//! (whose `additional_params` accumulates the unmodeled top-level chunk
//! fields — `service_tier`, `system_fingerprint` — and whose `ChatUsage`
//! keeps the dialect's extra usage counters), and the Responses API's
//! `openai::responses_api::streaming::StreamingCompletionResponse` (whose
//! `status` and `message_id` come from the terminal `response.completed`
//! event alone).
//!
//! Cells 5–6 are the streamed twins of the reasoning and tool-call cells in
//! `raw_capture_matrix`: a Responses reasoning stream, whose terminal
//! carries the `reasoning` echo of `response.completed` as
//! `reasoning_metadata` (a terminal-only field the normalized `StreamFinal`
//! does not model), and a forced Chat tool call, whose terminal spells
//! `finish_reason` as `"tool_calls"` and whose normalized twin reports
//! `FinishReason::ToolCalls`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_round_trips_typed` | chat, streamed | chat terminal type round trip; the decoder's mapper over `raw` ≡ terminal | recorded |
//! | 2 | `chat_stream_raw_exposes_service_tier` | chat, terminal-only field | `raw["additional_params"]["service_tier"]` = last chunk | recorded |
//! | 3 | `responses_stream_raw_round_trips_typed` | Responses, streamed | Responses terminal type round trip; its fields ≡ terminal | recorded |
//! | 4 | `responses_stream_raw_exposes_status` | Responses, terminal-only field | `raw["status"]` = `response.completed` status | recorded |
//! | 5 | `responses_reasoning_stream_raw_round_trips_typed` | Responses, reasoning stream (`reasoning: { effort, summary }`) | terminal round trip; `raw["reasoning_metadata"]` = `response.completed`'s `reasoning`; premise: a `reasoning` output item with `encrypted_content` | recorded |
//! | 6 | `chat_tool_call_stream_raw_round_trips_typed` | chat, forced tool call (`tool_choice: required`) | terminal round trip; `raw["finish_reason"] == "tool_calls"` = last finish chunk; normalized terminal reports `ToolCalls` | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture after the wrapper returns: the recorded stream ends with a
//! terminal frame carrying usage — Chat because the request the provider
//! sends already asks for `stream_options.include_usage`, Responses because
//! `response.completed` carries the whole response object. Cell 5 further
//! requires the completed response object to carry a `reasoning` output item
//! with a string `encrypted_content`; cell 6 requires a chunk whose delta
//! carries `tool_calls` and a chunk finishing with `"tool_calls"`.

use rig::completion::{CompletionModel, CompletionRequest, FinishReason, ToolDefinition};
use rig::message::ToolChoice;
use rig::providers::openai;
use rig::streaming::StreamFinal;
use serde_json::{Value, json};

use super::super::support::{sse_json_frames, with_openai_cassette_result};
use crate::raw_capture::{
    assert_normalized_lacks, capture_terminal, chat, responses, stream_normalized_without_raw,
};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
/// The reasoning cell needs a model that emits `reasoning` output items; the
/// same one the `reasoning_roundtrip` module records against.
const REASONING_MODEL: &str = openai::GPT_5_2;
const PROMPT: &str = "Reply with exactly the single word: pong";
/// A problem small enough to answer in a few tokens but not so trivial that
/// the model skips reasoning: at `effort: "low"` a one-step multiplication
/// came back with no `reasoning` output item at all, and a cell that cannot
/// find one fails on its premise.
const REASONING_PROMPT: &str = "A train leaves at 09:30 and travels 150 km at 60 km/h. \
     At what time does it arrive? Reply with only the time in HH:MM.";
const TOOL_PROMPT: &str = "Call ping exactly once with no arguments.";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

/// The reasoning request shape the `reasoning_roundtrip` module uses
/// (`effort: "medium"`), with a summary asked for; the provider adds
/// `reasoning.encrypted_content` to `include` on every reasoning request.
fn reasoning_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(REASONING_PROMPT)
        .additional_params(json!({
            "reasoning": { "effort": "medium", "summary": "auto" }
        }))
        .build()
}

fn ping_tool() -> ToolDefinition {
    ToolDefinition {
        name: "ping".to_owned(),
        description: "Matrix tool ping".to_owned(),
        parameters: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
    }
}

/// The forced tool call `raw_completion_parity_matrix` records: `required`
/// leaves the model no text-only exit.
fn tool_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .tool(ping_tool())
        .tool_choice(ToolChoice::Required)
        .temperature(0.0)
        .max_tokens(64)
        .build()
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
fn captured_raw<'a>(scenario: &str, terminal: &'a StreamFinal) -> &'a Value {
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
        |client| capture_terminal(client.openai.chat(MODEL), request, observed.clone()),
    )
    .await
    .expect("chat_stream_raw_round_trips_typed should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);

    // `raw` is there to read at all: it is the terminal record, serialized.
    captured_raw(SCENARIO, &terminal);
    // It reads back as the chat terminal type, re-serializes to exactly the
    // captured value, and agrees with the normalized terminal on identity,
    // model, finish reason and the accounting it normalized — two views of
    // one record.
    let typed = chat::assert_terminal_round_trips(&terminal);
    // The captured value is *this* stream's terminal.
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        last_chunk_field(&frames, "id").as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed.model.as_deref(),
        last_chunk_field(&frames, "model").as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(typed.finish_reason, Some(FinishReason::Stop));
    let recorded_usage = last_chunk_field(&frames, "usage");
    let usage = typed
        .usage
        .as_ref()
        .unwrap_or_else(|| panic!("{SCENARIO}: the terminal record carries the accounting"));
    assert_eq!(
        Some(usage.openai.prompt_tokens as u64),
        recorded_usage["prompt_tokens"].as_u64(),
        "{SCENARIO}: terminal prompt tokens"
    );
    assert_eq!(
        usage.openai.completion_tokens.map(|tokens| tokens as u64),
        recorded_usage["completion_tokens"].as_u64(),
        "{SCENARIO}: terminal completion tokens"
    );
}

#[tokio::test]
async fn chat_stream_raw_exposes_service_tier() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_stream_raw_exposes_service_tier";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/chat_stream_raw_exposes_service_tier",
        |client| capture_terminal(client.openai.chat(MODEL), request, observed.clone()),
    )
    .await
    .expect("chat_stream_raw_exposes_service_tier should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);
    let recorded_tier = last_chunk_field(&frames, "service_tier");
    assert!(
        recorded_tier.is_string(),
        "{SCENARIO}: the recorded chunks must report a `service_tier`"
    );

    let raw = captured_raw(SCENARIO, &terminal);
    assert_eq!(
        raw["additional_params"]["service_tier"], recorded_tier,
        "{SCENARIO}: `service_tier` is readable off the captured terminal and equals the \
         last chunk's"
    );
    assert_matches_recorded_token(
        raw["additional_params"]["system_fingerprint"].as_str(),
        last_chunk_field(&frames, "system_fingerprint").as_str(),
        &format!("{SCENARIO}: `system_fingerprint` off the captured terminal vs the fixture"),
    );
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(&normalized, &["additional_params", "service_tier"]);
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/responses_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/responses_stream_raw_round_trips_typed",
        |client| capture_terminal(client.openai.completion(MODEL), request, observed.clone()),
    )
    .await
    .expect("responses_stream_raw_round_trips_typed should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let completed = responses_completed_frame(SCENARIO, &bodies[0].1);

    // `raw` is there to read at all: it is the terminal record, serialized.
    captured_raw(SCENARIO, &terminal);
    // It reads back as the Responses terminal type, re-serializes to exactly
    // the captured value, and agrees with the normalized terminal on identity,
    // model and the accounting it normalized — two views of one record.
    let typed = responses::assert_terminal_round_trips(&terminal);
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        completed["id"].as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed.model.as_deref(),
        completed["model"].as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(
        typed.usage.as_ref().map(|usage| usage.input_tokens),
        completed["usage"]["input_tokens"].as_u64(),
        "{SCENARIO}: terminal input tokens"
    );
    assert_eq!(
        typed.usage.as_ref().map(|usage| usage.output_tokens),
        completed["usage"]["output_tokens"].as_u64(),
        "{SCENARIO}: terminal output tokens"
    );
    assert_eq!(
        typed.provider_request_id, None,
        "{SCENARIO}: the native record never carries the transport id"
    );
}

#[tokio::test]
async fn responses_stream_raw_exposes_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/responses_stream_raw_exposes_status";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/responses_stream_raw_exposes_status",
        |client| capture_terminal(client.openai.completion(MODEL), request, observed.clone()),
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
    assert_matches_recorded_token(
        raw["message_id"].as_str(),
        Some(recorded_message_id),
        &format!("{SCENARIO}: `message_id` off the captured terminal vs the fixture"),
    );
    // `message_id` *is* normalized (`StreamFinal::message_id`); the captured
    // terminal and the normalized record must agree on it.
    assert_eq!(
        raw["message_id"].as_str(),
        terminal.message_id.as_deref(),
        "{SCENARIO}: captured and normalized message ids agree"
    );
    assert_normalized_lacks(&stream_normalized_without_raw(&terminal), &["status"]);
}

// ---------------------------------------------------------------------------
// Reasoning and tool-call streams
// ---------------------------------------------------------------------------

/// A Responses reasoning stream: the terminal record round-trips, and the
/// `reasoning` echo of `response.completed` — which the normalized
/// `StreamFinal` does not model — is readable off `raw` as
/// `reasoning_metadata`. Premise: the completed response object carries a
/// `reasoning` output item with a string `encrypted_content`, i.e. this was a
/// reasoning turn on the wire and not merely a reasoning-configured request.
#[tokio::test]
async fn responses_reasoning_stream_raw_round_trips_typed() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/responses_reasoning_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/responses_reasoning_stream_raw_round_trips_typed",
        |client| {
            capture_terminal(
                client.openai.completion(REASONING_MODEL),
                reasoning_request,
                observed.clone(),
            )
        },
    )
    .await
    .expect("responses_reasoning_stream_raw_round_trips_typed should replay from its cassette");
    let terminal = observed.take();
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let completed = responses_completed_frame(SCENARIO, &bodies[0].1);
    // Premise: a reasoning request on the wire, answered with a reasoning
    // item carrying encrypted content, and a `reasoning` echo object.
    let request: Value =
        serde_json::from_str(&bodies[0].0).expect("recorded request should be JSON");
    assert_eq!(
        request["reasoning"],
        json!({ "effort": "medium", "summary": "auto" }),
        "{SCENARIO}: the recorded request carries the reasoning shape"
    );
    assert_eq!(
        request["include"],
        json!(["reasoning.encrypted_content"]),
        "{SCENARIO}: the provider asks for encrypted reasoning on every reasoning request"
    );
    let reasoning_item = completed["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "reasoning"))
        .unwrap_or_else(|| {
            panic!("{SCENARIO}: the completed response must carry a `reasoning` output item")
        });
    assert!(
        reasoning_item["encrypted_content"].is_string(),
        "{SCENARIO}: the reasoning item must carry a string `encrypted_content`"
    );
    let recorded_reasoning = completed["reasoning"].as_object().unwrap_or_else(|| {
        panic!("{SCENARIO}: the completed response echoes `reasoning` as an object")
    });
    assert_eq!(
        recorded_reasoning.get("effort"),
        Some(&json!("medium")),
        "{SCENARIO}: the echo reports the requested effort"
    );

    let raw = captured_raw(SCENARIO, &terminal);
    let typed = responses::assert_terminal_round_trips(&terminal);
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        completed["id"].as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed.model.as_deref(),
        completed["model"].as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(
        typed.usage.as_ref().map(|usage| usage.output_tokens),
        completed["usage"]["output_tokens"].as_u64(),
        "{SCENARIO}: terminal output tokens"
    );
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| usage.output_tokens_details.as_ref())
            .map(|details| details.reasoning_tokens),
        completed["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64(),
        "{SCENARIO}: terminal reasoning tokens"
    );
    // The terminal-only field: the completed event's `reasoning` echo.
    assert_eq!(
        raw["reasoning_metadata"],
        Value::Object(recorded_reasoning.clone()),
        "{SCENARIO}: `reasoning_metadata` off the captured terminal equals the completed \
         event's `reasoning`"
    );
    assert_normalized_lacks(
        &stream_normalized_without_raw(&terminal),
        &["reasoning_metadata"],
    );
    // The normalized terminal reports the reasoning tokens the completed
    // event carried.
    assert_eq!(
        terminal.usage.reasoning_tokens,
        completed["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64(),
        "{SCENARIO}: normalized reasoning tokens"
    );
}

/// A forced Chat tool-call stream: the terminal record round-trips, `raw`
/// spells `finish_reason` as OpenAI's own `"tool_calls"` — the same word the
/// last finishing chunk carried — and the normalized terminal reports
/// `FinishReason::ToolCalls`. Premise: a chunk's delta carried `tool_calls`.
#[tokio::test]
async fn chat_tool_call_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed",
        |client| capture_terminal(client.openai.chat(MODEL), tool_request, observed.clone()),
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
    // Two views of one record. The wire already said `tool_calls`, so the
    // `Stop -> ToolCalls` reconciliation the normalized stream layers on has
    // nothing to change and the round trip's comparison stays exact.
    let typed = chat::assert_terminal_round_trips(&terminal);
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        last_chunk_field(&frames, "id").as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        raw["finish_reason"], recorded_finish,
        "{SCENARIO}: raw keeps OpenAI's own finish-reason spelling"
    );
    assert_eq!(typed.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(
        terminal.finish_reason,
        Some(FinishReason::ToolCalls),
        "{SCENARIO}: the normalized terminal reports the tool call"
    );
    let recorded_usage = last_chunk_field(&frames, "usage");
    let usage = typed
        .usage
        .as_ref()
        .unwrap_or_else(|| panic!("{SCENARIO}: the terminal record carries the accounting"));
    assert_eq!(
        Some(usage.openai.prompt_tokens as u64),
        recorded_usage["prompt_tokens"].as_u64(),
        "{SCENARIO}: terminal prompt tokens"
    );
}
