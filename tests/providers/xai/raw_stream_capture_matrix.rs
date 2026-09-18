//! Raw provider response capture on xAI's streaming path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the provider-native terminal record
//! behind the stream's `StreamEvent::Final` — for xAI the Responses terminal
//! [`StreamingCompletionResponse`](rig::providers::openai::responses_api::streaming::StreamingCompletionResponse),
//! built from the `response.completed` event —
//! serialized. Capture is always on: there is no flag to request it, nothing
//! about it reaches the wire, and a `Value::Null` only ever means a terminal
//! built by hand with no provider record behind it. It is the terminal record
//! only, never the stream's events. The terminal carries the response `status`
//! the normalized terminal folds into a finish reason, so `status` is the
//! terminal-only field pinned here as reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` reads back as the Responses `StreamingCompletionResponse` and re-serializes equal, and that type's fields are the capture's; the normalized terminal reproduces the recorded `response.completed` event and `x-request-id` header | recorded |
//! | 2 | `stream_raw_exposes_terminal_status` | terminal-only field | `raw.status` and `raw.usage.output_tokens` equal the recorded `response.completed` event's | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that the recorded event stream ends with exactly one
//! `response.completed` event whose `response.usage` is populated — so the
//! raw terminal record is knowable from the bytes and a recording whose
//! stream stopped completing fails loudly instead of covering nothing. That
//! premise, and the comparison it feeds, are this file's own: a Responses
//! stream's terminal is reproduced from *one event's* `response` object
//! rather than from a reply body, so neither the premise reader nor
//! [`assert_terminal_reproduces_event`] is the shared body contract.

use rig::completion::{CompletionModel, CompletionRequest, FinishReason};
use rig::driver::Bound;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use rig::providers::xai;
use rig::streaming::StreamFinal;
use serde_json::{Value, json};

use super::support::with_xai_cassette_result;
use crate::cassettes::recorded_response_header;
use crate::raw_capture::{
    assert_contracted_request_id, assert_normalized_lacks, capture_terminal,
    capture_text_and_terminal, responses, stream_normalized_without_raw,
};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &Bound<OpenAiWire>) -> CompletionRequest {
    model.completion_request(PROMPT).build()
}

/// The `response` object of the single recorded `response.completed` event.
fn recorded_completed_response(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let mut completed = frames
        .iter()
        .filter(|frame| frame["type"] == "response.completed");
    let terminal = completed
        .next()
        .expect("the recorded stream must end with a response.completed event");
    assert!(
        completed.next().is_none(),
        "exactly one response.completed event"
    );
    assert!(
        terminal["response"]["usage"].is_object(),
        "the completed event must carry usage"
    );
    terminal["response"].clone()
}

/// The `x-request-id` the recorded SSE response carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_header(PROVIDER, scenario, 0, REQUEST_ID_HEADER)
}

/// The normalized terminal against the recorded `response.completed` event's
/// own `response` object.
///
/// Local to this file rather than the shared Responses contract: that one
/// reproduces a *reply body*, while a stream's terminal is reproduced from
/// one event's envelope, whose status is the event's own claim that the
/// stream completed.
fn assert_terminal_reproduces_event(terminal: &StreamFinal, response: &Value) {
    assert_eq!(terminal.provider, PROVIDER, "provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        response["id"].as_str(),
        "response id",
    );
    assert_eq!(
        terminal.model.as_deref(),
        response["model"].as_str(),
        "model"
    );
    assert_eq!(
        response["status"],
        json!("completed"),
        "the recorded turn completed"
    );
    assert_eq!(
        terminal.finish_reason,
        Some(FinishReason::Stop),
        "finish reason"
    );
    assert_eq!(
        (
            terminal.usage.input_tokens,
            terminal.usage.output_tokens,
            terminal.usage.total_tokens
        ),
        (
            response["usage"]["input_tokens"].as_u64(),
            response["usage"]["output_tokens"].as_u64(),
            response["usage"]["total_tokens"].as_u64(),
        ),
        "usage"
    );
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| capture_text_and_terminal(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let (text, terminal) = sink.take();
    assert!(!text.is_empty());
    let raw = &terminal.raw;
    // A streamed terminal has no single reply document behind it, so unlike
    // `CompletionResponse::raw` this capture IS the native record serialized
    // (`terminal_record`) — the round trip back through it must therefore be
    // exact, and the typed view's fields are the capture's fields.
    let typed = responses::assert_terminal_round_trips(&terminal);
    assert_eq!(
        typed.status,
        Some(responses_api::ResponseStatus::Completed),
        "the recorded stream completed"
    );
    assert_eq!(
        typed.usage.as_ref().map(|usage| usage.output_tokens),
        raw["usage"]["output_tokens"].as_u64(),
        "the typed view's usage is the capture's usage"
    );
    assert_eq!(
        typed.provider_request_id, None,
        "the transport id is stamped on the normalized terminal, not the native record"
    );

    let response = recorded_completed_response(SCENARIO);
    assert_terminal_reproduces_event(&terminal, &response);
    // xAI contracts `x-request-id`; the recorded header is the premise.
    assert_contracted_request_id(
        terminal.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_status";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_status",
        |client| capture_terminal(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_exposes_terminal_status should replay from its cassette");

    let terminal = sink.take();
    let response = recorded_completed_response(SCENARIO);
    let recorded_status = response["status"]
        .as_str()
        .expect("the completed event carries the response status");
    let recorded_output_tokens = response["usage"]["output_tokens"]
        .as_u64()
        .expect("the completed event carries output_tokens");

    let raw = &terminal.raw;
    assert_eq!(raw["status"], json!(recorded_status));
    assert_eq!(raw["usage"]["output_tokens"], json!(recorded_output_tokens));
    // The normalized terminal folds the status into a finish reason and keeps
    // no status slot.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(&normalized, &["status"]);
}
