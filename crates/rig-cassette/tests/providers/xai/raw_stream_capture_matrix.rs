//! Raw provider response capture on xAI's streaming path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::completion::CompletionResponse::raw`] carries the provider-native terminal record
//! behind the stream's finished response — for xAI the Responses terminal
//! Responses response object,
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

use rig::completion::{CompletionRequest, FinishReason};
use rig::providers::xai;
use serde_json::{Value, json};

use super::support::with_xai_cassette_result;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal};
use crate::support::Observed;
use crate::support::normalized_without_raw;

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
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

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_status";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_status",
        |client| capture_terminal(client.completion(MODEL), request(), sink.clone()),
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
    assert_eq!(terminal.finish_reason(), Some(FinishReason::Stop));
    let normalized = normalized_without_raw(terminal.clone());
    assert_normalized_lacks(&normalized, &["status"]);
}
