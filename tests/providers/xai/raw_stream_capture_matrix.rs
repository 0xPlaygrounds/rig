//! Raw provider response capture on xAI's streaming path.
//!
//! **The feature.** With `CompletionRequest::capture_raw_response` on, the
//! stream's terminal [`rig::streaming::StreamFinal::raw`] carries the value
//! the model's inherent `raw_stream` would have yielded as its terminal
//! record — for xAI the Responses terminal [`StreamingCompletionResponse`],
//! built from the `response.completed` event — serialized. It is the terminal
//! record only, never the stream's events. The terminal carries the response
//! `status` the normalized terminal folds into a finish reason, so `status`
//! is the terminal-only field pinned here as reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_leaves_raw_none` | flag off (the default) | terminal `raw == None` | recorded |
//! | 2 | `stream_capture_on_round_trips_terminal_type` | flag on | terminal `raw` deserializes into the Responses `StreamingCompletionResponse` and re-serializes equal | recorded |
//! | 3 | `stream_capture_on_exposes_terminal_status` | terminal-only field | `raw.status` and `raw.usage.output_tokens` equal the recorded `response.completed` event's | recorded |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that the recorded event stream ends with exactly one
//! `response.completed` event whose `response.usage` is populated — so the
//! raw terminal record is knowable from the bytes and a recording whose
//! stream stopped completing fails loudly instead of covering nothing.

use rig::completion::{CompletionModel, CompletionRequest, FinishReason};
use rig::prelude::*;
use rig::providers::openai::responses_api::streaming::StreamingCompletionResponse;
use rig::providers::xai;
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_xai_cassette_result,
};
use crate::support::collect_text_and_terminal;

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &xai::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .capture_raw_response(capture)
        .build()
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

fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_headers(scenario)[0]
        .iter()
        .find(|(name, _)| name == "x-request-id")
        .map(|(_, value)| value.clone())
}

fn assert_terminal_reproduces_event(
    terminal: &StreamFinal,
    response: &Value,
    request_id: Option<&str>,
    context: &str,
) {
    assert_eq!(terminal.provider, PROVIDER, "{context}: provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        response["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_eq!(
        terminal.model.as_deref(),
        response["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        response["status"],
        json!("completed"),
        "{context}: the recorded turn completed"
    );
    assert_eq!(
        terminal.finish_reason,
        Some(FinishReason::Stop),
        "{context}: finish reason"
    );
    assert_eq!(
        (
            terminal.usage.input_tokens,
            terminal.usage.output_tokens,
            terminal.usage.total_tokens
        ),
        (
            response["usage"]["input_tokens"].as_u64().expect("input"),
            response["usage"]["output_tokens"].as_u64().expect("output"),
            response["usage"]["total_tokens"].as_u64().expect("total"),
        ),
        "{context}: usage"
    );
    assert!(
        request_id.is_some(),
        "{context}: the recorded SSE response must carry x-request-id"
    );
    assert_matches_recorded_token(
        terminal.provider_request_id.as_deref(),
        request_id,
        &format!("{context}: request id"),
    );
}

// ================================================================
// 1. Off leaves the terminal raw None
// ================================================================

#[tokio::test]
async fn stream_capture_off_leaves_raw_none() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model.stream(request(&model, false)).await?;
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(
                terminal.raw.is_none(),
                "raw must stay None unless asked for"
            );
            assert!(!text.is_empty());
            *sink.lock().expect("observation lock") = Some(terminal);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_off_leaves_raw_none should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let response = recorded_completed_response(SCENARIO);
    assert_terminal_reproduces_event(
        &terminal,
        &response,
        recorded_request_id(SCENARIO).as_deref(),
        "flag off",
    );
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
    assert!(request_body.get("capture_raw_response").is_none());
}

// ================================================================
// 2. On round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_capture_on_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            let raw = terminal
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
            let typed = StreamingCompletionResponse::deserialize(raw)
                .expect("raw is the Responses streaming terminal");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed terminal serialized, nothing more"
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.message_id, terminal.message_id);
            assert_eq!(typed.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("observation lock") = Some(terminal);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_on_round_trips_terminal_type should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let response = recorded_completed_response(SCENARIO);
    assert_terminal_reproduces_event(
        &terminal,
        &response,
        recorded_request_id(SCENARIO).as_deref(),
        "flag on",
    );
}

// ================================================================
// 3. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_capture_on_exposes_terminal_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_status";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_status",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_on_exposes_terminal_status should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let response = recorded_completed_response(SCENARIO);
    let recorded_status = response["status"]
        .as_str()
        .expect("the completed event carries the response status");
    let recorded_output_tokens = response["usage"]["output_tokens"]
        .as_u64()
        .expect("the completed event carries output_tokens");

    let raw = terminal.raw.as_deref().expect("raw is captured");
    assert_eq!(raw["status"], json!(recorded_status));
    assert_eq!(raw["usage"]["output_tokens"], json!(recorded_output_tokens));
    // The normalized terminal folds the status into a finish reason and keeps
    // no status slot.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    assert!(normalized.get("status").is_none());
}

// ================================================================
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn stream_request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_xai_cassette_result(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let (_, off) =
                collect_text_and_terminal(model.stream(request(&model, false)).await?).await;
            let (_, on) =
                collect_text_and_terminal(model.stream(request(&model, true)).await?).await;
            assert!(off.expect("off terminal").raw.is_none());
            assert!(on.expect("on terminal").raw.is_some());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_request_invariant_off_vs_on should replay from its cassette");

    let interactions = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "one flag-off and one flag-on stream");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "the flag-off and flag-on request bodies must be byte-identical"
    );
    assert!(!interactions[0].0.contains("capture_raw"));
}
