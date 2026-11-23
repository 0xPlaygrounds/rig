//! Matrix for raw terminal-record capture on ChatGPT's streaming `/responses` path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! Capture is always on. The terminal record of every stream carries `raw`: the response object
//! the terminal `response.completed` event carried, with its usage, status, ids and model,
//! verbatim. `raw == Value::Null` means only that a `CompletionResponse` was built by hand.
//! `stream_raw_terminal_round_trips_provider_type` checks `raw` agrees with the normalized
//! fields, and `stream_raw_exposes_terminal_status` reads back the provider's `status`, which
//! the normalized response folds into a finish reason and does not carry.
//!
//! The cells are ignored until recorded with ChatGPT credentials; a fixture is never
//! fabricated.

use super::GPT_5_4;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{recorded_interaction_bodies, recorded_sse_json_frames};
use crate::raw_capture::{assert_normalized_lacks, capture_terminal, responses};
use crate::support::Observed;
use crate::support::normalized_without_raw;
use rig::completion::CompletionRequest;

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose SSE stream ends with a `response.completed` frame
/// carrying usage. Returns its `response`.
///
/// Stays local: the usage a Responses stream reports hangs under a typed
/// event's `response` envelope, not on the frame itself, so neither
/// [`crate::raw_capture::chat::recorded_sole_usage_frame`]'s rule nor
/// [`crate::raw_capture::chat::recorded_agreeing_usage_frames`]' describes
/// this wire.
fn recorded_terminal_response(scenario: &str) -> Value {
    assert_eq!(
        recorded_interaction_bodies(CHATGPT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let frames = recorded_sse_json_frames(CHATGPT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must end with response.completed")
        });
    let response = terminal["response"].clone();
    assert!(
        response.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the terminal frame must report usage — without it the \
         terminal record carries no usage and this cell proves nothing"
    );
    response
}

// ---------------------------------------------------------------------------
// 1: raw is the provider-native terminal record, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            capture_terminal(client.completion(MODEL), request(), sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let raw = &terminal.raw;
    // The Responses stream's terminal record is its own document, so this is the
    // Responses contract's round trip, not `chat::assert_terminal_round_trips`,
    // which speaks the chat-completions terminal.
    let typed = responses::assert_terminal_round_trips(&terminal);
    // The shared round trip compares the whole normalized accounting, which a
    // terminal carrying no usage at all would satisfy by both sides being
    // empty; this cell's premise is that the record has usage.
    assert!(
        typed["usage"].is_object(),
        "the Responses terminal carries usage"
    );

    let recorded = recorded_terminal_response(scenario);
    assert_eq!(
        raw["usage"]["total_tokens"], recorded["usage"]["total_tokens"],
        "raw usage must be the terminal frame's usage"
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only field
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_raw_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_terminal_status";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_status",
        |client| async move {
            capture_terminal(client.completion(MODEL), request(), sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    assert_normalized_lacks(&normalized_without_raw(terminal.clone()), &["status"]);

    let raw = &terminal.raw;
    let recorded = recorded_terminal_response(scenario);
    assert_eq!(
        recorded["status"],
        Value::String("completed".to_string()),
        "{scenario}: premise — the recorded terminal frame is completed"
    );
    assert_eq!(raw["status"], recorded["status"]);
    assert_eq!(raw["usage"], recorded["usage"]);
    let typed = responses::assert_terminal_round_trips(&terminal);
    assert_eq!(typed["status"], "completed");
}
