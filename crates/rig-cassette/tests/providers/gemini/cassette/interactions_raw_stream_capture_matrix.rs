//! Feature matrix for raw provider response capture on the Gemini
//! Interactions API streaming seam (`POST /v1beta/interactions?alt=sse`).
//!
//! # The feature
//!
//! Raw capture is always on: the Interactions decoder builds its terminal
//! record from the `interaction.completed` event — a JSON object carrying the
//! finished interaction verbatim under `interaction`, its `usage` and the
//! `model_version` — and puts it on the terminal
//! [`rig::completion::CompletionResponse::raw`] its `finish` returns.
//! There is no opt-in and nothing about it reaches the wire; `raw` is
//! `Value::Null` only on a terminal constructed without a provider stream
//! behind it, never because capture "was not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the terminal record. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded stream must end with an
//! `interaction.completed` event carrying usage, or the terminal it asserts on
//! is not the one this matrix is about.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_roundtrips_streaming_completion_response` | document access | `raw` holds the completed interaction verbatim, and its `model_version`, `interaction.id` and `usage` agree with the normalized terminal | recorded |
//! | 2 | `raw_exposes_terminal_only_fields` | un-normalized terminal fields | `interaction.status` spelled `"completed"`, `interaction.object`, `usage.total_tokens` == completed event, absent from the normalized terminal | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain streaming interactions route.
//!
//! The "wire" side of each premise is the `interaction.completed` SSE frame:
//! it is the only frame that carries the finished interaction and its usage,
//! and it is what the decoder's terminal record is built from.

use futures::StreamExt;
use rig::completion::FinishReason;
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_interactions_cassette;
use rig::completion::CompletionRequest;
use rig::providers::gemini::interactions_api::Interactions;

const PROVIDER: &str = "gemini";
const MODEL: &str = "gemini-3-flash-preview";
const PROMPT: &str = "Reply with exactly this one word and nothing else: streamed";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).temperature(0.0)
}

/// Drain a model stream and return its single terminal record.
async fn stream_to_terminal(
    model: &rig::driver::Model<Interactions>,
    request: rig::completion::CompletionRequest,
) -> rig::completion::CompletionResponse {
    let mut stream = model.stream(request).expect("stream should open");
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: delta, .. }) =
            item.expect("stream item should succeed")
        {
            text.push_str(&delta)
        }
    }
    let terminal = stream
        .finish()
        .await
        .expect("stream should yield a terminal record");
    assert!(!text.is_empty(), "the stream should have carried text");
    terminal
}

/// The recorded `interaction.completed` frame — the premise every cell rests
/// on, and the wire source of the terminal record.
fn recorded_completed_event(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let completed = frames
        .iter()
        .find(|frame| {
            frame.get("event_type") == Some(&Value::String("interaction.completed".to_string()))
        })
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "{scenario}: the recorded stream should end with an interaction.completed \
                 event; without one the terminal this cell asserts on is not the shape under \
                 test"
            )
        });
    assert_eq!(
        completed.pointer("/interaction/status"),
        Some(&Value::String("completed".to_string())),
        "{scenario}: the completed event should carry the finished interaction"
    );
    assert!(
        completed
            .pointer("/interaction/usage/total_tokens")
            .and_then(Value::as_u64)
            .is_some_and(|total| total > 0),
        "{scenario}: the completed event should carry usage"
    );
    completed
}

fn contains_key(value: &Value, needle: &str) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| key == needle || contains_key(value, needle)),
        Value::Array(items) => items.iter().any(|item| contains_key(item, needle)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// 1: the terminal record is recoverable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_streaming_completion_response() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response",
        |client| async move {
            let model = client.interactions(MODEL);
            let terminal = stream_to_terminal(&model, request()).await;

            let raw = &terminal.raw;

            assert!(
                raw.get("interaction").is_some_and(Value::is_object),
                "raw must carry the Interactions streaming terminal's interaction, got {raw}"
            );

            // The record agrees with the normalized terminal next to it.
            assert_eq!(
                raw.get("model_version").and_then(Value::as_str),
                terminal.model()
            );
            assert_eq!(
                raw.pointer("/interaction/id").and_then(Value::as_str),
                terminal.response_id()
            );
            assert_eq!(
                raw.pointer("/usage/total_tokens").and_then(Value::as_u64),
                terminal.usage.total_tokens
            );
            *sink.lock().expect("observation lock") = Some(raw.clone());
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let completed = recorded_completed_event(SCENARIO);
    assert_eq!(
        raw.get("interaction"),
        completed.get("interaction"),
        "{SCENARIO}: the captured terminal holds the completed interaction verbatim"
    );
    assert_eq!(
        raw.pointer("/usage/total_tokens"),
        completed.pointer("/interaction/usage/total_tokens"),
        "{SCENARIO}: the captured terminal usage must be the completed event's total"
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_terminal_only_fields() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/raw_exposes_terminal_only_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/raw_exposes_terminal_only_fields",
        |client| async move {
            let model = client.interactions(MODEL);
            let terminal = stream_to_terminal(&model, request()).await;

            let raw = &terminal.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized terminal provably lacks these: `object` has no
            // normalized home and `status` reaches it only as rig's finish-reason
            // vocabulary.
            let mut normalized =
                serde_json::to_value(&terminal).expect("normalized terminal serializes");
            normalized
                .as_object_mut()
                .expect("terminal is an object")
                .shift_remove("raw");
            assert!(!contains_key(&normalized, "object"));
            assert!(!contains_key(&normalized, "status"));
            assert_eq!(terminal.finish_reason(), Some(FinishReason::Stop));
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let completed = recorded_completed_event(SCENARIO);
    assert_eq!(
        raw.pointer("/interaction/status"),
        completed.pointer("/interaction/status"),
        "raw keeps the API's own status spelling"
    );
    assert_eq!(
        raw.pointer("/interaction/object"),
        completed.pointer("/interaction/object"),
        "raw carries the interaction envelope's object tag"
    );
    assert_eq!(
        raw.pointer("/interaction/usage/total_tokens"),
        completed.pointer("/interaction/usage/total_tokens"),
        "raw carries the completed event's usage untouched"
    );
}
