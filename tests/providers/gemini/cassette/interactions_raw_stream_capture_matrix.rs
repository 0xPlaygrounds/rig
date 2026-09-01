//! Feature matrix for raw provider response capture on the Gemini
//! Interactions API streaming seam (`POST /v1beta/interactions?alt=sse`).
//!
//! # The feature
//!
//! Raw capture is always on: `normalize_stream` serializes the value the
//! inherent `raw_stream` yielded as its `FinalResponse` — the Interactions
//! API's own [`StreamingCompletionResponse`] terminal record
//! (`map_stream_final`'s input, built from the `interaction.completed` event)
//! — onto the terminal [`rig::streaming::StreamFinal::raw`]. There is no
//! opt-in and nothing about it reaches the wire; `raw` is `Value::Null` only
//! on a terminal constructed without a provider stream behind it, never
//! because capture "was not requested".
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
//! | 1 | `raw_roundtrips_streaming_completion_response` | typed access | `StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal and agrees with the normalized terminal | recorded |
//! | 2 | `raw_exposes_terminal_only_fields` | un-normalized terminal fields | `interaction.status` spelled `"completed"`, `interaction.object`, `usage.total_tokens` == completed event, absent from the normalized terminal | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain streaming interactions route.
//!
//! The "wire" side of each premise is the `interaction.completed` SSE frame:
//! it is the only frame that carries the finished interaction and its usage,
//! and it is what rig's terminal record is built from.

use futures::StreamExt;
use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::gemini::interactions_api::InteractionsCompletionModel;
use rig::providers::gemini::interactions_api::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_interactions_cassette;

const PROVIDER: &str = "gemini";
const MODEL: &str = "gemini-3-flash-preview";
const PROMPT: &str = "Reply with exactly this one word and nothing else: streamed";

type Model = InteractionsCompletionModel<rig::http_client::ReqwestClient>;

fn request(model: &Model) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// Drain a model stream and return its single terminal record.
async fn stream_to_terminal(
    model: &Model,
    request: rig::completion::CompletionRequest,
) -> StreamFinal {
    let mut stream = model.stream(request).await.expect("stream should open");
    let mut terminal = None;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should succeed") {
            StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
            StreamedAssistantContent::Final(final_record) => {
                assert!(
                    terminal.replace(final_record).is_none(),
                    "a stream yields exactly one terminal record"
                );
            }
            _ => {}
        }
    }
    assert!(!text.is_empty(), "the stream should have carried text");
    terminal.expect("stream should yield a terminal record")
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
// 1: typed access is recoverable
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
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

            let raw = &terminal.raw;

            let typed = StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into the Interactions streaming terminal type");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "StreamingCompletionResponse must round-trip through its own Serialize/Deserialize"
            );

            // The typed value agrees with the normalized terminal next to it.
            assert_eq!(typed.model_version, terminal.model);
            assert_eq!(
                typed
                    .interaction
                    .as_ref()
                    .map(|interaction| interaction.id.as_str()),
                terminal.response_id.as_deref()
            );
            assert_eq!(
                typed.usage.as_ref().and_then(|usage| usage.total_tokens),
                Some(terminal.usage.total_tokens)
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
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

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
                .remove("raw");
            assert!(!contains_key(&normalized, "object"));
            assert!(!contains_key(&normalized, "status"));
            assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
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
