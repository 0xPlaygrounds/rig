//! Feature matrix for opt-in raw provider response capture on the Gemini
//! Interactions API streaming seam (`POST /v1beta/interactions?alt=sse`).
//!
//! # The feature
//!
//! [`rig::completion::CompletionRequest::capture_raw_response`] read before
//! `raw_stream` and handed to `normalize_stream`: when set, the terminal
//! [`rig::streaming::StreamFinal`] carries, serialized, the value the inherent
//! `raw_stream` would have yielded as its `FinalResponse` — the Interactions
//! API's own [`StreamingCompletionResponse`] terminal record
//! (`map_stream_final`'s input, built from the `interaction.completed` event).
//! Off (the default) it stays `None`; the flag never reaches the wire.
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
//! | 1 | `flag_off_leaves_terminal_raw_unset` | default (`false`) | `StreamFinal.raw.is_none()` | recorded |
//! | 2 | `flag_on_roundtrips_streaming_completion_response` | `true` → typed access | `StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_terminal_only_fields` | `true` → un-normalized terminal fields | `interaction.status` spelled `"completed"`, `interaction.object`, `usage.total_tokens` == completed event, absent from the normalized terminal | recorded |
//! | 4 | `request_bytes_invariant_across_flag` | request boundary | recorded off/on request bodies byte-identical | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain streaming interactions route.
//!
//! Cell 4 records one scenario with **two** interactions — the flag-off stream
//! first, then its flag-on twin — because the invariant is between the two;
//! the harness replays interactions in order.
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

type Model = InteractionsCompletionModel<reqwest::Client>;

fn request(model: &Model, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .capture_raw_response(capture)
        .build()
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

fn assert_request_body_never_names_the_flag(scenario: &str, body: &str) {
    for spelling in ["capture_raw_response", "captureRawResponse"] {
        assert!(
            !body.contains(spelling),
            "{scenario}: the recorded request body must not carry {spelling:?}; the flag is \
             `#[serde(skip)]` local policy and must never reach Gemini"
        );
    }
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
// 1: default off
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_off_leaves_terminal_raw_unset() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/flag_off_leaves_terminal_raw_unset";
    let observed_total: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed_total);
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/flag_off_leaves_terminal_raw_unset",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model, false)).await;

            assert!(
                terminal.raw.is_none(),
                "capture was not requested, so the terminal record must not carry raw"
            );
            assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
            assert_eq!(terminal.model.as_deref(), Some(MODEL));
            *sink.lock().expect("observation lock") = Some(terminal.usage.total_tokens);
        },
    )
    .await;

    let completed = recorded_completed_event(SCENARIO);
    assert_eq!(
        observed_total.lock().expect("observation lock").take(),
        completed
            .pointer("/interaction/usage/total_tokens")
            .and_then(Value::as_u64),
        "{SCENARIO}: the terminal usage should be the completed event's total"
    );
    let (request_body, _) = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_request_body_never_names_the_flag(SCENARIO, &request_body);
}

// ---------------------------------------------------------------------------
// 2: on → typed access is recoverable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_roundtrips_streaming_completion_response() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/flag_on_roundtrips_streaming_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/flag_on_roundtrips_streaming_completion_response",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model, true)).await;

            let raw = terminal
                .raw
                .as_deref()
                .expect("capture was requested, so the terminal must carry raw");

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
// 3: on → terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_terminal_only_fields() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/flag_on_exposes_terminal_only_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/flag_on_exposes_terminal_only_fields",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model, true)).await;

            let raw = terminal
                .raw
                .as_deref()
                .expect("capture was requested, so the terminal must carry raw");
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

// ---------------------------------------------------------------------------
// 4: the request boundary never sees the flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_bytes_invariant_across_flag() {
    const SCENARIO: &str =
        "interactions_raw_stream_capture_matrix/request_bytes_invariant_across_flag";
    with_gemini_interactions_cassette(
        "interactions_raw_stream_capture_matrix/request_bytes_invariant_across_flag",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = stream_to_terminal(&model, request(&model, false)).await;
            let on = stream_to_terminal(&model, request(&model, true)).await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            assert_eq!(off.finish_reason, on.finish_reason);
            assert_eq!(off.model, on.model);
            assert_eq!(off.provider, on.provider);
            assert_eq!(off.usage.input_tokens, on.usage.input_tokens);
        },
    )
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: the cell records the flag-off stream and then its flag-on twin"
    );
    let (off_request, _) = &bodies[0];
    let (on_request, _) = &bodies[1];
    assert_eq!(
        off_request, on_request,
        "the flag-on request must be byte-identical to the flag-off request; the flag is local \
         policy and never reaches Gemini"
    );
    assert_request_body_never_names_the_flag(SCENARIO, on_request);
}
