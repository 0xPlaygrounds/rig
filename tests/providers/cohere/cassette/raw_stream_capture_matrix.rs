//! Feature matrix for raw provider response capture on the Cohere `/v2/chat`
//! streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: `normalize_stream` serializes the value the
//! inherent `raw_stream` yielded as its `FinalResponse` — Cohere's own
//! [`StreamingCompletionResponse`] terminal record, built from the
//! `message-end` event — onto the terminal [`rig::streaming::StreamFinal::raw`].
//! There is no opt-in and nothing about it reaches the wire; `raw` is
//! `Value::Null` only on a terminal constructed without a provider stream
//! behind it, never because capture "was not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the terminal record. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded stream must end with a `message-end` frame
//! whose `delta` carries usage and a `COMPLETE` finish reason, or the terminal
//! it asserts on is not the one this matrix is about.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_roundtrips_streaming_completion_response` | typed access | `StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal and agrees with the normalized terminal | recorded |
//! | 2 | `raw_exposes_terminal_only_fields` | un-normalized terminal fields | `finish_reason` spelled `"COMPLETE"`, `usage.billed_units.*` == `message-end` delta, absent from the normalized terminal | recorded |
//!
//! Every cell is recorded: `COHERE_API_KEY` was available and the seam under
//! test is the plain streaming `/v2/chat` route.
//!
//! The "wire" side of each premise is the `message-end` SSE frame: it is the
//! only frame carrying usage and the finish reason, and it is what rig's
//! terminal record is built from.

use futures::StreamExt;
use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::cohere;
use rig::providers::cohere::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: streamed";

fn request(model: &cohere::CompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

/// Drain a model stream and return its single terminal record.
async fn stream_to_terminal(
    model: &cohere::CompletionModel,
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

/// The recorded `message-end` frame's `delta` — the premise every cell rests
/// on, and the wire source of the terminal record.
fn recorded_message_end_delta(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let message_end = frames
        .iter()
        .find(|frame| frame.get("type") == Some(&Value::String("message-end".to_string())))
        .unwrap_or_else(|| {
            panic!(
                "{scenario}: the recorded stream should end with a message-end frame; without \
                 one the terminal this cell asserts on is not the shape under test"
            )
        });
    let delta = message_end
        .get("delta")
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: the recorded message-end should carry a delta"));
    assert_eq!(
        delta.get("finish_reason"),
        Some(&Value::String("COMPLETE".to_string())),
        "{scenario}: the recorded stream should have finished COMPLETE"
    );
    assert!(
        delta
            .pointer("/usage/billed_units/input_tokens")
            .and_then(Value::as_f64)
            .is_some(),
        "{scenario}: the recorded message-end should carry billed_units, the un-normalized \
         field cell 2 reads through `raw`"
    );
    delta
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

/// Cohere's counters are `f64` on rig's wire type, so a captured `6.0`
/// must be compared numerically against the fixture's `6`.
fn number_at(value: &Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer).and_then(Value::as_f64)
}

// ---------------------------------------------------------------------------
// 1: typed access is recoverable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_streaming_completion_response() {
    const SCENARIO: &str = "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

            let raw = &terminal.raw;

            let typed = StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into Cohere's streaming terminal type");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "cohere::streaming::StreamingCompletionResponse must round-trip through its own \
             Serialize/Deserialize"
            );

            // The typed value agrees with the normalized terminal next to it.
            assert_eq!(typed.message_id, terminal.response_id);
            assert_eq!(
                typed
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.tokens.as_ref())
                    .and_then(|tokens| tokens.input_tokens)
                    .map(|tokens| tokens as u64),
                Some(terminal.usage.input_tokens)
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
    let delta = recorded_message_end_delta(SCENARIO);
    assert_eq!(
        number_at(&raw, "/usage/tokens/input_tokens"),
        number_at(&delta, "/usage/tokens/input_tokens"),
        "{SCENARIO}: the captured terminal usage must be the message-end token counter"
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_terminal_only_fields() {
    const SCENARIO: &str = "raw_stream_capture_matrix/raw_exposes_terminal_only_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_stream_capture_matrix/raw_exposes_terminal_only_fields",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

            let raw = &terminal.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized terminal provably lacks these: billed units have no
            // normalized home, and the finish reason reaches it only as rig's
            // vocabulary.
            let mut normalized =
                serde_json::to_value(&terminal).expect("normalized terminal serializes");
            normalized
                .as_object_mut()
                .expect("terminal is an object")
                .remove("raw");
            assert!(!contains_key(&normalized, "billed_units"));
            assert_ne!(
                normalized.get("finish_reason"),
                Some(&Value::String("COMPLETE".to_string())),
                "the normalized finish reason is rig's spelling, not Cohere's"
            );
            assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let delta = recorded_message_end_delta(SCENARIO);
    assert_eq!(
        raw.get("finish_reason"),
        Some(&Value::String("COMPLETE".to_string())),
        "raw keeps Cohere's own finish_reason spelling"
    );
    for pointer in [
        "/usage/billed_units/input_tokens",
        "/usage/billed_units/output_tokens",
        "/usage/tokens/input_tokens",
        "/usage/tokens/output_tokens",
    ] {
        assert_eq!(
            number_at(&raw, pointer),
            number_at(&delta, pointer),
            "raw must carry {pointer} exactly as the message-end delta sent it"
        );
        assert!(
            number_at(&delta, pointer).is_some(),
            "{SCENARIO}: the recorded message-end delta should carry {pointer}"
        );
    }
}
