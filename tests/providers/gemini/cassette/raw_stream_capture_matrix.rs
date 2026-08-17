//! Feature matrix for raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! Raw capture is always on: `normalize_stream` serializes the value the
//! inherent `raw_stream` yielded as its `FinalResponse` — Gemini's own
//! [`StreamingCompletionResponse`] terminal record (`map_stream_final`'s
//! input) — onto the terminal [`rig::streaming::StreamFinal::raw`]. There is
//! no opt-in and nothing about it reaches the wire; `raw` is `Value::Null`
//! only on a terminal constructed without a provider stream behind it, never
//! because capture "was not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the terminal record. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded stream must end with a frame carrying
//! `usageMetadata` and a natural `finishReason`, or the terminal it asserts on
//! is not the one this matrix is about.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_roundtrips_streaming_completion_response` | typed access | `StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal and agrees with the normalized terminal | recorded |
//! | 2 | `raw_exposes_terminal_only_fields` | un-normalized terminal fields | `finish_reason` spelled `"STOP"`, `usage_metadata.promptTokensDetails` == last frame, absent from the normalized terminal | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `streamGenerateContent` route.
//!
//! Gemini's terminal record is assembled by rig from the stream's frames
//! (usage is cumulative per chunk; `finishReason` arrives on the last content
//! frame), so the "wire" side of each premise is the last frame that carries
//! `usageMetadata`, read with the same `data:` framing the streaming tests use.

use futures::StreamExt;
use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_cassette;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded stream stays short.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: streamed";

fn request(model: &gemini::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// Drain a model stream and return its single terminal record.
async fn stream_to_terminal(
    model: &gemini::CompletionModel,
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

/// The last recorded frame carrying `usageMetadata` — Gemini's usage is
/// cumulative, so this is the frame whose numbers the terminal must report.
fn last_usage_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let last = frames
        .iter()
        .rev()
        .find(|frame| frame.get("usageMetadata").is_some())
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: no recorded frame carries usageMetadata"));
    // The premise proper: the recorded stream finished naturally.
    let finished = frames.iter().any(|frame| {
        frame.pointer("/candidates/0/finishReason") == Some(&Value::String("STOP".to_string()))
    });
    assert!(
        finished,
        "{scenario}: the recorded stream should carry a finishReason of STOP; without a \
         natural finish the terminal this cell asserts on is not the shape under test"
    );
    assert!(
        last.pointer("/usageMetadata/promptTokensDetails")
            .and_then(Value::as_array)
            .is_some_and(|details| !details.is_empty()),
        "{scenario}: the terminal usage frame should carry promptTokensDetails, the \
         un-normalized field cell 2 reads through `raw`"
    );
    last
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
    const SCENARIO: &str = "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_stream_capture_matrix/raw_roundtrips_streaming_completion_response",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

            let raw = &terminal.raw;

            // `raw` is the value `raw_stream` yielded as its terminal,
            // serialized: Gemini's own terminal type reads it back and
            // re-serializes to the same value.
            let typed = StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's streaming terminal type");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "StreamingCompletionResponse must round-trip through its own Serialize/Deserialize"
            );

            // And the typed value agrees with the normalized terminal next to it.
            assert_eq!(typed.model_version, terminal.model);
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(
                typed.usage_metadata.total_token_count as u64,
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
    let last = last_usage_frame(SCENARIO);
    assert_eq!(
        raw.pointer("/usage_metadata/totalTokenCount"),
        last.pointer("/usageMetadata/totalTokenCount"),
        "{SCENARIO}: the captured terminal usage must be the last frame's total"
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
    with_gemini_cassette(
        "raw_stream_capture_matrix/raw_exposes_terminal_only_fields",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model)).await;

            let raw = &terminal.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized terminal provably lacks these: `finish_reason` is
            // rig's vocabulary (`stop`), and the per-modality breakdown has no
            // normalized home.
            let mut normalized =
                serde_json::to_value(&terminal).expect("normalized terminal serializes");
            normalized
                .as_object_mut()
                .expect("terminal is an object")
                .remove("raw");
            assert!(!contains_key(&normalized, "promptTokensDetails"));
            assert_ne!(
                normalized.get("finish_reason"),
                Some(&Value::String("STOP".to_string())),
                "the normalized finish reason is rig's spelling, not Gemini's"
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
    let last = last_usage_frame(SCENARIO);
    assert_eq!(
        raw.get("finish_reason"),
        Some(&Value::String("STOP".to_string())),
        "raw keeps Gemini's own finishReason spelling"
    );
    assert_eq!(
        raw.pointer("/usage_metadata/promptTokensDetails"),
        last.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry the terminal frame's promptTokensDetails exactly as the wire sent it"
    );
    assert_eq!(
        raw.pointer("/usage_metadata/candidatesTokenCount"),
        last.pointer("/usageMetadata/candidatesTokenCount"),
        "raw must carry the terminal frame's candidatesTokenCount untouched"
    );
}
