//! Feature matrix for opt-in raw provider response capture on the Gemini REST
//! (`streamGenerateContent?alt=sse`) streaming seam.
//!
//! # The feature
//!
//! [`rig::completion::CompletionRequest::capture_raw_response`] read before
//! `raw_stream` and handed to `normalize_stream`: when set, the terminal
//! [`rig::streaming::StreamFinal`] carries, serialized, the value the inherent
//! `raw_stream` would have yielded as its `FinalResponse` — Gemini's own
//! [`StreamingCompletionResponse`] terminal record (`map_stream_final`'s
//! input). Off (the default) it stays `None`; the flag never reaches the wire.
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
//! | 1 | `flag_off_leaves_terminal_raw_unset` | default (`false`) | `StreamFinal.raw.is_none()` | recorded |
//! | 2 | `flag_on_roundtrips_streaming_completion_response` | `true` → typed access | `StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_terminal_only_fields` | `true` → un-normalized terminal fields | `finish_reason` spelled `"STOP"`, `usage_metadata.promptTokensDetails` == last frame, absent from the normalized terminal | recorded |
//! | 4 | `request_bytes_invariant_across_flag` | request boundary | recorded off/on request bodies byte-identical | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `streamGenerateContent` route.
//!
//! Cell 4 records one scenario with **two** interactions — the flag-off stream
//! first, then its flag-on twin — because the invariant is between the two;
//! the harness replays interactions in order.
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

fn request(model: &gemini::CompletionModel, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .capture_raw_response(capture)
        .build()
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
         un-normalized field cell 3 reads through `raw`"
    );
    last
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
    const SCENARIO: &str = "raw_stream_capture_matrix/flag_off_leaves_terminal_raw_unset";
    let observed_total: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed_total);
    with_gemini_cassette(
        "raw_stream_capture_matrix/flag_off_leaves_terminal_raw_unset",
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

    let last = last_usage_frame(SCENARIO);
    // The normalized terminal is untouched by the flag being off: its usage is
    // the wire's cumulative total.
    assert_eq!(
        observed_total.lock().expect("observation lock").take(),
        last.pointer("/usageMetadata/totalTokenCount")
            .and_then(Value::as_u64),
        "{SCENARIO}: the terminal usage should be the last frame's total"
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
        "raw_stream_capture_matrix/flag_on_roundtrips_streaming_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_stream_capture_matrix/flag_on_roundtrips_streaming_completion_response",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model, true)).await;

            let raw = terminal
                .raw
                .as_deref()
                .expect("capture was requested, so the terminal must carry raw");

            // `raw` is the value `raw_stream` would have yielded as its terminal,
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
// 3: on → terminal-only fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_terminal_only_fields() {
    const SCENARIO: &str = "raw_stream_capture_matrix/flag_on_exposes_terminal_only_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_stream_capture_matrix/flag_on_exposes_terminal_only_fields",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = stream_to_terminal(&model, request(&model, true)).await;

            let raw = terminal
                .raw
                .as_deref()
                .expect("capture was requested, so the terminal must carry raw");
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

// ---------------------------------------------------------------------------
// 4: the request boundary never sees the flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_bytes_invariant_across_flag() {
    const SCENARIO: &str = "raw_stream_capture_matrix/request_bytes_invariant_across_flag";
    with_gemini_cassette(
        "raw_stream_capture_matrix/request_bytes_invariant_across_flag",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = stream_to_terminal(&model, request(&model, false)).await;
            let on = stream_to_terminal(&model, request(&model, true)).await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            // Same normalized meaning either way — `raw` is additive.
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
