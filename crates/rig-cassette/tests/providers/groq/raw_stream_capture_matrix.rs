//! Raw provider response capture on Groq's streaming chat-completions path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::completion::CompletionResponse::raw`] carries the provider-native
//! terminal record the decoder assembled behind the stream's finished
//! response: for Groq the shared chat-completions terminal record, a JSON
//! object with the keys `usage`, `finish_reason`, `response_id`, `model`,
//! `logprobs` and `additional_params`. Capture is always on: there is no flag
//! to request it, nothing about it reaches the wire, and a `Value::Null` only
//! ever means a terminal built by hand with no provider record behind it. It
//! is the terminal record only, never the stream's frames. That is the one
//! place `raw` is *not* a reply document: an SSE reply is many frames and no
//! single one of them is the answer, so the decoder's reassembled record is
//! what rides along. Groq's terminal usage carries its timing accounting
//! (`queue_time`, `prompt_time`, ...) that the normalized `Usage` has no slot
//! for, and the accumulated `additional_params` carries the `x_groq` envelope
//! and `system_fingerprint` the frames repeat; both are reachable only through
//! `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `stream_raw_exposes_terminal_queue_time` | terminal-only field | `raw.usage.queue_time` and `raw.additional_params.x_groq.id` equal the recorded terminal frame's | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears only on the stream's closing frames (Groq
//! sends it on the finish frame and again on a trailing empty-choices frame,
//! with identical counts) and never on a content frame — so the raw terminal
//! record's usage is knowable from the bytes and a recording whose stream
//! stopped reporting usage fails loudly instead of covering nothing. That is
//! [`chat::recorded_agreeing_usage_frames`], a different premise from the
//! sole-terminal-frame rule its single-frame siblings use.

use rig::completion::CompletionRequest;
use serde_json::json;

use super::RAW_CAPTURE_MATRIX_MODEL;
use super::support::with_groq_cassette_result;
use crate::raw_capture::{capture_terminal, chat};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(16)
}

// ================================================================
// 1. raw is the terminal record
// ================================================================

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_queue_time() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_queue_time";
    let sink = Observed::default();
    with_groq_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_queue_time",
        |client| {
            capture_terminal(
                client.completion(RAW_CAPTURE_MATRIX_MODEL),
                request(),
                sink.clone(),
            )
        },
    )
    .await
    .expect("stream_raw_exposes_terminal_queue_time should replay from its cassette");

    let terminal = sink.take();
    let frame = chat::recorded_agreeing_usage_frames(PROVIDER, SCENARIO);
    let recorded_queue_time = frame["usage"]["queue_time"]
        .as_f64()
        .expect("Groq's terminal usage reports queue_time");
    // `x_groq` rides only on the finish frame; the accumulated terminal
    // params keep it, so its id is knowable from the frame that carried it.
    let recorded_x_groq_id = crate::cassettes::recorded_sse_json_frames(PROVIDER, SCENARIO)
        .into_iter()
        .find_map(|frame| frame["x_groq"]["id"].as_str().map(str::to_owned))
        .expect("Groq's closing frames carry an x_groq envelope with an id");

    let raw = &terminal.raw;
    assert_eq!(raw["usage"]["queue_time"], json!(recorded_queue_time));
    assert_matches_recorded_token(
        raw["additional_params"]["x_groq"]["id"].as_str(),
        Some(recorded_x_groq_id.as_str()),
        "x_groq.id",
    );
    // The normalized terminal has no slot for either.
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("queue_time").is_none(),
        "the normalized usage has no timing slot: {normalized_usage}"
    );
    let normalized = crate::support::normalized_without_raw(terminal.clone());
    crate::raw_capture::assert_normalized_lacks(&normalized, &["x_groq", "additional_params"]);
}
