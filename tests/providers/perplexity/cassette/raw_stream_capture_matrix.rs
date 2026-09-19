//! Raw provider response capture on Perplexity's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the decoder's own terminal
//! record — for Perplexity the shared chat-completions
//! [`StreamingCompletionResponse`] over [`ChatUsage`] — serialized. Capture is
//! always on: there is no flag to request it, nothing about it reaches the
//! wire, and a `Value::Null` only ever means a terminal built by hand with no
//! provider record behind it. It is the terminal record only, never the
//! stream's frames. Perplexity reports usage on *every* frame and the last
//! frame's counts are the terminal's; the terminal's accumulated
//! `additional_params` carries the `object` tag the frames repeat. Neither the
//! tag nor the raw usage block has a slot on the normalized terminal.
//!
//! [`ChatUsage`] flattens the OpenAI-compatible counters and keeps whatever
//! else the dialect sent in a sibling map, so Perplexity's `cost` and
//! `search_context_size` survive into `raw` instead of being dropped at the
//! type boundary. Cell 2 pins that against a fixture that carries them.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into the terminal record and re-serializes equal; its `ChatUsage` normalizes to the terminal usage; the normalized terminal reproduces the recorded last frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_usage_and_object` | terminal-only fields | `raw.usage` equals the recorded last frame's counters *including* the unmodeled `cost` block; `raw.additional_params.object` equals the frames' tag | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that the recorded stream's last data frame carries usage and a
//! finish reason — so the raw terminal record's usage is knowable from the
//! bytes and a recording whose stream stopped reporting usage fails loudly
//! instead of covering nothing. That premise is this file's own
//! [`recorded_terminal_frame`] and deliberately neither of the shared readers:
//! Perplexity repeats its accounting on *every* frame, content frames
//! included, so `chat::recorded_sole_usage_frame`'s "exactly one
//! usage-bearing frame" and `chat::recorded_agreeing_usage_frames`'s "never on
//! a content frame" are both false here — the rule is "the last frame wins".
//! Perplexity contracts no request-id header, so the terminal's
//! `provider_request_id` is `None` — pinned as the documented outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::perplexity;
use serde_json::{Value, json};

use super::super::support::with_perplexity_cassette;
use crate::raw_capture::{
    assert_no_request_id, assert_normalized_lacks, capture_terminal, capture_text_and_terminal,
    chat, stream_normalized_without_raw,
};
use crate::support::Observed;

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";
/// Names the dialect in the "no id header" outcome the cells pin.
const DIALECT: &str = "Perplexity";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The recorded stream's last data frame: it carries the finish reason and
/// the usage the terminal record reports (Perplexity repeats usage on every
/// frame; the last frame's counts win).
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let terminal = frames
        .last()
        .expect("the recorded stream must carry at least one frame")
        .clone();
    assert!(
        terminal["usage"].is_object(),
        "the last frame must carry usage: {terminal}"
    );
    assert!(
        terminal["choices"][0]["finish_reason"].is_string(),
        "the last frame must carry the finish reason: {terminal}"
    );
    terminal
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let observed = Observed::default();
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| async move {
            capture_text_and_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("the stream should open");
        },
    )
    .await;

    let (text, terminal) = observed.take();
    assert!(!text.is_empty());
    // Unlike a unary `CompletionResponse::raw`, a stream's terminal `raw` IS
    // this record serialized (`openai/wire/chat.rs`'s `emit_terminal`), so
    // the round trip is exact — and `ChatUsage::to_normalized`, the mapping
    // the terminal's usage went through, is pinned directly.
    chat::assert_terminal_round_trips(&terminal);

    let frame = recorded_terminal_frame(SCENARIO);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded last frame");
    assert_no_request_id(terminal.provider_request_id.as_deref(), DIALECT);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_usage_and_object() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object";
    let observed = Observed::default();
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object",
        |client| async move {
            capture_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("the stream should open");
        },
    )
    .await;

    let terminal = observed.take();
    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_object = frame["object"]
        .as_str()
        .expect("Perplexity tags every chunk with an object");
    let recorded_usage = &frame["usage"];

    let raw = &terminal.raw;
    assert_eq!(
        raw["usage"]["prompt_tokens"],
        recorded_usage["prompt_tokens"]
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        recorded_usage["completion_tokens"]
    );
    assert_eq!(raw["usage"]["total_tokens"], recorded_usage["total_tokens"]);
    assert_eq!(raw["additional_params"]["object"], json!(recorded_object));
    // The normalized terminal has no slot for the tag. The usage block, by
    // contrast, is the dialect's own: `ChatUsage` keeps the fields no
    // OpenAI-compatible shape models, so Perplexity's `cost` reaches a caller
    // through `raw` rather than being lost at the type boundary.
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(&normalized, &["object", "additional_params"]);
    assert!(
        recorded_usage.get("cost").is_some(),
        "the recorded usage carries Perplexity's cost block: {recorded_usage}"
    );
    assert_eq!(
        raw["usage"].get("cost"),
        recorded_usage.get("cost"),
        "the dialect's extra usage fields ride along in raw: {raw}"
    );
}
