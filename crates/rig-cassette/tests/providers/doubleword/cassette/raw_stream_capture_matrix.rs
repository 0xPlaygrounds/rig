//! Raw provider response capture on Doubleword's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the decoder's own terminal
//! record behind the stream's `StreamEvent::Final` — for Doubleword the
//! shared chat-completions [`StreamingCompletionResponse`] over
//! [`ChatUsage`] — serialized. Capture is always on: there is no flag to
//! request it, nothing about it reaches the wire, and a `Value::Null` only
//! ever means a terminal built by hand with no provider record behind it. It
//! is the terminal record only, never the stream's frames. Doubleword reports
//! usage on the terminal frame alone, and the terminal's accumulated
//! `additional_params` carries the `object` tag the frames repeat; the
//! normalized terminal keeps the usage counts but has no slot for the tag or
//! the raw usage block.
//!
//! [`ChatUsage`] flattens the OpenAI-compatible counters and keeps whatever
//! else the dialect sent in a sibling map, so Doubleword's backend extras
//! (`cache_creation`, `cache_creation_input_tokens`,
//! `cache_read_input_tokens`) survive into `raw` instead of being dropped at
//! the type boundary. Cell 2 pins that against a fixture that carries them.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<ChatUsage>` and re-serializes equal; its accounting normalizes to the terminal's usage; the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_usage_and_object` | terminal-only field | `raw.usage` counts equal the sole usage-bearing frame's *including* its unmodeled cache extras; `raw.additional_params.object` equals the frames' tag | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last
//! data frame — so the raw terminal record's usage is knowable from the bytes
//! and a recording whose stream stopped reporting usage fails loudly instead
//! of covering nothing. That is [`chat::recorded_sole_usage_frame`], the
//! single-frame rule rather than the agreeing-closing-frames one. Doubleword
//! contracts no request-id header, so the terminal's `provider_request_id` is
//! `None` — pinned as the documented outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_doubleword_cassette_result;
use crate::raw_capture::{
    assert_no_request_id, assert_normalized_lacks, capture_terminal, capture_text_and_terminal,
    chat, stream_normalized_without_raw,
};
use crate::support::Observed;

const PROVIDER: &str = "doubleword";
const PROMPT: &str = "Reply with the single word: pong";

/// The backend usage fields Doubleword sends beyond the OpenAI-compatible
/// counters; `ChatUsage` keeps them in its sibling map.
const UNMODELLED_USAGE: [&str; 3] = [
    "cache_creation",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
];

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(256).build()
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let sink = Observed::default();
    with_doubleword_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let (text, terminal) = sink.take();
    assert!(!text.is_empty());
    // The captured document is the decoder's terminal record serialized, so
    // it round-trips exactly and its native fields are the normalized ones.
    chat::assert_terminal_round_trips(&terminal);

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded frame");
    assert_no_request_id(terminal.provider_request_id.as_deref(), PROVIDER);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_usage_and_object() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object";
    let sink = Observed::default();
    with_doubleword_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object",
        |client| capture_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_exposes_terminal_usage_and_object should replay from its cassette");

    let terminal = sink.take();
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    let recorded_object = frame["object"]
        .as_str()
        .expect("Doubleword tags every chunk with an object");
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
    // The normalized terminal has no slot for the tag, and none for
    // Doubleword's backend usage extras — which is why the terminal record
    // keeps them: `ChatUsage` holds whatever the dialect sent beside the
    // OpenAI-compatible counters.
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(&normalized, &["object", "additional_params"]);
    for field in UNMODELLED_USAGE {
        assert!(
            recorded_usage.get(field).is_some(),
            "the recorded usage carries Doubleword's `{field}`: {recorded_usage}"
        );
        assert_eq!(
            raw["usage"][field], recorded_usage[field],
            "`usage.{field}` reaches the caller through the terminal record"
        );
    }
}
