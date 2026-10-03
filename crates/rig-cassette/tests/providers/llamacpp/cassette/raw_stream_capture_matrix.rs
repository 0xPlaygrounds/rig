//! Matrix for raw terminal-record capture on llama.cpp's streaming path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. llama.cpp streams through the OpenAI `Chat` wire,
//! whose decoder builds a terminal record: a JSON object with the keys
//! `usage`, `finish_reason`, `response_id`, `model`, `logprobs` and
//! `additional_params`. Its usage comes from the stream's final `data:`
//! frame, and the envelope fields the chunks carried (`object`, `created`,
//! `system_fingerprint`) accumulate under `additional_params`. Every
//! terminal record the seam yields carries `raw` — that record serialized by
//! the decoder before it folds into a `CompletionResponse` — the terminal record
//! only, never the frames, and nothing about it is sent to the server.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by hand
//! without a provider terminal behind it, which no cell here can produce.
//!
//! The envelope fields are exactly what the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse) has no home for, so cell 2
//! reads them back through `raw` and checks them against the recorded frames.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns: the recorded SSE stream must end with a frame
//! carrying `usage`, or the cell fails loudly.
//!
//! Every cell streams its one turn through
//! [`capture_terminal`](crate::raw_capture::capture_terminal) — the
//! stream must yield exactly one terminal record — and keeps its cassette
//! wrapper call, scenario literal included, at the test site, which is where
//! `cassette_safety` reads a scenario from. The fixture premises stay local:
//! "the last data frame carries the usage" and "every chunk agrees on an
//! envelope key" are this dialect's rules, and neither is the shared
//! chat-completions contract of one usage-bearing frame
//! ([`chat::recorded_sole_usage_frame`](crate::raw_capture::chat::recorded_sole_usage_frame)).
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | record shape | `raw` is the terminal record; its id, model, finish reason and usage are the terminal's | recorded |
//! | 2 | `stream_raw_exposes_envelope_fields` | terminal-only fields | `additional_params.system_fingerprint`/`object` in `raw` equal the recorded frames; usage equals the terminal frame | recorded |
//! | 3 | `stream_raw_preserves_llamacpp_timings` | Part 4: dropped fields | `timings` from the terminal frame survives under `additional_params` | recorded |
//!
//! Cell 3 is the streaming half of `raw_capture_matrix`'s timings cell. The
//! terminal record's `additional_params` collects every top-level chunk
//! field the shared wire shape does not name, so a streamed terminal does not
//! erase a field merely because the shape does not know it. llama.cpp's
//! `timings` reach the caller here *without* any provider-specific type. On
//! the blocking path `timings` reach the caller because `raw` is the reply
//! document.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10964-b29c606e2.
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test llamacpp raw_stream_capture_matrix -- --test-threads=1`

use serde_json::Value;

use super::super::cassette_support::*;
use crate::cassettes::CassetteMode;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal, chat};
use crate::support::Observed;
use crate::support::normalized_without_raw;
use rig::completion::CompletionRequest;

const LLAMACPP_PROVIDER: &str = "llamacpp";
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(1024)
}

// ---------------------------------------------------------------------------
// 1: raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let sink = Observed::default();
    with_llamacpp_cassette_result(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| capture_terminal(client.completion(CASSETTE_MODEL), request(), sink.clone()),
    )
    .await
    .expect("stream_raw_terminal_round_trips_provider_type should replay from its cassette");
    let terminal = sink.take();

    // `raw` is the terminal record the decoder emitted, read as JSON. Its
    // identity and accounting are the normalized ones, pinned against the
    // provider's own counters rather than compared to a copy of the mapping.
    chat::assert_terminal_round_trips(&terminal);

    let (_, terminal_frame) = chat::recorded_frames_with_terminal(LLAMACPP_PROVIDER, scenario);
    let raw = &terminal.raw;
    assert_eq!(
        raw["usage"]["prompt_tokens"], terminal_frame["usage"]["prompt_tokens"],
        "raw usage must be the terminal frame's usage"
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        terminal_frame["usage"]["completion_tokens"]
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_raw_exposes_envelope_fields() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_envelope_fields";
    let sink = Observed::default();
    with_llamacpp_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_envelope_fields",
        |client| capture_terminal(client.completion(CASSETTE_MODEL), request(), sink.clone()),
    )
    .await
    .expect("stream_raw_exposes_envelope_fields should replay from its cassette");
    let terminal = sink.take();

    // The normalized terminal record provably lacks the envelope.
    let normalized = normalized_without_raw(terminal.clone());
    assert_normalized_lacks(
        &normalized,
        &[
            "system_fingerprint",
            "object",
            "created",
            "additional_params",
        ],
    );

    let raw = terminal.raw;
    let (frames, terminal_frame) = chat::recorded_frames_with_terminal(LLAMACPP_PROVIDER, scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated envelope under additional_params");
    for key in ["system_fingerprint", "object"] {
        assert_eq!(
            params.get(key),
            Some(&chat::recorded_envelope_field(&frames, key, scenario)),
            "raw.additional_params.{key} must equal the recorded chunk envelope"
        );
    }
    // `created` is volatile: the scrubber placeholders it on disk, so only a
    // replay compares it exactly; a live recording proves raw carries it with
    // the wire's type.
    let created = chat::recorded_envelope_field(&frames, "created", scenario);
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(params.get("created"), Some(&created)),
        CassetteMode::Record => assert!(
            params.get("created").is_some_and(Value::is_u64) && created.is_u64(),
            "raw.additional_params.created must carry the chunk envelope's integer"
        ),
    }
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    // The terminal record keeps the envelope fields the chunks carried.
    let typed_params = &raw["additional_params"];
    assert_eq!(
        typed_params.get("system_fingerprint"),
        frames[0].get("system_fingerprint")
    );
}

// ---------------------------------------------------------------------------
// 3: `timings` reach the caller on the streaming path with no provider type
// ---------------------------------------------------------------------------

/// llama.cpp's `timings` ride the terminal frame and land under
/// `additional_params`.
///
/// The blocking path keeps this field because `raw` is the reply document.
/// The streaming path keeps it because the terminal record's
/// `additional_params` collects the chunk fields the shared shape does not
/// name. Pinning both paths makes that a measured fact rather than a reading
/// of the source.
#[tokio::test]
async fn stream_raw_preserves_llamacpp_timings() {
    let scenario = "raw_stream_capture_matrix/stream_raw_preserves_llamacpp_timings";
    let sink = Observed::default();

    with_llamacpp_cassette_result(
        "raw_stream_capture_matrix/stream_raw_preserves_llamacpp_timings",
        |client| capture_terminal(client.completion(CASSETTE_MODEL), request(), sink.clone()),
    )
    .await
    .expect("stream_raw_preserves_llamacpp_timings should replay from its cassette");

    let raw = sink.take().raw;
    let (_, terminal_frame) = chat::recorded_frames_with_terminal(LLAMACPP_PROVIDER, scenario);

    let recorded_timings = terminal_frame
        .get("timings")
        .expect("llama.cpp puts timings on the terminal streaming frame");
    let carried = raw
        .get("additional_params")
        .and_then(|params| params.get("timings"))
        .expect("the flattened catch-all must carry timings through to the caller");
    assert_eq!(
        carried, recorded_timings,
        "the terminal frame's timings must survive verbatim"
    );
    assert!(
        carried["predicted_per_second"]
            .as_f64()
            .is_some_and(|rate| rate > 0.0),
        "{carried}"
    );
}
