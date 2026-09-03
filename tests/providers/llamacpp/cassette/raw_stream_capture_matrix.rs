//! Matrix for raw terminal-record capture on llama.cpp's streaming path
//! ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. llama.cpp streams through the shared OpenAI Chat
//! Completions model, whose
//! [`raw_stream`](rig::providers::openai::GenericCompletionModel::raw_stream)
//! yields [`openai::StreamingCompletionResponse`] as its terminal record: the
//! usage from the stream's final `data:` frame plus the envelope fields the
//! chunks carried (`object`, `created`, `system_fingerprint`) accumulated under
//! `additional_params`. Every terminal record the seam yields carries `raw` —
//! that record serialized by the adapter's `final_record` — the terminal record only,
//! never the frames, and nothing about it is sent to the server. `raw ==
//! Value::Null` means only that a `StreamFinal` was built by hand without a
//! provider terminal behind it, which no cell here can produce.
//!
//! The envelope fields are exactly what the normalized
//! [`StreamFinal`](rig::streaming::StreamFinal) has no home for, so cell 2
//! reads them back through `raw` and checks them against the recorded frames.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns: the recorded SSE stream must end with a frame
//! carrying `usage`, or the cell fails loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | typed access | `openai::StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 2 | `stream_raw_exposes_envelope_fields` | terminal-only fields | `additional_params.system_fingerprint`/`object` in `raw` equal the recorded frames; usage equals the terminal frame | recorded |
//! | 3 | `stream_raw_preserves_llamacpp_timings` | Part 4: dropped fields | `timings` from the terminal frame survives under `additional_params` | recorded |
//!
//! Cell 3 is the streaming half of the asymmetry `raw_capture_matrix`'s
//! timings cell describes. `StreamingCompletionChunk` carries a
//! `#[serde(flatten)]` catch-all for exactly this reason — "raw_stream must
//! not erase them merely because the shared wire shape does not know their
//! names yet" — so llama.cpp's `timings` reach the caller here *without* any
//! provider-specific type. `openai::CompletionResponse` on the blocking path
//! has no such catch-all, which is why `llamacpp::CompletionResponse` exists.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10499-6d05498.
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test llamacpp raw_stream_capture_matrix -- --test-threads=1`

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::providers::{llamacpp, openai};
use rig::streaming::{StreamEvent, StreamFinal};
use serde::Deserialize;
use serde_json::Value;

use super::super::cassette_support::*;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};

const LLAMACPP_PROVIDER: &str = "llamacpp";
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &llamacpp::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(1024).build()
}

async fn terminal_of(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut finals = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(record) = item.expect("stream item should be ok") {
            finals.push(record);
        }
    }
    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    finals.remove(0)
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose SSE body's last JSON frame carries `usage`. Returns
/// `(all frames, terminal frame)`.
fn recorded_frames_with_terminal(scenario: &str) -> (Vec<Value>, Value) {
    assert_eq!(
        recorded_interaction_bodies(LLAMACPP_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let frames = recorded_sse_json_frames(LLAMACPP_PROVIDER, scenario);
    let terminal = frames
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream should carry frames"));
    assert!(
        terminal.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded stream must end with a usage-bearing frame — \
         without it the terminal record carries no usage and this cell proves nothing"
    );
    (frames, terminal)
}

/// Every recorded chunk stamps the same envelope value for `key`; returns it.
fn recorded_envelope_field(frames: &[Value], key: &str, scenario: &str) -> Value {
    let mut values = frames.iter().filter_map(|frame| frame.get(key)).cloned();
    let first = values
        .next()
        .unwrap_or_else(|| panic!("{scenario}: recorded chunks must carry `{key}`"));
    assert!(
        values.all(|value| value == first),
        "{scenario}: every recorded chunk must agree on `{key}`"
    );
    first
}

// ---------------------------------------------------------------------------
// 1: raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_llamacpp_cassette(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;

            let raw = &terminal.raw;
            let typed = openai::StreamingCompletionResponse::<openai::Usage>::deserialize(raw)
                .expect("raw must deserialize into openai::StreamingCompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "openai::StreamingCompletionResponse must round-trip through its own serde"
            );

            // The typed terminal agrees with the normalized one on usage and
            // identity: raw is the record the adapter's `final_record` mapped.
            assert_eq!(
                typed.usage.prompt_tokens as u64,
                terminal.usage.input_tokens
            );
            assert_eq!(
                typed.usage.completion_tokens.map(|tokens| tokens as u64),
                Some(terminal.usage.output_tokens)
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.model, terminal.model);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let (_, terminal_frame) = recorded_frames_with_terminal(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
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
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_llamacpp_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_envelope_fields",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;

            // The normalized terminal record provably lacks the envelope.
            let mut without_raw = terminal.clone();
            without_raw.raw = Value::Null;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            for field in [
                "system_fingerprint",
                "object",
                "created",
                "additional_params",
            ] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized StreamFinal must not grow a `{field}` field"
                );
            }

            let raw = terminal.raw;
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (frames, terminal_frame) = recorded_frames_with_terminal(scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated envelope under additional_params");
    for key in ["system_fingerprint", "object"] {
        assert_eq!(
            params.get(key),
            Some(&recorded_envelope_field(&frames, key, scenario)),
            "raw.additional_params.{key} must equal the recorded chunk envelope"
        );
    }
    // `created` is volatile: the scrubber placeholders it on disk, so only a
    // replay compares it exactly; a live recording proves raw carries it with
    // the wire's type.
    let created = recorded_envelope_field(&frames, "created", scenario);
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(params.get("created"), Some(&created)),
        CassetteMode::Record => assert!(
            params.get("created").is_some_and(Value::is_u64) && created.is_u64(),
            "raw.additional_params.created must carry the chunk envelope's integer"
        ),
    }
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    let typed = openai::StreamingCompletionResponse::<openai::Usage>::deserialize(&raw)
        .expect("raw must deserialize into openai::StreamingCompletionResponse");
    let typed_params = typed
        .additional_params
        .expect("typed terminal must carry additional_params");
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
/// The blocking path needed `llamacpp::CompletionResponse` to keep this field;
/// the streaming path keeps it for free, because the chunk type has a
/// `#[serde(flatten)]` catch-all. Pinning both halves is what makes the
/// asymmetry a measured fact rather than a reading of the source.
#[tokio::test]
async fn stream_raw_preserves_llamacpp_timings() {
    let scenario = "raw_stream_capture_matrix/stream_raw_preserves_llamacpp_timings";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);

    with_llamacpp_cassette(
        "raw_stream_capture_matrix/stream_raw_preserves_llamacpp_timings",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;
            *sink.lock().expect("capture mutex") = Some(terminal.raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (_, terminal_frame) = recorded_frames_with_terminal(scenario);

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

    // And it really is the same shape the blocking path's typed field reads.
    let typed: rig::providers::llamacpp::Timings =
        serde_json::from_value(carried.clone()).expect("timings must decode into the typed form");
    assert!(
        typed.predicted_per_second.is_some_and(|rate| rate > 0.0),
        "{typed:?}"
    );
}
