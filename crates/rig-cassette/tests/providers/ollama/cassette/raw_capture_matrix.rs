//! Matrix for raw response capture on Ollama's blocking `/api/chat` path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! Capture is always on. Every completion the provider seam returns carries
//! `raw`: the reply document the daemon sent, which the decoder also parsed as
//! [`ollama::CompletionResponse`]. It never replaces a normalized field, and
//! it is not a request-side concern: nothing about it is sent to the daemon.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by
//! hand without a provider response behind it, which no cell here can
//! produce.
//!
//! Ollama is the natural provider for cell 2: its response carries
//! nanosecond timings (`total_duration`, `load_duration`, `eval_duration`) that
//! the normalized [`rig::completion::CompletionResponse`] has no field for, so
//! `raw` is the only way a caller can read them without a second request.
//!
//! Ollama speaks its own `/api/chat` dialect — `done`/`done_reason`,
//! `prompt_eval_count`/`eval_count`, no provider-reported total and no
//! response id — so the shared chat-completions format contract does not
//! describe this wire. Only the execution layer
//! ([`capture_completion`](crate::raw_capture::capture_completion)) and the
//! format-agnostic normalized-surface assertion
//! ([`assert_normalized_lacks`](crate::raw_capture::assert_normalized_lacks))
//! are shared; every claim in Ollama's own vocabulary is a visible line in the
//! cell that makes it.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns (record mode writes the fixture on the way
//! out): a fixture without the durations, or one that is not a completed
//! (`done: true`) turn, fails loudly rather than passing vacuously.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_provider_type` | typed access | `ollama::CompletionResponse::deserialize(&*raw)` reads back and agrees with the normalized response | recorded |
//! | 2 | `raw_exposes_ollama_durations` | provider-only fields | `total_duration`/`load_duration`/`eval_duration` in `raw` equal the fixture body | recorded |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the provider type read out of `raw`, and out of the fixture body, agrees with the normalized response field by field | recorded |
//!
//! Every cell is recorded: Ollama runs locally with no credential, so there is
//! nothing here the harness cannot reproduce.
//!
//! Re-record with a local Ollama daemon serving `qwen3:4b`:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test ollama ollama::cassette::raw_capture_matrix -- --nocapture --test-threads=1`

use rig::providers::ollama;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_ollama_cassette;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{Observed, normalized_without_raw};

const OLLAMA_PROVIDER: &str = "ollama";
const MODEL: &str = "qwen3:4b";

/// A prompt whose answer is a single token keeps the recorded body small; the
/// matrix asserts on the response's metadata, never its prose.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// `think: false` keeps qwen3's reasoning trace out of the recording; the
/// durations this matrix reads are reported either way.
fn request(
    model: &(impl rig::completion::CompletionModel + Clone),
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .additional_params(json!({ "think": false }))
        .build()
}

/// The premise every duration cell rests on: the recorded body is a completed
/// (`done: true`) Ollama chat response that reports its timings.
fn assert_recorded_completed_with_durations(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("done"),
        Some(&Value::Bool(true)),
        "{scenario}: the recorded turn must be a completed Ollama response"
    );
    for field in ["total_duration", "load_duration", "eval_duration"] {
        assert!(
            body.get(field).and_then(Value::as_u64).is_some(),
            "{scenario}: the recorded body must report `{field}` — without it \
             this cell cannot prove raw exposes a provider-only field"
        );
    }
}

// ---------------------------------------------------------------------------
// 1: raw is exactly what raw_completion would have returned, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_ollama_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();

    let raw = &response.raw;

    // Typed access is recoverable: the provider's own wire type reads
    // the captured value back. The capture is the reply *document*, so
    // it is a superset of what the type models rather than that type's
    // re-serialization — which is why this asserts the type reads back
    // and agrees, not that it round-trips byte for byte.
    let typed = ollama::CompletionResponse::deserialize(raw)
        .expect("raw must deserialize into ollama::CompletionResponse");

    // The typed view agrees with the normalized one on what the model
    // said, so raw is a superset, not a divergent copy.
    assert_eq!(typed.model, MODEL);
    assert!(typed.done, "raw carries the completed turn");
    assert_eq!(
        Some(typed.model.as_str()),
        response.model.as_deref(),
        "normalized model equals the raw model"
    );

    // Premise: what was captured is what the wire carried — the fixture body
    // deserializes into the same provider type.
    let (_, body) = recorded_json_turn(OLLAMA_PROVIDER, scenario);
    let recorded = ollama::CompletionResponse::deserialize(&body)
        .expect("recorded body must be an Ollama chat response");
    assert!(recorded.done);
}

// ---------------------------------------------------------------------------
// 2: a provider-only field rig does not normalize is readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_ollama_durations() {
    let scenario = "raw_capture_matrix/raw_exposes_ollama_durations";
    let captured = Observed::default();
    let sink = captured.clone();
    with_ollama_cassette(
        "raw_capture_matrix/raw_exposes_ollama_durations",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();

    // The normalized response provably lacks the timings: its serialized
    // form has no such keys, and `Usage` models tokens only.
    assert_normalized_lacks(
        &normalized_without_raw(response.clone()),
        &["total_duration", "load_duration", "eval_duration"],
    );

    let raw = response.raw;

    // Premise + assertion in one: the fixture body reports the durations, and
    // raw carries exactly the values the wire did.
    let (_, body) = recorded_json_turn(OLLAMA_PROVIDER, scenario);
    assert_recorded_completed_with_durations(&body, scenario);
    for field in [
        "total_duration",
        "load_duration",
        "eval_duration",
        "prompt_eval_duration",
    ] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    let typed = ollama::CompletionResponse::deserialize(&raw)
        .expect("raw must deserialize into ollama::CompletionResponse");
    assert_eq!(typed.total_duration, body["total_duration"].as_u64());
    assert_eq!(typed.eval_duration, body["eval_duration"].as_u64());
    assert_eq!(typed.load_duration, body["load_duration"].as_u64());
}

// ---------------------------------------------------------------------------
// 3: raw and the typed route tell one story
// ---------------------------------------------------------------------------

/// One decoder, two views. The provider's own type, read back out of `raw` and
/// read out of the recorded wire body, must agree with the normalized
/// response on every field the normalized response has: capture neither
/// alters a normalized field nor diverges from the bytes the daemon sent.
///
/// This used to compare the normalized response with a second normalization
/// of `raw`. There is one mapping now, so that comparison would only be the
/// mapping against a copy of itself; asserting the provider's fields against
/// the normalized ones pins the mapping instead.
#[tokio::test]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = Observed::default();
    let sink = captured.clone();
    with_ollama_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();

    let raw = &response.raw;
    let typed = ollama::CompletionResponse::deserialize(raw)
        .expect("raw must deserialize into ollama::CompletionResponse");

    assert_eq!(response.provider, OLLAMA_PROVIDER);
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    // The provider's own vocabulary, paired with what the decoder made
    // of it. Read from the payload rather than hardcoded, so the cell
    // pins the mapping for whichever reason the fixture holds and fails
    // loudly on one this wire has never seen.
    let expected = match typed.done_reason.as_deref() {
        Some("stop") => rig::completion::FinishReason::Stop,
        Some("length") => rig::completion::FinishReason::Length,
        other => {
            panic!("the recorded turn should stop naturally or hit the cap, got {other:?}")
        }
    };
    assert_eq!(
        response.finish_reason(),
        Some(expected),
        "the decoder maps Ollama's `done_reason` onto the normalized vocabulary"
    );
    assert_eq!(typed.prompt_eval_count, response.usage.input_tokens);
    assert_eq!(typed.eval_count, response.usage.output_tokens);
    assert_eq!(
        typed
            .prompt_eval_count
            .zip(typed.eval_count)
            .map(|(i, o)| i + o),
        response.usage.total_tokens,
        "Ollama reports no total; the decoder derives it from both counts"
    );
    // Ollama's chat reply carries no response id, so the normalized
    // identity reports none — the documented outcome.
    assert_eq!(response.identity().response_id, None);
    assert!(!response.choice.is_empty());

    let (_, body) = recorded_json_turn(OLLAMA_PROVIDER, scenario);
    assert_recorded_completed_with_durations(&body, scenario);
    let from_wire = ollama::CompletionResponse::deserialize(&body)
        .expect("recorded body must be an Ollama chat response");
    assert_eq!(
        Some(from_wire.model.as_str()),
        response.model.as_deref(),
        "the normalized response names the model the wire bytes named"
    );
    assert_eq!(from_wire.prompt_eval_count, response.usage.input_tokens);
    assert_eq!(from_wire.eval_count, response.usage.output_tokens);
    assert_eq!(
        normalized_without_raw(response)
            .get("finish_reason")
            .cloned(),
        from_wire
            .done_reason
            .as_deref()
            .map(|reason| serde_json::json!(reason)),
        "the wire's own `done_reason` reaches the normalized finish reason verbatim"
    );
}
