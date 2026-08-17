//! Matrix for raw response capture on Ollama's blocking `/api/chat` path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion returned by the provider seam
//! carries `raw`: the value the model's inherent
//! [`CompletionModel::raw_completion`](rig::providers::ollama::CompletionModel::raw_completion)
//! would have returned — the response as [`ollama::CompletionResponse`] parsed
//! it — serialized with `serde_json::to_value` before normalization. It never
//! replaces a normalized field, and it is not a request-side concern: nothing
//! about it is sent to the daemon. `raw == Value::Null` means only that a
//! `CompletionResponse` was built by hand without a provider response behind
//! it, which no cell here can produce.
//!
//! Ollama is the natural provider for cell 2: its response carries
//! nanosecond timings (`total_duration`, `load_duration`, `eval_duration`) that
//! the normalized [`rig::completion::CompletionResponse`] has no field for, so
//! `raw` is the only way a caller can read them without a second request.
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
//! | 1 | `raw_round_trips_provider_type` | typed access | `ollama::CompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 2 | `raw_exposes_ollama_durations` | provider-only fields | `total_duration`/`load_duration`/`eval_duration` in `raw` equal the fixture body | recorded |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the normalized response equals `raw` re-normalized (`try_into`) and the fixture body re-normalized | recorded |
//!
//! Every cell is recorded: Ollama runs locally with no credential, so there is
//! nothing here the harness cannot reproduce.
//!
//! Re-record with a local Ollama daemon serving `qwen3:4b`:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test ollama ollama::cassette::raw_capture_matrix -- --nocapture --test-threads=1`

use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::ollama;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_ollama_cassette;
use crate::cassettes::recorded_interaction_bodies;

const OLLAMA_PROVIDER: &str = "ollama";
const MODEL: &str = "qwen3:4b";

/// A prompt whose answer is a single token keeps the recorded body small; the
/// matrix asserts on the response's metadata, never its prose.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// `think: false` keeps qwen3's reasoning trace out of the recording; the
/// durations this matrix reads are reported either way.
fn request(model: &ollama::CompletionModel) -> rig::completion::CompletionRequest {
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

/// The single recorded interaction of a scenario, request and response parsed
/// as JSON.
fn recorded_json_interaction(scenario: &str) -> (Value, Value) {
    let bodies = recorded_interaction_bodies(OLLAMA_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let (request, response) = &bodies[0];
    let request: Value = serde_json::from_str(request)
        .unwrap_or_else(|err| panic!("{scenario}: recorded request should be JSON: {err}"));
    let response: Value = serde_json::from_str(response)
        .unwrap_or_else(|err| panic!("{scenario}: recorded response should be JSON: {err}"));
    (request, response)
}

/// The normalized response minus its `raw`, as JSON, so a response can be
/// compared field-for-field against a re-normalization that has no `raw`.
fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = Value::Null;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: raw is exactly what raw_completion would have returned, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    with_ollama_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;

            // Typed access is recoverable: the provider's own wire type reads the
            // captured value back, and re-serializing it reproduces the capture
            // exactly — the escape hatch is the raw_completion value, nothing
            // more and nothing less.
            let typed = ollama::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into ollama::CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "ollama::CompletionResponse must round-trip through its own serde"
            );

            // The typed view agrees with the normalized one on what the model
            // said, so raw is a superset, not a divergent copy.
            assert_eq!(typed.model, MODEL);
            assert!(typed.done, "raw carries the completed turn");
            assert_eq!(
                Some(typed.model.as_str()),
                response.model.as_deref(),
                "normalized model equals the raw model"
            );
        },
    )
    .await;

    // Premise: what was captured is what the wire carried — the fixture body
    // deserializes into the same provider type.
    let (_, body) = recorded_json_interaction(scenario);
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
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_capture_matrix/raw_exposes_ollama_durations",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            // The normalized response provably lacks the timings: its serialized
            // form has no such keys, and `Usage` models tokens only.
            let normalized = normalized_without_raw(response.clone());
            for field in ["total_duration", "load_duration", "eval_duration"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized CompletionResponse must not grow a `{field}` field"
                );
            }

            let raw = response.raw.clone();
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");

    // Premise + assertion in one: the fixture body reports the durations, and
    // raw carries exactly the values the wire did.
    let (_, body) = recorded_json_interaction(scenario);
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

/// The normalized response, with `raw` stripped, must equal the normalization
/// (`try_into`) of `raw` read back through the provider type — and equal the
/// normalization of the recorded wire body. Capture is a pure serialization of
/// the value normalization consumed: it neither alters a normalized field nor
/// diverges from the bytes the daemon sent.
#[tokio::test]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            let from_raw: RigCompletionResponse = ollama::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into ollama::CompletionResponse")
                .try_into()
                .expect("raw must normalize");

            assert_eq!(response.provider, OLLAMA_PROVIDER);
            assert_eq!(from_raw.provider, response.provider);
            assert_eq!(from_raw.model, response.model);
            assert_eq!(from_raw.finish_reason(), response.finish_reason());
            assert_eq!(from_raw.identity(), response.identity());
            assert_eq!(from_raw.usage, response.usage);
            assert!(!response.choice.is_empty());
            assert_eq!(
                normalized_without_raw(from_raw),
                normalized_without_raw(response.clone()),
                "re-normalizing raw must reproduce the normalized response field-for-field"
            );

            *sink.lock().expect("capture mutex") = Some(response);
        },
    )
    .await;

    let response = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured the response");
    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_completed_with_durations(&body, scenario);
    let from_wire: RigCompletionResponse = ollama::CompletionResponse::deserialize(&body)
        .expect("recorded body must be an Ollama chat response")
        .try_into()
        .expect("recorded body must normalize");
    assert_eq!(
        normalized_without_raw(response),
        normalized_without_raw(from_wire),
        "the normalized response must equal the normalization of the wire bytes \
         it was built from"
    );
}
