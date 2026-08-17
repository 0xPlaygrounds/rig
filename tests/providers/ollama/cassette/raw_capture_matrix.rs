//! Matrix for opt-in raw response capture on Ollama's blocking `/api/chat`
//! path (`CompletionRequest::capture_raw_response` →
//! `CompletionResponse::raw`).
//!
//! # The feature
//!
//! `raw` is the value the model's inherent
//! [`CompletionModel::raw_completion`](rig::providers::ollama::CompletionModel::raw_completion)
//! would have returned — the response as [`ollama::CompletionResponse`] parsed
//! it — serialized with `serde_json::to_value`. It is populated only when the
//! request opted in, never replaces a normalized field, and never reaches the
//! wire: the flag is `#[serde(skip)]` and Ollama's request struct is built
//! from named fields, so an opted-in request must serialize byte-for-byte like
//! an opted-out one.
//!
//! Ollama is the natural provider for cell 3: its response carries
//! nanosecond timings (`total_duration`, `load_duration`, `eval_duration`) that
//! the normalized [`rig::completion::CompletionResponse`] has no field for, so
//! `raw` is the only way a caller can read them without a second request.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns (record mode writes the fixture on the way
//! out): a fixture without the durations, or a pair of interactions whose
//! request bodies differ, fails loudly rather than passing vacuously.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_raw_is_none` | flag off (default) | `raw == None` | recorded |
//! | 2 | `capture_on_raw_round_trips_provider_type` | flag on | `ollama::CompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `capture_on_exposes_ollama_durations` | provider-only fields | `total_duration`/`load_duration`/`eval_duration` in `raw` equal the fixture body | recorded |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | recorded |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | recorded |
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
use crate::cassettes::{recorded_interaction_bodies, recorded_json_request};

const OLLAMA_PROVIDER: &str = "ollama";
const MODEL: &str = "qwen3:4b";

/// A prompt whose answer is a single token keeps the recorded body small; the
/// matrix asserts on the response's metadata, never its prose.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// `think: false` keeps qwen3's reasoning trace out of the recording; the
/// durations this matrix reads are reported either way.
fn request(
    model: &ollama::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .additional_params(json!({ "think": false }))
        .capture_raw_response(capture_raw)
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

/// The recorded interaction bodies of a scenario, request and response parsed
/// as JSON, in wire order.
fn recorded_json_interactions(scenario: &str) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(OLLAMA_PROVIDER, scenario)
        .into_iter()
        .map(|(request, response)| {
            let request: Value = serde_json::from_str(&request)
                .unwrap_or_else(|err| panic!("{scenario}: recorded request should be JSON: {err}"));
            let response: Value = serde_json::from_str(&response).unwrap_or_else(|err| {
                panic!("{scenario}: recorded response should be JSON: {err}")
            });
            (request, response)
        })
        .collect()
}

/// The normalized response minus its `raw`, as JSON, so two responses can be
/// compared field-for-field regardless of whether one captured raw.
fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: the default is off, and off means None — not an empty object
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/capture_off_raw_is_none";
    with_ollama_cassette(
        "raw_capture_matrix/capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = request(&model, false);
            assert!(
                !request.capture_raw_response,
                "premise: the builder default is off"
            );

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            assert!(
                response.raw.is_none(),
                "raw must stay None when capture was not requested, got {:?}",
                response.raw
            );
            assert!(
                !response.choice.is_empty(),
                "the normalized choice is unaffected by the flag"
            );
        },
    )
    .await;

    // The recording is a real completed turn: `None` above means "not
    // requested", not "the provider sent nothing".
    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_eq!(body.get("done"), Some(&Value::Bool(true)));
}

// ---------------------------------------------------------------------------
// 2: on → raw is exactly what raw_completion would have returned, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/capture_on_raw_round_trips_provider_type";
    with_ollama_cassette(
        "raw_capture_matrix/capture_on_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("raw must be populated when capture was requested");

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
    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    let recorded = ollama::CompletionResponse::deserialize(&body)
        .expect("recorded body must be an Ollama chat response");
    assert!(recorded.done);
}

// ---------------------------------------------------------------------------
// 3: a provider-only field rig does not normalize is readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_on_exposes_ollama_durations() {
    let scenario = "raw_capture_matrix/capture_on_exposes_ollama_durations";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_capture_matrix/capture_on_exposes_ollama_durations",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
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

            let raw = response
                .raw
                .as_deref()
                .expect("raw must be populated when capture was requested")
                .clone();
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
    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
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
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on — so the two
/// request bodies came from the same process and the same builder and differ
/// in nothing but the local flag.
#[tokio::test]
async fn request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/request_invariant_off_vs_on";
    with_ollama_cassette(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("flag-off completion should succeed");
            let on = model
                .completion(request(&model, true))
                .await
                .expect("flag-on completion should succeed");
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
        },
    )
    .await;

    let bodies = recorded_interaction_bodies(OLLAMA_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: the scenario must record exactly the off and on requests"
    );
    let (off_request, _) = &bodies[0];
    let (on_request, _) = &bodies[1];
    assert_eq!(
        off_request, on_request,
        "the flag-on request body must be byte-identical to the flag-off one — \
         capture_raw_response is local policy and must never reach Ollama"
    );
    // And neither body mentions the flag under any spelling.
    let first: Value = recorded_json_request(OLLAMA_PROVIDER, scenario);
    assert!(
        !off_request.contains("capture_raw"),
        "the request body must not carry the flag: {off_request}"
    );
    assert_eq!(first["model"], MODEL);
    assert_eq!(first["stream"], Value::Bool(false));
}

// ---------------------------------------------------------------------------
// 5: normalization is a pure function of the wire bytes, flag or no flag
// ---------------------------------------------------------------------------

/// Two interactions (off, on). Each response, with `raw` stripped, must equal
/// the normalization of *its own* recorded body — proving the flag changed
/// nothing on the normalized surface — and the two must agree on every field
/// the model does not decide token-by-token.
#[tokio::test]
async fn normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("flag-off completion should succeed");
            let on = model
                .completion(request(&model, true))
                .await
                .expect("flag-on completion should succeed");

            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            assert_eq!(off.provider, OLLAMA_PROVIDER);
            assert_eq!(on.provider, off.provider);
            assert_eq!(on.model, off.model);
            assert_eq!(on.finish_reason(), off.finish_reason());
            assert_eq!(on.identity(), off.identity());
            assert!(!off.choice.is_empty());
            assert!(!on.choice.is_empty());

            *sink.lock().expect("capture mutex") = vec![off, on];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let interactions = recorded_json_interactions(scenario);
    assert_eq!(
        interactions.len(),
        2,
        "{scenario}: expected off and on turns"
    );

    for ((_, body), response) in interactions.into_iter().zip(responses) {
        assert_recorded_completed_with_durations(&body, scenario);
        let from_wire: RigCompletionResponse = ollama::CompletionResponse::deserialize(&body)
            .expect("recorded body must be an Ollama chat response")
            .try_into()
            .expect("recorded body must normalize");
        assert_eq!(
            normalized_without_raw(response),
            normalized_without_raw(from_wire),
            "the normalized response must equal the normalization of its own \
             wire bytes — capture must not touch any normalized field"
        );
    }
}
