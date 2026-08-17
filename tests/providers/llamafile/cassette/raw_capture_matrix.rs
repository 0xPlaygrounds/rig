//! Matrix for opt-in raw response capture on llamafile's blocking path
//! (`CompletionRequest::capture_raw_response` → `CompletionResponse::raw`).
//!
//! # The feature
//!
//! llamafile is driven by the shared OpenAI Chat Completions model
//! (`openai::GenericCompletionModel<LlamafileExt>`), whose wire type is
//! [`openai::CompletionResponse`]. `raw` is therefore the value
//! [`raw_completion`](rig::providers::openai::GenericCompletionModel::raw_completion)
//! would have returned, serialized with `serde_json::to_value`; it is populated
//! only when the request opted in and it never reaches the wire.
//!
//! The chat-completions body carries envelope fields the normalized
//! [`rig::completion::CompletionResponse`] has no home for — `object`,
//! `created`, `system_fingerprint` — and those are what cell 3 reads back
//! through `raw`.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns: a fixture without the envelope fields, or a
//! pair of interactions whose request bodies differ, fails loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_raw_is_none` | flag off (default) | `raw == None` | recorded |
//! | 2 | `capture_on_raw_round_trips_provider_type` | flag on | `openai::CompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `capture_on_exposes_envelope_fields` | provider-only fields | `object`/`created`/`system_fingerprint` in `raw` equal the fixture body | recorded |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | recorded |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | recorded |
//!
//! Every cell is recorded. The fixtures were recorded against Ollama's
//! OpenAI-compatible endpoint (the `cassette_support` default upstream) serving
//! `qwen3:4b`, so the model name here is that one rather than the
//! `llama-server` model the older llamafile cassettes use; the wire shape is
//! the same chat-completions envelope either way. Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test llamafile llamafile::cassette::raw_capture_matrix -- --nocapture --test-threads=1`

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::{llamafile, openai};
use serde::Deserialize;
use serde_json::Value;

use super::super::cassette_support::with_llamafile_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_json_request};

const LLAMAFILE_PROVIDER: &str = "llamafile";
/// The model Ollama's OpenAI-compatible endpoint served at recording time.
const MODEL: &str = "qwen3:4b";
const PROMPT: &str = "Reply with exactly the single word: pong";

/// qwen3 spends tokens on a reasoning trace before the one-word answer and the
/// chat-completions route has no `think` switch, so the cap is generous
/// enough that the turn stops on its own (`finish_reason: "stop"`).
fn request(
    model: &llamafile::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(1024)
        .capture_raw_response(capture_raw)
        .build()
}

/// The premise every envelope cell rests on: the recorded body is a
/// chat-completions response carrying the envelope fields.
fn assert_recorded_envelope(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("chat.completion"),
        "{scenario}: the recorded body must be a chat.completion envelope"
    );
    assert!(
        body.get("created").and_then(Value::as_u64).is_some(),
        "{scenario}: the recorded body must carry `created`"
    );
    assert!(
        body.get("system_fingerprint")
            .and_then(Value::as_str)
            .is_some(),
        "{scenario}: the recorded body must carry `system_fingerprint` — without \
         it this cell cannot prove raw exposes a provider-only field"
    );
    assert!(
        body.get("usage").is_some(),
        "{scenario}: the recorded body must report usage"
    );
}

fn recorded_json_interactions(scenario: &str) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(LLAMAFILE_PROVIDER, scenario)
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

/// Compares a value rig read live against the fixture's copy of it.
///
/// Replay reads the scrubbed fixture back, so the two are byte-equal. Record
/// mode sees the provider's live value while the fixture holds the scrubber's
/// placeholder for volatile fields (`created` → 0, `id` → `..._REDACTED_n`),
/// so there the assertion is that both sides carry the field with the same
/// JSON type — the strongest claim a live recording can make.
fn assert_wire_value_matches(live: &Value, recorded: &Value, field: &str) {
    let (live_value, recorded_value) = (live.get(field), recorded.get(field));
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(
            live_value, recorded_value,
            "{field}: replayed value must equal the recorded wire value"
        ),
        CassetteMode::Record => {
            let (Some(live_value), Some(recorded_value)) = (live_value, recorded_value) else {
                panic!("{field}: both the live value and the recording must carry it");
            };
            assert_eq!(
                std::mem::discriminant(live_value),
                std::mem::discriminant(recorded_value),
                "{field}: live and recorded values must share a JSON type"
            );
        }
    }
}

fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/capture_off_raw_is_none";
    with_llamafile_cassette(
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
            assert!(!response.choice.is_empty());
            assert_eq!(response.provider, LLAMAFILE_PROVIDER);
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_envelope(&body, scenario);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_completion value, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/capture_on_raw_round_trips_provider_type";
    with_llamafile_cassette(
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
            let typed = openai::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into openai::CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "openai::CompletionResponse must round-trip through its own serde"
            );

            // And the typed view normalizes to the same normalized surface
            // the model returned — raw is the value `completion` normalized.
            let renormalized = typed
                .normalize(LLAMAFILE_PROVIDER)
                .expect("typed raw must normalize")
                .with_optional_provider_request_id(response.provider_request_id.clone());
            assert_eq!(
                normalized_without_raw(renormalized),
                normalized_without_raw(response),
                "normalizing the captured raw must reproduce the normalized response"
            );
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_envelope(&body, scenario);
    openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
}

// ---------------------------------------------------------------------------
// 3: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_on_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/capture_on_exposes_envelope_fields";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_llamafile_cassette(
        "raw_capture_matrix/capture_on_exposes_envelope_fields",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "created", "system_fingerprint"] {
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
    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_envelope(&body, scenario);
    for field in ["object", "system_fingerprint"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    // `created` and `id` are volatile: the cassette scrubber placeholders
    // them on the way to disk, so only a replay — which reads the scrubbed
    // bytes back — can compare them exactly. A live recording proves the
    // weaker shape claim: raw carries them with the wire's types.
    for field in ["created", "id"] {
        assert_wire_value_matches(&raw, &body, field);
    }
    let typed = openai::CompletionResponse::deserialize(&raw)
        .expect("raw must deserialize into openai::CompletionResponse");
    assert_eq!(Some(typed.object.as_str()), body["object"].as_str());
    assert!(typed.created > 0 || matches!(CassetteMode::current(), CassetteMode::Replay));
    assert_eq!(
        typed.system_fingerprint.as_deref(),
        body["system_fingerprint"].as_str()
    );
}

// ---------------------------------------------------------------------------
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
async fn request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/request_invariant_off_vs_on";
    with_llamafile_cassette(
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

    let bodies = recorded_interaction_bodies(LLAMAFILE_PROVIDER, scenario);
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
         capture_raw_response is local policy and must never reach the server"
    );
    assert!(!off_request.contains("capture_raw"));
    let first: Value = recorded_json_request(LLAMAFILE_PROVIDER, scenario);
    assert_eq!(first["model"], MODEL);
    assert!(
        first
            .get("stream")
            .is_none_or(|stream| stream == &Value::Bool(false))
    );
}

// ---------------------------------------------------------------------------
// 5: normalization is a pure function of the wire bytes, flag or no flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_llamafile_cassette(
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
            assert_eq!(off.provider, LLAMAFILE_PROVIDER);
            assert_eq!(on.provider, off.provider);
            assert_eq!(on.model, off.model);
            assert_eq!(on.finish_reason(), off.finish_reason());
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
        assert_recorded_envelope(&body, scenario);
        // The transport request id lives in the response headers, not the
        // body, so the body-derived normalization is compared without it.
        let from_wire = openai::CompletionResponse::deserialize(&body)
            .expect("recorded body must be a chat-completions response")
            .normalize(LLAMAFILE_PROVIDER)
            .expect("recorded body must normalize")
            .with_optional_provider_request_id(response.provider_request_id.clone());
        let mut live = normalized_without_raw(response);
        let mut from_wire = normalized_without_raw(from_wire);
        // The response id is a generated per-call id the scrubber
        // placeholders on disk; only a replay compares it exactly. Live, it
        // must still be present on both sides with the wire's shape.
        assert_wire_value_matches(&live, &from_wire, "response_id");
        if matches!(CassetteMode::current(), CassetteMode::Record) {
            live["response_id"] = Value::Null;
            from_wire["response_id"] = Value::Null;
        }
        assert_eq!(
            live, from_wire,
            "the normalized response must equal the normalization of its own \
             wire bytes — capture must not touch any normalized field"
        );
    }
}
