//! Matrix for opt-in raw response capture on mistral.rs's
//! `/v1/chat/completions` route (`CompletionRequest::capture_raw_response` →
//! `CompletionResponse::raw`).
//!
//! # The feature
//!
//! mistral.rs is driven through rig's OpenAI chat-completions client
//! (`openai::CompletionsClient`), so `raw` is the value
//! [`raw_completion`](rig::providers::openai::GenericCompletionModel::raw_completion)
//! would have returned — [`openai::CompletionResponse`] — serialized with
//! `serde_json::to_value`. It is populated only when the request opted in and
//! never reaches the wire.
//!
//! mistral.rs stamps `system_fingerprint: "local"` and the `object`/`created`
//! envelope on every response; none has a home on the normalized
//! [`rig::completion::CompletionResponse`], and cell 3 reads them back through
//! `raw`. (Its per-second throughput fields inside `usage` —
//! `avg_compl_tok_per_sec` and friends — are *not* on `raw`: the shared
//! [`openai::Usage`] does not model them, and raw is the wire type as parsed.)
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_raw_is_none` | flag off (default) | `raw == None` | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `capture_on_raw_round_trips_provider_type` | flag on | `openai::CompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no mistral.rs server in this environment) |
//! | 3 | `capture_on_exposes_envelope_fields` | provider-only fields | `system_fingerprint`/`object`/`created` in `raw` equal the fixture body | unrecorded (no mistral.rs server in this environment) |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no mistral.rs server in this environment) |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | unrecorded (no mistral.rs server in this environment) |
//!
//! Every cell is unrecorded: no mistral.rs server was listening on
//! `127.0.0.1:1234` when this matrix was written, and a fixture is never
//! fabricated. To record: start `mistralrs-server` on that port serving
//! `Qwen/Qwen3-4B` (or export `MISTRALRS_BASE_URL`/`MISTRALRS_MODEL`), remove
//! the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test mistralrs mistralrs::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/mistralrs/raw_capture_matrix/`.

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::openai;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_json_request};

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The OpenAI-compatible client labels its normalized responses `openai`.
const NORMALIZED_PROVIDER: &str = "openai";
/// `/no_think` keeps Qwen3's reasoning trace out of the recording, exactly as
/// the neighbouring mistral.rs cassettes do.
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request(
    model: &openai::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .capture_raw_response(capture_raw)
        .build()
}

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
        body.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded body must report usage"
    );
}

fn recorded_json_interactions(scenario: &str) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario)
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

/// Replay reads the scrubbed fixture back, so volatile fields (`created` → 0,
/// `chatcmpl-…` → placeholders) compare exactly; a live recording proves only
/// that both sides carry the field with the same JSON type.
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
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/capture_off_raw_is_none";
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(model_name());
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
            assert_eq!(response.provider, NORMALIZED_PROVIDER);
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
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/capture_on_raw_round_trips_provider_type";
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/capture_on_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(model_name());
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
            let renormalized = typed
                .normalize(NORMALIZED_PROVIDER)
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
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn capture_on_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/capture_on_exposes_envelope_fields";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/capture_on_exposes_envelope_fields",
        |client| async move {
            let model = client.completion_model(model_name());
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
    for field in ["object", "system_fingerprint", "model"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    for field in ["created", "id"] {
        assert_wire_value_matches(&raw, &body, field);
    }
    let typed = openai::CompletionResponse::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(
        typed.system_fingerprint.as_deref(),
        body["system_fingerprint"].as_str()
    );
    assert_eq!(Some(typed.object.as_str()), body["object"].as_str());
}

// ---------------------------------------------------------------------------
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/request_invariant_off_vs_on";
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(model_name());
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

    let bodies = recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: expected the off and on requests"
    );
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "the flag-on request body must be byte-identical to the flag-off one — \
         capture_raw_response is local policy and must never reach the server"
    );
    assert!(!bodies[0].0.contains("capture_raw"));
    let first: Value = recorded_json_request(MISTRALRS_PROVIDER, scenario);
    assert!(first.get("messages").is_some_and(Value::is_array));
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
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(model_name());
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
            assert_eq!(on.provider, off.provider);
            assert_eq!(on.model, off.model);
            assert_eq!(on.finish_reason(), off.finish_reason());
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
        let from_wire = openai::CompletionResponse::deserialize(&body)
            .expect("recorded body must be a chat-completions response")
            .normalize(NORMALIZED_PROVIDER)
            .expect("recorded body must normalize")
            .with_optional_provider_request_id(response.provider_request_id.clone());
        let mut live = normalized_without_raw(response);
        let mut from_wire = normalized_without_raw(from_wire);
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
