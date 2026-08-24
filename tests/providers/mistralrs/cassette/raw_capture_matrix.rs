//! Matrix for raw response capture on mistral.rs's `/v1/chat/completions`
//! route ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. mistral.rs is driven through rig's OpenAI
//! chat-completions client (`openai::CompletionsClient`), so every completion
//! the seam returns carries `raw`: the value
//! [`raw_completion`](rig::providers::openai::GenericCompletionModel::raw_completion)
//! would have returned — [`openai::CompletionResponse`] — serialized with
//! `serde_json::to_value` before normalization. Nothing about it is sent to the
//! server. `raw == Value::Null` means only that a `CompletionResponse` was
//! built by hand without a provider response behind it, which no cell here can
//! produce.
//!
//! mistral.rs stamps `system_fingerprint: "local"` and the `object`/`created`
//! envelope on every response; none has a home on the normalized
//! [`rig::completion::CompletionResponse`], and cell 2 reads them back through
//! `raw`. (Its per-second throughput fields inside `usage` —
//! `avg_compl_tok_per_sec` and friends — are *not* on `raw`: the shared
//! [`openai::Usage`] does not model them, and raw is the wire type as parsed.)
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_provider_type` | typed access | `openai::CompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `raw_exposes_envelope_fields` | provider-only fields | `system_fingerprint`/`object`/`created` in `raw` equal the fixture body | unrecorded (no mistral.rs server in this environment) |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the normalized response equals `raw` re-normalized (`normalize`) and the fixture body re-normalized | unrecorded (no mistral.rs server in this environment) |
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
use crate::cassettes::{CassetteMode, recorded_interaction_bodies};

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The OpenAI-compatible client labels its normalized responses `openai`.
const NORMALIZED_PROVIDER: &str = "openai";
/// `/no_think` keeps Qwen3's reasoning trace out of the recording, exactly as
/// the neighbouring mistral.rs cassettes do.
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request(model: &openai::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
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

/// The single recorded interaction of a scenario, request and response parsed
/// as JSON.
fn recorded_json_interaction(scenario: &str) -> (Value, Value) {
    let bodies = recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario);
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
    response.raw = Value::Null;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: raw is the raw_completion value, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let raw = &response.raw;
            let typed = openai::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into openai::CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "openai::CompletionResponse must round-trip through its own serde"
            );
            // The typed view agrees with the normalized one on what the model
            // said, so raw is a superset, not a divergent copy.
            assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
            assert_eq!(response.provider, NORMALIZED_PROVIDER);
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_envelope(&body, scenario);
    openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_exposes_envelope_fields",
        |client| async move {
            let model = client.completion_model(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "created", "system_fingerprint"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized CompletionResponse must not grow a `{field}` field"
                );
            }
            let raw = response.raw;
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (_, body) = recorded_json_interaction(scenario);
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
// 3: raw and the typed route tell one story
// ---------------------------------------------------------------------------

/// The normalized response, with `raw` stripped, must equal the normalization
/// of `raw` read back through the provider type — and equal the normalization
/// of the recorded wire body. Capture is a pure serialization of the value
/// normalization consumed: it neither alters a normalized field nor diverges
/// from the bytes the server sent.
#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            // The transport request id lives in the response headers, not the
            // body, so the raw-derived normalization is given the same one.
            let from_raw = openai::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into openai::CompletionResponse")
                .normalize(NORMALIZED_PROVIDER)
                .expect("raw must normalize")
                .with_optional_provider_request_id(response.provider_request_id.clone());

            assert_eq!(response.provider, NORMALIZED_PROVIDER);
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
    assert_recorded_envelope(&body, scenario);
    let from_wire = openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response")
        .normalize(NORMALIZED_PROVIDER)
        .expect("recorded body must normalize")
        .with_optional_provider_request_id(response.provider_request_id.clone());
    let mut live = normalized_without_raw(response);
    let mut from_wire = normalized_without_raw(from_wire);
    // The response id is a generated per-call id the scrubber placeholders
    // on disk; only a replay compares it exactly. Live, it must still be
    // present on both sides with the wire's shape.
    assert_wire_value_matches(&live, &from_wire, "response_id");
    if matches!(CassetteMode::current(), CassetteMode::Record) {
        live["response_id"] = Value::Null;
        from_wire["response_id"] = Value::Null;
    }
    assert_eq!(
        live, from_wire,
        "the normalized response must equal the normalization of the wire bytes \
         it was built from"
    );
}
