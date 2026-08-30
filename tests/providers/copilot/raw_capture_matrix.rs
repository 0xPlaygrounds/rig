//! Matrix for raw response capture on both Copilot blocking routes
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the seam returns carries `raw`: the
//! value
//! [`CompletionModel::raw_completion`](rig::providers::copilot::CompletionModel::raw_completion)
//! would have returned — the route-tagged
//! [`CopilotCompletionResponse`](rig::providers::copilot::CopilotCompletionResponse)
//! (`{"api":"chat", …}` wrapping [`rig::providers::copilot::ChatCompletionResponse`] on the
//! chat-completions route, `{"api":"responses", …}` wrapping
//! [`rig::providers::copilot::ResponsesCompletionResponse`] on the Responses route) — serialized
//! with `serde_json::to_value` before normalization. Nothing about it is sent
//! to Copilot. `raw == Value::Null` means only that a `CompletionResponse` was
//! built by hand without a provider response behind it, which no cell here can
//! produce. Because the tag rides along, a caller reads raw back into the same
//! enum the typed escape hatch yields, without knowing the route in advance.
//!
//! Provider-only fields per route: the chat route's `system_fingerprint`
//! (Copilot omits `object`/`created` on this route — the wire type tolerates
//! that — and its `copilot_usage` block is not modelled by the shared
//! chat-completions type, so it is not on `raw` either: raw is the wire type
//! as parsed, not the bytes); the Responses route's `object`/`status`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_round_trips_provider_type` | chat route, typed access | `CopilotCompletionResponse::deserialize(&*raw)` is `Chat(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_raw_exposes_system_fingerprint` | chat route, provider-only field | `raw.system_fingerprint` equals the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `chat_normalized_fields_equal_raw_renormalized` | chat route, normalized view | the normalized response equals `raw` re-normalized (`normalize`) and the fixture body re-normalized | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `responses_raw_round_trips_provider_type` | responses route, typed access | `CopilotCompletionResponse::deserialize(&*raw)` is `Responses(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 5 | `responses_raw_exposes_envelope` | responses route, provider-only fields | `raw.object`/`raw.status` equal the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 6 | `responses_normalized_fields_equal_raw_renormalized` | responses route, normalized view | the normalized response equals `raw` re-normalized (`normalize`) and the fixture body re-normalized | unrecorded (no COPILOT credentials in this environment) |
//!
//! Every cell is unrecorded: none of `GITHUB_COPILOT_API_KEY`,
//! `COPILOT_API_KEY`, `COPILOT_GITHUB_ACCESS_TOKEN`/`GITHUB_TOKEN` nor a Copilot
//! OAuth cache was present when this matrix was written, and a fixture is
//! never fabricated. To record: export `GITHUB_COPILOT_API_KEY` (the harness
//! placeholders it on disk), remove the `#[ignore]` attributes, flip the table
//! to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test copilot copilot::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/copilot/raw_capture_matrix/`.

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::copilot::{self, CopilotCompletionResponse};
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_interaction_bodies};
use crate::copilot::with_copilot_cassette;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &copilot::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The single recorded interaction of a scenario, request and response parsed
/// as JSON.
fn recorded_json_interaction(scenario: &str) -> (Value, Value) {
    let bodies = recorded_interaction_bodies(COPILOT_PROVIDER, scenario);
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

/// Chat-route premise: the recorded body is a chat-completions response with
/// usage and a `system_fingerprint`.
fn assert_recorded_chat_body(body: &Value, scenario: &str) {
    assert!(
        body.get("choices").is_some_and(Value::is_array),
        "{scenario}: the recorded body must be a chat-completions response"
    );
    assert!(
        body.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded body must report usage"
    );
    assert!(
        body.get("system_fingerprint")
            .and_then(Value::as_str)
            .is_some(),
        "{scenario}: the recorded body must carry `system_fingerprint` — without \
         it this cell cannot prove raw exposes a provider-only field"
    );
}

/// Responses-route premise: the recorded body is a completed Responses
/// envelope with usage.
fn assert_recorded_responses_body(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("response"),
        "{scenario}: the recorded body must be a Responses envelope"
    );
    assert_eq!(
        body.get("status").and_then(Value::as_str),
        Some("completed"),
        "{scenario}: the recorded turn must be completed"
    );
    assert!(
        body.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the recorded body must report usage"
    );
}

/// Replay reads the scrubbed fixture back, so volatile fields (`created` → 0,
/// `fp_…`/`chatcmpl-…`/`resp_…` → placeholders) compare exactly; a live
/// recording proves only that both sides carry the field with the same JSON
/// type.
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

/// Compares a route's normalized response with the normalization of the
/// recorded body it was built from, masking only the generated ids the
/// scrubber placeholders (and only while recording).
fn assert_normalizes_like_own_wire(live: RigCompletionResponse, from_wire: RigCompletionResponse) {
    let mut live = normalized_without_raw(live);
    let mut from_wire = normalized_without_raw(from_wire);
    for id in ["response_id", "message_id"] {
        assert_wire_value_matches(&live, &from_wire, id);
        if matches!(CassetteMode::current(), CassetteMode::Record) {
            live[id] = Value::Null;
            from_wire[id] = Value::Null;
        }
    }
    assert_eq!(
        live, from_wire,
        "the normalized response must equal the normalization of the wire bytes \
         it was built from"
    );
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/chat_raw_round_trips_provider_type";
    with_copilot_cassette(
        "raw_capture_matrix/chat_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            assert_eq!(raw["api"], "chat", "the route tag rides along on raw");
            let typed = CopilotCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotCompletionResponse");
            assert!(
                matches!(typed, CopilotCompletionResponse::Chat(_)),
                "chat-route raw reads back as the Chat variant"
            );
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "CopilotCompletionResponse must round-trip through its own serde"
            );
            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_chat_body(&body, scenario);
    rig::providers::copilot::ChatCompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_exposes_system_fingerprint() {
    let scenario = "raw_capture_matrix/chat_raw_exposes_system_fingerprint";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/chat_raw_exposes_system_fingerprint",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            assert!(
                normalized.get("system_fingerprint").is_none(),
                "normalized CompletionResponse must not grow a `system_fingerprint` field"
            );
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
    assert_recorded_chat_body(&body, scenario);
    // `fp_…` fingerprints are placeholdered on disk like generated ids.
    assert_wire_value_matches(&raw, &body, "system_fingerprint");
    assert_eq!(raw["model"], body["model"]);
    let CopilotCompletionResponse::Chat(typed) =
        CopilotCompletionResponse::deserialize(&raw).expect("raw must deserialize")
    else {
        panic!("chat-route raw must read back as the Chat variant");
    };
    assert!(
        typed.system_fingerprint.is_some(),
        "the typed raw carries the fingerprint the wire sent"
    );
}

/// The normalized response, with `raw` stripped, must equal the normalization
/// of `raw` read back through the route-tagged enum — and equal the
/// normalization of the recorded wire body. Capture is a pure serialization of
/// the value normalization consumed on this route.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/chat_normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/chat_normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            // The transport id is a response header, not body: reattach it so
            // the raw-derived normalization is comparable field-for-field.
            let from_raw = CopilotCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotCompletionResponse")
                .normalize(COPILOT_PROVIDER)
                .expect("raw must normalize")
                .with_optional_provider_request_id(response.provider_request_id.clone());

            assert_eq!(response.provider, COPILOT_PROVIDER);
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
    assert_recorded_chat_body(&body, scenario);
    let from_wire = rig::providers::copilot::ChatCompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response")
        .normalize(COPILOT_PROVIDER)
        .expect("recorded body must normalize")
        .with_optional_provider_request_id(response.provider_request_id.clone());
    assert_normalizes_like_own_wire(response, from_wire);
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/responses_raw_round_trips_provider_type";
    with_copilot_cassette(
        "raw_capture_matrix/responses_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            assert_eq!(raw["api"], "responses", "the route tag rides along on raw");
            let typed = CopilotCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotCompletionResponse");
            assert!(
                matches!(typed, CopilotCompletionResponse::Responses(_)),
                "responses-route raw reads back as the Responses variant"
            );
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "CopilotCompletionResponse must round-trip through its own serde"
            );
            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_responses_body(&body, scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_exposes_envelope() {
    let scenario = "raw_capture_matrix/responses_raw_exposes_envelope";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/responses_raw_exposes_envelope",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "status", "created_at"] {
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
    assert_recorded_responses_body(&body, scenario);
    for field in ["object", "status", "model"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    assert_wire_value_matches(&raw, &body, "created_at");
    let CopilotCompletionResponse::Responses(typed) =
        CopilotCompletionResponse::deserialize(&raw).expect("raw must deserialize")
    else {
        panic!("responses-route raw must read back as the Responses variant");
    };
    assert_eq!(
        typed.status,
        rig::providers::openai_compatible::responses_api::ResponseStatus::Completed
    );
}

/// The Responses-route twin of `chat_normalized_fields_equal_raw_renormalized`.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/responses_normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/responses_normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            // On this route the wire type has its own `provider_request_id`
            // slot, stamped from the header by the request driver; its
            // `Serialize` mirrors the wire body and never emits it, so raw
            // reads back without it and gets the live one reattached — the
            // same reassembly the typed escape hatch contracts.
            let from_raw = CopilotCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotCompletionResponse")
                .normalize(COPILOT_PROVIDER)
                .expect("raw must normalize")
                .with_optional_provider_request_id(response.provider_request_id.clone());

            assert_eq!(response.provider, COPILOT_PROVIDER);
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
    assert_recorded_responses_body(&body, scenario);
    let from_wire = rig::providers::copilot::ResponsesCompletionResponse::deserialize(&body)
        .expect("recorded body must be a Responses envelope")
        .normalize(COPILOT_PROVIDER)
        .expect("recorded body must normalize")
        .with_optional_provider_request_id(response.provider_request_id.clone());
    assert_normalizes_like_own_wire(response, from_wire);
}
