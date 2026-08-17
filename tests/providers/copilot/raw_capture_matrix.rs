//! Matrix for opt-in raw response capture on both Copilot blocking routes
//! (`CompletionRequest::capture_raw_response` → `CompletionResponse::raw`).
//!
//! # The feature
//!
//! `raw` is the value
//! [`CompletionModel::raw_completion`](rig::providers::copilot::CompletionModel::raw_completion)
//! would have returned — the route-tagged
//! [`CopilotCompletionResponse`](rig::providers::copilot::CopilotCompletionResponse)
//! (`{"api":"chat", …}` wrapping [`openai::CompletionResponse`] on the
//! chat-completions route, `{"api":"responses", …}` wrapping
//! [`responses_api::CompletionResponse`] on the Responses route) — serialized
//! with `serde_json::to_value`. It is populated only when the request opted in
//! and never reaches the wire. Because the tag rides along, a caller reads raw
//! back into the same enum the typed escape hatch yields, without knowing the
//! route in advance.
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
//! | 1 | `chat_capture_off_raw_is_none` | chat route, flag off | `raw == None` | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_capture_on_raw_round_trips_provider_type` | chat route, flag on | `CopilotCompletionResponse::deserialize(&*raw)` is `Chat(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `chat_capture_on_exposes_system_fingerprint` | chat route, provider-only field | `raw.system_fingerprint` equals the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `chat_request_invariant_off_vs_on` | chat route, on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no COPILOT credentials in this environment) |
//! | 5 | `chat_normalized_fields_identical_off_vs_on` | chat route, normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | unrecorded (no COPILOT credentials in this environment) |
//! | 6 | `responses_capture_off_raw_is_none` | responses route, flag off | `raw == None` | unrecorded (no COPILOT credentials in this environment) |
//! | 7 | `responses_capture_on_raw_round_trips_provider_type` | responses route, flag on | `CopilotCompletionResponse::deserialize(&*raw)` is `Responses(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 8 | `responses_capture_on_exposes_envelope` | responses route, provider-only fields | `raw.object`/`raw.status` equal the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 9 | `responses_request_invariant_off_vs_on` | responses route, on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no COPILOT credentials in this environment) |
//! | 10 | `responses_normalized_fields_identical_off_vs_on` | responses route, normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | unrecorded (no COPILOT credentials in this environment) |
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
use rig::providers::openai;
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_json_request};
use crate::copilot::with_copilot_cassette;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(
    model: &copilot::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .capture_raw_response(capture_raw)
        .build()
}

fn recorded_json_interactions(scenario: &str) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(COPILOT_PROVIDER, scenario)
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
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

/// Compares a route's normalized response with the normalization of its own
/// recorded body, masking only the generated ids the scrubber placeholders
/// (and only while recording).
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
        "the normalized response must equal the normalization of its own wire \
         bytes — capture must not touch any normalized field"
    );
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/chat_capture_off_raw_is_none";
    with_copilot_cassette(
        "raw_capture_matrix/chat_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
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
            assert_eq!(response.provider, COPILOT_PROVIDER);
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_chat_body(&body, scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/chat_capture_on_raw_round_trips_provider_type";
    with_copilot_cassette(
        "raw_capture_matrix/chat_capture_on_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("raw must be populated when capture was requested");
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
            let renormalized = typed
                .normalize(COPILOT_PROVIDER)
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
    assert_recorded_chat_body(&body, scenario);
    openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_capture_on_exposes_system_fingerprint() {
    let scenario = "raw_capture_matrix/chat_capture_on_exposes_system_fingerprint";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/chat_capture_on_exposes_system_fingerprint",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            assert!(
                normalized.get("system_fingerprint").is_none(),
                "normalized CompletionResponse must not grow a `system_fingerprint` field"
            );
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

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/chat_request_invariant_off_vs_on";
    with_copilot_cassette(
        "raw_capture_matrix/chat_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
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

    let bodies = recorded_interaction_bodies(COPILOT_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: expected the off and on requests"
    );
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "the flag-on request body must be byte-identical to the flag-off one — \
         capture_raw_response is local policy and must never reach Copilot"
    );
    assert!(!bodies[0].0.contains("capture_raw"));
    let first: Value = recorded_json_request(COPILOT_PROVIDER, scenario);
    assert_eq!(first["model"], CHAT_MODEL);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/chat_normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/chat_normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
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
        assert_recorded_chat_body(&body, scenario);
        // The transport id is a response header, not body: reattach it so the
        // body-derived normalization is comparable.
        let from_wire = openai::CompletionResponse::deserialize(&body)
            .expect("recorded body must be a chat-completions response")
            .normalize(COPILOT_PROVIDER)
            .expect("recorded body must normalize")
            .with_optional_provider_request_id(response.provider_request_id.clone());
        assert_normalizes_like_own_wire(response, from_wire);
    }
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/responses_capture_off_raw_is_none";
    with_copilot_cassette(
        "raw_capture_matrix/responses_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
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
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_responses_body(&body, scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/responses_capture_on_raw_round_trips_provider_type";
    with_copilot_cassette(
        "raw_capture_matrix/responses_capture_on_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("raw must be populated when capture was requested");
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
            let renormalized = typed
                .normalize(COPILOT_PROVIDER)
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
    assert_recorded_responses_body(&body, scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_capture_on_exposes_envelope() {
    let scenario = "raw_capture_matrix/responses_capture_on_exposes_envelope";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/responses_capture_on_exposes_envelope",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "status", "created_at"] {
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
        rig::providers::openai::responses_api::ResponseStatus::Completed
    );
}

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/responses_request_invariant_off_vs_on";
    with_copilot_cassette(
        "raw_capture_matrix/responses_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
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

    let bodies = recorded_interaction_bodies(COPILOT_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: expected the off and on requests"
    );
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "the flag-on request body must be byte-identical to the flag-off one — \
         capture_raw_response is local policy and must never reach Copilot"
    );
    assert!(!bodies[0].0.contains("capture_raw"));
    let first: Value = recorded_json_request(COPILOT_PROVIDER, scenario);
    assert_eq!(first["model"], RESPONSES_MODEL);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/responses_normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_capture_matrix/responses_normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
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
        assert_recorded_responses_body(&body, scenario);
        // On this route the wire type has its own `provider_request_id` slot,
        // stamped from the header by the request driver — the body-parsed
        // copy has none, so reattach the live one for a like-for-like compare.
        let from_wire =
            rig::providers::openai::responses_api::CompletionResponse::deserialize(&body)
                .expect("recorded body must be a Responses envelope")
                .normalize(COPILOT_PROVIDER)
                .expect("recorded body must normalize")
                .with_optional_provider_request_id(response.provider_request_id.clone());
        assert_normalizes_like_own_wire(response, from_wire);
    }
}
