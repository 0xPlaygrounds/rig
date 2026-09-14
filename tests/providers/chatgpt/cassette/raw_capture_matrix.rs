//! Matrix for raw response capture on ChatGPT's blocking `/responses` path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the seam returns carries `raw`: the
//! value
//! [`ResponsesCompletionModel::raw_completion`](rig::providers::chatgpt::ResponsesCompletionModel::raw_completion)
//! would have returned — the Responses API's
//! [`CompletionResponse`](rig::providers::openai_compatible::responses_api::CompletionResponse),
//! reassembled from the terminal `response.completed` event of the SSE body
//! ChatGPT answers even a non-streaming request with — serialized with
//! `serde_json::to_value` before normalization. Nothing about it is sent to
//! ChatGPT. `raw == Value::Null` means only that a `CompletionResponse` was
//! built by hand without a provider response behind it, which no cell here can
//! produce. The provider's empty-`output` fallback (the terminal event carrying
//! no items, the content rebuilt from earlier events) captures the same
//! `raw_response` on its branch too; see `raw_completion_parity_matrix` for
//! that state.
//!
//! The Responses envelope carries fields the normalized
//! [`rig::completion::CompletionResponse`] has no home for — `object`,
//! `status`, `created_at` — and cell 2 reads them back through `raw`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_provider_type` | typed access | `responses_api::CompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `raw_exposes_response_envelope` | provider-only fields | `object`/`status`/`created_at` in `raw` equal the terminal `response.completed` frame | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the normalized response equals `raw` re-normalized (`normalize`) and the terminal frame re-normalized | unrecorded (no CHATGPT credentials in this environment) |
//!
//! Every cell is unrecorded: neither `CHATGPT_ACCESS_TOKEN`/`CHATGPT_ACCOUNT_ID`
//! nor a usable ChatGPT OAuth cache was present when this matrix was written,
//! and a fixture is never fabricated. The bodies are complete and would pass
//! once recorded. To record: export `CHATGPT_ACCESS_TOKEN` and
//! `CHATGPT_ACCOUNT_ID` (the harness placeholders both on disk), remove the
//! `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test chatgpt chatgpt::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/chatgpt/raw_capture_matrix/`.

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai_compatible::responses_api;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &chatgpt::ResponsesCompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The premise every cell rests on: the scenario recorded exactly one
/// interaction whose SSE body ends with a `response.completed` frame whose
/// `response` carries usage. Returns that terminal `response` object.
fn recorded_terminal_response(scenario: &str) -> Value {
    assert_eq!(
        recorded_interaction_bodies(CHATGPT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let frames = recorded_sse_json_frames(CHATGPT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded body must carry a response.completed frame")
        });
    let response = terminal["response"].clone();
    assert!(
        response.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the terminal response must report usage"
    );
    assert_eq!(
        response.get("object").and_then(Value::as_str),
        Some("response"),
        "{scenario}: the terminal response must be a Responses envelope"
    );
    assert_eq!(
        response.get("status").and_then(Value::as_str),
        Some("completed"),
        "{scenario}: the terminal response must be completed"
    );
    response
}

/// Replay reads the scrubbed fixture back, so volatile fields (`created_at`
/// → 0, ids → placeholders) compare exactly; a live recording proves only
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
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    with_chatgpt_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            let typed = responses_api::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into responses_api::CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "responses_api::CompletionResponse must round-trip through its own serde"
            );

            // The typed view agrees with the normalized one on what the model
            // said, so raw is a superset, not a divergent copy.
            assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
            assert_eq!(response.provider, CHATGPT_PROVIDER);
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let terminal = recorded_terminal_response(scenario);
    responses_api::CompletionResponse::deserialize(&terminal)
        .expect("recorded terminal response must be a Responses envelope");
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_exposes_response_envelope() {
    let scenario = "raw_capture_matrix/raw_exposes_response_envelope";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_capture_matrix/raw_exposes_response_envelope",
        |client| async move {
            let model = client.completion_model(MODEL);
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
    let terminal = recorded_terminal_response(scenario);
    for field in ["object", "status", "model"] {
        assert_eq!(
            raw.get(field),
            terminal.get(field),
            "raw.{field} must equal the recorded terminal frame"
        );
    }
    assert_wire_value_matches(&raw, &terminal, "created_at");
    assert_wire_value_matches(&raw, &terminal, "id");
    let typed = responses_api::CompletionResponse::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(typed.status, responses_api::ResponseStatus::Completed);
    assert!(matches!(
        typed.object,
        responses_api::ResponseObject::Response
    ));
}

// ---------------------------------------------------------------------------
// 3: raw and the typed route tell one story
// ---------------------------------------------------------------------------

/// The normalized response, with `raw` stripped, must equal the normalization
/// of `raw` read back through the Responses wire type — and equal the
/// normalization of the recorded terminal `response.completed` frame. Capture
/// is a pure serialization of the value normalization consumed: it neither
/// alters a normalized field nor diverges from the bytes ChatGPT sent.
#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            // ChatGPT reads no transport request-id header, so the whole
            // identity lives in the body and needs no reassembly here.
            let from_raw = responses_api::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into responses_api::CompletionResponse")
                .normalize(CHATGPT_PROVIDER)
                .expect("raw must normalize");

            assert_eq!(response.provider, CHATGPT_PROVIDER);
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
    let terminal = recorded_terminal_response(scenario);
    let from_wire = responses_api::CompletionResponse::deserialize(&terminal)
        .expect("recorded terminal response must be a Responses envelope")
        .normalize(CHATGPT_PROVIDER)
        .expect("recorded terminal response must normalize");

    let mut live = normalized_without_raw(response);
    let mut from_wire = normalized_without_raw(from_wire);
    // Generated ids are placeholdered on disk; only a replay compares them
    // exactly, a live recording checks presence and shape.
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
