//! Matrix for opt-in raw response capture on ChatGPT's blocking `/responses`
//! path (`CompletionRequest::capture_raw_response` → `CompletionResponse::raw`).
//!
//! # The feature
//!
//! `raw` is the value
//! [`ResponsesCompletionModel::raw_completion`](rig::providers::chatgpt::ResponsesCompletionModel::raw_completion)
//! would have returned — the Responses API's
//! [`CompletionResponse`](rig::providers::openai::responses_api::CompletionResponse),
//! reassembled from the terminal `response.completed` event of the SSE body
//! ChatGPT answers even a non-streaming request with — serialized with
//! `serde_json::to_value`. It is populated only when the request opted in and
//! never reaches the wire. The provider's empty-`output` fallback (the
//! terminal event carrying no items, the content rebuilt from earlier events)
//! captures the same `raw_response` on its branch too; see
//! `raw_completion_parity_matrix` for that state.
//!
//! The Responses envelope carries fields the normalized
//! [`rig::completion::CompletionResponse`] has no home for — `object`,
//! `status`, `created_at` — and cell 3 reads them back through `raw`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_raw_is_none` | flag off (default) | `raw == None` | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `capture_on_raw_round_trips_provider_type` | flag on | `responses_api::CompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `capture_on_exposes_response_envelope` | provider-only fields | `object`/`status`/`created_at` in `raw` equal the terminal `response.completed` frame | unrecorded (no CHATGPT credentials in this environment) |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no CHATGPT credentials in this environment) |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | off/on responses normalize their own terminal frame identically; only `raw` differs | unrecorded (no CHATGPT credentials in this environment) |
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
use rig::providers::openai::responses_api;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(
    model: &chatgpt::ResponsesCompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .capture_raw_response(capture_raw)
        .build()
}

/// The premise every cell rests on: the recorded SSE body ends with a
/// `response.completed` frame whose `response` carries usage. Returns that
/// terminal `response` object.
fn recorded_terminal_response(scenario: &str) -> Value {
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
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/capture_off_raw_is_none";
    with_chatgpt_cassette(
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
            assert_eq!(response.provider, CHATGPT_PROVIDER);
        },
    )
    .await;

    recorded_terminal_response(scenario);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_completion value, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/capture_on_raw_round_trips_provider_type";
    with_chatgpt_cassette(
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
            let typed = responses_api::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into responses_api::CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "responses_api::CompletionResponse must round-trip through its own serde"
            );

            // The typed view normalizes to the same surface `completion`
            // produced — raw is the value the seam normalized.
            let renormalized = typed
                .normalize(CHATGPT_PROVIDER)
                .expect("typed raw must normalize");
            assert_eq!(
                normalized_without_raw(renormalized),
                normalized_without_raw(response),
                "normalizing the captured raw must reproduce the normalized response"
            );
        },
    )
    .await;

    let terminal = recorded_terminal_response(scenario);
    responses_api::CompletionResponse::deserialize(&terminal)
        .expect("recorded terminal response must be a Responses envelope");
}

// ---------------------------------------------------------------------------
// 3: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn capture_on_exposes_response_envelope() {
    let scenario = "raw_capture_matrix/capture_on_exposes_response_envelope";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_capture_matrix/capture_on_exposes_response_envelope",
        |client| async move {
            let model = client.completion_model(MODEL);
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
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/request_invariant_off_vs_on";
    with_chatgpt_cassette(
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

    let bodies = recorded_interaction_bodies(CHATGPT_PROVIDER, scenario);
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
         capture_raw_response is local policy and must never reach ChatGPT"
    );
    assert!(!off_request.contains("capture_raw"));
    let first: Value = serde_json::from_str(off_request).expect("recorded request should be JSON");
    assert_eq!(first["model"], MODEL);
}

// ---------------------------------------------------------------------------
// 5: normalization is a pure function of the wire bytes, flag or no flag
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
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
    let bodies = recorded_interaction_bodies(CHATGPT_PROVIDER, scenario);
    assert_eq!(bodies.len(), 2, "{scenario}: expected off and on turns");

    for ((_, body), response) in bodies.into_iter().zip(responses) {
        // Each interaction is its own SSE body; take its terminal frame.
        let terminal = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("data:"))
            .map(str::trim)
            .filter(|payload| *payload != "[DONE]")
            .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
            .rev()
            .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
            .map(|frame| frame["response"].clone())
            .unwrap_or_else(|| {
                panic!("{scenario}: each interaction must end with response.completed")
            });
        let from_wire = responses_api::CompletionResponse::deserialize(&terminal)
            .expect("recorded terminal response must be a Responses envelope")
            .normalize(CHATGPT_PROVIDER)
            .expect("recorded terminal response must normalize");

        let mut live = normalized_without_raw(response);
        let mut from_wire = normalized_without_raw(from_wire);
        // Generated ids are placeholdered on disk; only a replay compares
        // them exactly, a live recording checks presence and shape.
        for id in ["response_id", "message_id"] {
            assert_wire_value_matches(&live, &from_wire, id);
            if matches!(CassetteMode::current(), CassetteMode::Record) {
                live[id] = Value::Null;
                from_wire[id] = Value::Null;
            }
        }
        assert_eq!(
            live, from_wire,
            "the normalized response must equal the normalization of its own \
             wire bytes — capture must not touch any normalized field"
        );
    }
}
