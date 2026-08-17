//! Feature matrix for opt-in raw provider response capture on the Cohere
//! `/v2/chat` unary seam.
//!
//! # The feature
//!
//! [`rig::completion::CompletionRequest::capture_raw_response`] is local
//! policy: when set, `CompletionModel::completion` serializes the value its
//! inherent `raw_completion` would have returned — Cohere's own
//! [`CompletionResponse`] — onto [`rig::completion::CompletionResponse::raw`]
//! before `try_into` normalizes it. Off (the default) it stays `None`, and the
//! flag never reaches the wire either way.
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the normalized response. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded turn must have finished `COMPLETE` and carry
//! the billing metadata the cells read.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `flag_off_leaves_raw_unset` | default (`false`) | `raw.is_none()` | recorded |
//! | 2 | `flag_on_roundtrips_cohere_completion_response` | `true` → typed access | `cohere::completion::CompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_billing_metadata` | `true` → un-normalized fields | `usage.billed_units.*` and `finish_reason` spelled `"COMPLETE"` == fixture, absent from the normalized response | recorded |
//! | 4 | `request_bytes_invariant_across_flag` | request boundary | recorded off/on request bodies byte-identical | recorded |
//! | 5 | `normalized_fields_invariant_across_flag` | normalized fields | choice / finish reason / model / prompt usage identical off vs on | recorded |
//!
//! Every cell is recorded: `COHERE_API_KEY` was available and the seam under
//! test is the plain `/v2/chat` route.
//!
//! Cells 4 and 5 record one scenario each with **two** interactions — the
//! flag-off request first, then the flag-on twin — because the invariant is
//! between the two; the harness replays interactions in order.
//!
//! The un-normalized fields of choice are `usage.billed_units` (Cohere bills
//! excluding cached input and system overhead, so rig deliberately reports
//! `usage.tokens` instead — the billed figures have no normalized home) and
//! the wire spelling of `finish_reason` (`"COMPLETE"`, which normalizes to
//! rig's `Stop`). Cohere's generation `id` is normalized into `response_id`,
//! so it proves nothing about `raw` on its own.

use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::cohere;
use rig::providers::cohere::completion::CompletionResponse;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

fn request(model: &cohere::CompletionModel, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .capture_raw_response(capture)
        .build()
}

/// The premise every cell rests on: the recorded body is a `/v2/chat` answer
/// that finished `COMPLETE` and carries `usage.billed_units`.
fn assert_recorded_complete_turn(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.get("finish_reason"),
        Some(&Value::String("COMPLETE".to_string())),
        "{scenario}: the recorded turn should have finished COMPLETE"
    );
    assert!(
        body.pointer("/usage/billed_units/input_tokens")
            .and_then(Value::as_f64)
            .is_some(),
        "{scenario}: the recorded usage should carry billed_units, the un-normalized field \
         this matrix reads through `raw`"
    );
    body
}

fn assert_request_body_never_names_the_flag(scenario: &str, body: &str) {
    for spelling in ["capture_raw_response", "captureRawResponse"] {
        assert!(
            !body.contains(spelling),
            "{scenario}: the recorded request body must not carry {spelling:?}; the flag is \
             `#[serde(skip)]` local policy and must never reach Cohere"
        );
    }
}

fn normalized_without_raw(response: &rig::completion::CompletionResponse) -> Value {
    let mut value = serde_json::to_value(response).expect("normalized response serializes");
    value
        .as_object_mut()
        .expect("normalized response is an object")
        .remove("raw");
    value
}

fn contains_key(value: &Value, needle: &str) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| key == needle || contains_key(value, needle)),
        Value::Array(items) => items.iter().any(|item| contains_key(item, needle)),
        _ => false,
    }
}

/// Cohere's counters are `f64` on rig's wire type, so a captured `6.0`
/// must be compared numerically against the fixture's `6`.
fn number_at(value: &Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer).and_then(Value::as_f64)
}

// ---------------------------------------------------------------------------
// 1: default off
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_off_leaves_raw_unset() {
    const SCENARIO: &str = "raw_capture_matrix/flag_off_leaves_raw_unset";
    with_cohere_cassette(
        "raw_capture_matrix/flag_off_leaves_raw_unset",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(request(&model, false))
                .await
                .expect("completion should succeed");

            assert!(
                response.raw.is_none(),
                "capture was not requested, so the normalized response must not carry raw"
            );
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            // Cohere's `/v2/chat` payload names no model, so the normalized field
            // stays unset regardless of the flag.
            assert_eq!(response.model, None);
            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "Cohere's generation id is normalized as the response id"
            );
        },
    )
    .await;

    assert_recorded_complete_turn(SCENARIO);
    let (request_body, _) = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_request_body_never_names_the_flag(SCENARIO, &request_body);
}

// ---------------------------------------------------------------------------
// 2: on → typed access is recoverable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_roundtrips_cohere_completion_response() {
    const SCENARIO: &str = "raw_capture_matrix/flag_on_roundtrips_cohere_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_capture_matrix/flag_on_roundtrips_cohere_completion_response",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("capture was requested, so raw must be populated");

            let typed = CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into Cohere's CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "cohere::completion::CompletionResponse must round-trip through its own \
             Serialize/Deserialize"
            );

            // The typed value agrees with the normalized fields next to it.
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            assert_eq!(
                typed
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.tokens.as_ref())
                    .and_then(|tokens| tokens.input_tokens)
                    .map(|tokens| tokens as u64),
                Some(response.usage.input_tokens)
            );
            *sink.lock().expect("observation lock") = Some(raw.clone());
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_complete_turn(SCENARIO);
    // Cohere's generation id is not scrubbed, so the captured value can be
    // pinned to the recorded bytes.
    assert_eq!(raw.get("id"), body.get("id"));
}

// ---------------------------------------------------------------------------
// 3: on → un-normalized billing metadata is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_billing_metadata() {
    const SCENARIO: &str = "raw_capture_matrix/flag_on_exposes_billing_metadata";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_capture_matrix/flag_on_exposes_billing_metadata",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("capture was requested, so raw must be populated");
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized response provably lacks these: billed units have no
            // normalized home, and the finish reason reaches it only as rig's
            // vocabulary.
            let normalized = normalized_without_raw(&response);
            assert!(!contains_key(&normalized, "billed_units"));
            assert_ne!(
                normalized.get("finish_reason"),
                Some(&Value::String("COMPLETE".to_string())),
                "the normalized finish reason is rig's spelling, not Cohere's"
            );
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_complete_turn(SCENARIO);
    assert_eq!(
        raw.get("finish_reason"),
        Some(&Value::String("COMPLETE".to_string())),
        "raw keeps Cohere's own finish_reason spelling"
    );
    for pointer in [
        "/usage/billed_units/input_tokens",
        "/usage/billed_units/output_tokens",
        "/usage/tokens/input_tokens",
        "/usage/tokens/output_tokens",
    ] {
        assert_eq!(
            number_at(&raw, pointer),
            number_at(&body, pointer),
            "raw must carry {pointer} exactly as the wire sent it"
        );
        assert!(
            number_at(&body, pointer).is_some(),
            "{SCENARIO}: the recorded body should carry {pointer}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4: the request boundary never sees the flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_bytes_invariant_across_flag() {
    const SCENARIO: &str = "raw_capture_matrix/request_bytes_invariant_across_flag";
    with_cohere_cassette(
        "raw_capture_matrix/request_bytes_invariant_across_flag",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
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

    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: the cell records the flag-off request and then its flag-on twin"
    );
    let (off_request, _) = &bodies[0];
    let (on_request, _) = &bodies[1];
    assert_eq!(
        off_request, on_request,
        "the flag-on request must be byte-identical to the flag-off request; the flag is local \
         policy and never reaches Cohere"
    );
    assert_request_body_never_names_the_flag(SCENARIO, on_request);
}

// ---------------------------------------------------------------------------
// 5: normalized fields mean the same thing with or without capture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn normalized_fields_invariant_across_flag() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_invariant_across_flag";
    with_cohere_cassette(
        "raw_capture_matrix/normalized_fields_invariant_across_flag",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
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

            assert_eq!(off.choice, on.choice);
            assert_eq!(off.finish_reason(), on.finish_reason());
            assert_eq!(off.finish_reason(), Some(FinishReason::Stop));
            assert_eq!(off.model, on.model);
            assert_eq!(off.provider, on.provider);
            assert_eq!(off.message_id, on.message_id);
            assert_eq!(off.provider_request_id, on.provider_request_id);
            // Identical request bytes tokenize identically; the output side is the
            // model's to vary, so only the prompt count is pinned.
            assert_eq!(off.usage.input_tokens, on.usage.input_tokens);
            // Two live turns get two generation ids; both are populated, neither
            // is shaped by the flag.
            assert!(off.response_id.as_deref().is_some_and(|id| !id.is_empty()));
            assert!(on.response_id.as_deref().is_some_and(|id| !id.is_empty()));
        },
    )
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(bodies.len(), 2, "{SCENARIO}: off then on");
    for (request_body, response_body) in &bodies {
        assert_request_body_never_names_the_flag(SCENARIO, request_body);
        let response: Value =
            serde_json::from_str(response_body).expect("recorded response should be JSON");
        assert_eq!(
            response.get("finish_reason"),
            Some(&Value::String("COMPLETE".to_string())),
            "{SCENARIO}: both recorded turns should have finished COMPLETE"
        );
    }
}
