//! Feature matrix for raw provider response capture on the Cohere `/v2/chat`
//! unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: `CompletionModel::completion` serializes the value
//! its inherent `raw_completion` returned — Cohere's own [`CompletionResponse`]
//! — onto [`rig::completion::CompletionResponse::raw`] before `try_into`
//! normalizes it. There is no opt-in and nothing about it reaches the wire;
//! `raw` is `Value::Null` only on a response constructed without a provider
//! payload behind it (hand-built, or persisted before the field existed),
//! never because capture "was not requested".
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
//! | 1 | `raw_roundtrips_cohere_completion_response` | typed access | `cohere::completion::CompletionResponse::deserialize(&*raw)` re-serializes equal, and its `try_into` reproduces the normalized response | recorded |
//! | 2 | `raw_exposes_billing_metadata` | un-normalized fields | `usage.billed_units.*` and `finish_reason` spelled `"COMPLETE"` == fixture, absent from the normalized response | recorded |
//!
//! Every cell is recorded: `COHERE_API_KEY` was available and the seam under
//! test is the plain `/v2/chat` route.
//!
//! Cell 1 also carries the "one story" contract: re-normalizing `raw` by hand
//! lands on the same choice / finish reason / model / usage / identity the
//! typed route reported, so `raw` and the normalized response can never
//! disagree about the turn they describe.
//!
//! The un-normalized fields of choice are `usage.billed_units` (Cohere bills
//! excluding cached input and system overhead, so rig deliberately reports
//! `usage.tokens` instead — the billed figures have no normalized home) and
//! the wire spelling of `finish_reason` (`"COMPLETE"`, which normalizes to
//! rig's `Stop`). Cohere's generation `id` is normalized into `response_id`,
//! so it proves nothing about `raw` on its own.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::prelude::*;
use rig::providers::cohere;
use rig::providers::cohere::completion::CompletionResponse;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

fn request(model: &cohere::CompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
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

fn normalized_without_raw(response: &RigCompletionResponse) -> Value {
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
// 1: typed access is recoverable, and re-normalizes to the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_cohere_completion_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_cohere_completion_response";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_capture_matrix/raw_roundtrips_cohere_completion_response",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;

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

            // And re-normalizing it by hand tells the same story the typed
            // route told: `raw` is additive, never a divergent second view.
            let renormalized: RigCompletionResponse =
                typed.try_into().expect("typed raw should normalize");
            assert_eq!(renormalized.choice, response.choice);
            assert_eq!(renormalized.finish_reason(), response.finish_reason());
            assert_eq!(renormalized.model, response.model);
            assert_eq!(renormalized.usage, response.usage);
            assert_eq!(renormalized.identity(), response.identity());
            assert_eq!(renormalized.provider, response.provider);
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
// 2: un-normalized billing metadata is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_billing_metadata() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_billing_metadata";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_cohere_cassette(
        "raw_capture_matrix/raw_exposes_billing_metadata",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
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
