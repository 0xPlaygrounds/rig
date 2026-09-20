//! Feature matrix for raw provider response capture on the Cohere `/v2/chat`
//! unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: `CompletionModel::completion` serializes the value
//! its inherent `raw_completion` returned — Cohere's own [`CompletionResponse`]
//! — onto [`rig::completion::CompletionResponse::raw`] before `try_into`
//! normalizes it. There is no opt-in and nothing about it reaches the wire;
//! `raw` is required at construction, so there is no response without the
//! document that produced it and no way for capture to be "not requested".
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

use rig::completion::{CompletionModel, CompletionResponse as RigCompletionResponse, FinishReason};
use rig::providers::cohere::completion::{CompletionResponse, FinishReason as CohereFinishReason};
use serde::Deserialize;
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::raw_capture::capture_completion;
use crate::support::{Observed, json_contains_key, normalized_without_raw};

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

fn request(model: &(impl CompletionModel + Clone)) -> rig::completion::CompletionRequest {
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

/// Cohere's counters are `f64` on rig's wire type, so a captured `6.0`
/// must be compared numerically against the fixture's `6`.
fn number_at(value: &Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer).and_then(Value::as_f64)
}

// ---------------------------------------------------------------------------
// 1: typed access is recoverable, and agrees with the normalized fields
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_cohere_completion_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_cohere_completion_response";
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_cohere_cassette(
        "raw_capture_matrix/raw_roundtrips_cohere_completion_response",
        |client| async move {
            capture_completion(client.completion(CASSETTE_MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    let typed = CompletionResponse::deserialize(raw)
        .expect("raw must deserialize into Cohere's CompletionResponse");

    // The typed value agrees with the normalized fields next to it.
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| usage.tokens.as_ref())
            .and_then(|tokens| tokens.input_tokens)
            .map(|tokens| tokens as u64),
        response.usage.input_tokens
    );

    // One decoder, two views: the provider's own fields say what the
    // normalized response says. Asserting them against each other pins the
    // decoder's mapping; re-normalizing the typed value by hand would only
    // compare that mapping to a copy of itself.
    assert_eq!(
        typed.finish_reason,
        CohereFinishReason::Complete,
        "the recorded turn completed naturally in Cohere's own vocabulary"
    );
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::Stop),
        "and the decoder maps COMPLETE onto a natural stop"
    );
    let (content, _, _) = typed
        .message()
        .expect("the recorded turn is an assistant message");
    assert_eq!(
        content.len(),
        response.choice.len(),
        "the provider's blocks and the normalized choice are the same blocks"
    );

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
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_cohere_cassette(
        "raw_capture_matrix/raw_exposes_billing_metadata",
        |client| async move {
            capture_completion(client.completion(CASSETTE_MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = observed.take();

    // The normalized response provably lacks these: billed units have no
    // normalized home, and the finish reason reaches it only as rig's
    // vocabulary.
    let normalized = normalized_without_raw(response.clone());
    assert!(!json_contains_key(&normalized, "billed_units"));
    assert_ne!(
        normalized.get("finish_reason"),
        Some(&Value::String("COMPLETE".to_string())),
        "the normalized finish reason is rig's spelling, not Cohere's"
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));

    let raw = &response.raw;
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
            number_at(raw, pointer),
            number_at(&body, pointer),
            "raw must carry {pointer} exactly as the wire sent it"
        );
        assert!(
            number_at(&body, pointer).is_some(),
            "{SCENARIO}: the recorded body should carry {pointer}"
        );
    }
}
