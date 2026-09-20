//! Feature matrix for raw provider response capture on the Gemini
//! Interactions API (`POST /v1beta/interactions`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: the driver puts the provider's reply body,
//! parsed as JSON, onto [`rig::completion::CompletionResponse::raw`] — here
//! the API's own [`Interaction`] document, verbatim. There is no opt-in and
//! nothing about it reaches the wire; `raw` is required at construction, so
//! there is no response without the document that produced it and no way
//! for capture to be "not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the normalized response. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns: the recorded interaction must have completed and carry the
//! lifecycle fields the cells read.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_roundtrips_interaction` | typed access | `Interaction::deserialize(&raw)` reads the document back, and its provider-native fields agree with the normalized response | recorded |
//! | 2 | `raw_exposes_lifecycle_fields` | un-normalized fields | `object` / `status` spelling / `steps` == fixture, absent from the normalized response | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain non-streaming interactions route.
//!
//! Cell 1 also carries the "one story" contract: the normalized response was
//! folded by one decoder out of these very bytes, so the identity, finish
//! reason, model and usage read off `raw` are the ones the response reports —
//! `raw` and the normalized response can never disagree about the turn they
//! describe.
//!
//! The un-normalized fields of choice are the interaction's lifecycle
//! envelope: `object` (`"interaction"`), the wire spelling of `status`
//! (`"completed"`, which normalizes to rig's `Stop`), and the `steps` log —
//! the interaction `id` is normalized into `response_id` *and* scrubbed into
//! the fixture, so it cannot prove anything against the recorded bytes.

use rig::completion::{CompletionModel, FinishReason};
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::providers::gemini::interactions_api::{Interaction, InteractionStatus, Interactions};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_interactions_cassette;
use crate::support::{json_contains_key, normalized_without_raw};

const PROVIDER: &str = "gemini";

/// The model the neighbouring interactions cassettes record against; the
/// Interactions API is served for the Gemini 3 family.
const MODEL: &str = "gemini-3-flash-preview";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

/// The Interactions wire bound to the bundled cassette transport. One Gemini
/// config serves both surfaces, so the wrapper hands out the config and each
/// cell names the surface it is about.
type Model = Bound<Interactions, BoxedHttpClient>;

fn request(model: &Model) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The premise every cell rests on: the recorded body is a completed
/// interaction carrying the lifecycle envelope cell 2 reads.
fn assert_recorded_completed_interaction(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.get("object"),
        Some(&Value::String("interaction".to_string())),
        "{scenario}: the recorded body should be an interaction object"
    );
    assert_eq!(
        body.get("status"),
        Some(&Value::String("completed".to_string())),
        "{scenario}: the recorded interaction should have completed"
    );
    assert!(
        body.get("steps")
            .and_then(Value::as_array)
            .is_some_and(|steps| !steps.is_empty()),
        "{scenario}: the recorded interaction should carry its steps log"
    );
    body
}

// ---------------------------------------------------------------------------
// 1: typed access is recoverable, and tells the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_interaction() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/raw_roundtrips_interaction";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/raw_roundtrips_interaction",
        |client| async move {
            let model = client.map_wire(|config| config.interactions(MODEL));
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;

            let typed = Interaction::deserialize(raw)
                .expect("raw must deserialize into the Interactions API's Interaction");

            // One decoder folded the normalized response out of these very
            // bytes, so every field it kept must be the one the document
            // carries — `raw` is additive, never a divergent second view.
            assert_eq!(typed.model, response.model);
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            assert_eq!(
                typed
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.total_input_tokens),
                response.usage.input_tokens
            );
            assert_eq!(
                typed.usage.as_ref().and_then(|usage| usage.total_tokens),
                response.usage.total_tokens
            );
            assert!(
                matches!(typed.status, Some(InteractionStatus::Completed)),
                "the document keeps the API's own lifecycle spelling, got {:?}",
                typed.status
            );
            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Stop),
                "and `completed` reaches the caller as rig's Stop"
            );
        },
    )
    .await;

    assert_recorded_completed_interaction(SCENARIO);
}

// ---------------------------------------------------------------------------
// 2: un-normalized lifecycle fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_lifecycle_fields() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/raw_exposes_lifecycle_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/raw_exposes_lifecycle_fields",
        |client| async move {
            let model = client.map_wire(|config| config.interactions(MODEL));
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized response provably lacks these: `object` and `steps`
            // have no normalized home, and `status` reaches it only as rig's
            // finish-reason vocabulary.
            let normalized = normalized_without_raw(response.clone());
            assert!(!json_contains_key(&normalized, "object"));
            assert!(!json_contains_key(&normalized, "steps"));
            assert!(!json_contains_key(&normalized, "status"));
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_completed_interaction(SCENARIO);
    assert_eq!(raw.get("object"), body.get("object"));
    assert_eq!(
        raw.get("status"),
        body.get("status"),
        "raw keeps the API's own status spelling"
    );
    assert_eq!(
        raw.get("steps").and_then(Value::as_array).map(Vec::len),
        body.get("steps").and_then(Value::as_array).map(Vec::len),
        "raw carries the interaction's steps log, one entry per recorded step"
    );
    assert_eq!(
        raw.pointer("/usage/total_tokens"),
        body.pointer("/usage/total_tokens"),
        "raw carries the wire's total token count untouched"
    );
}
