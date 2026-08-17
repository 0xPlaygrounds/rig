//! Feature matrix for raw provider response capture on the Gemini
//! Interactions API (`POST /v1beta/interactions`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: `InteractionsCompletionModel::completion`
//! serializes the value its inherent `raw_completion` returned — the API's
//! own [`Interaction`] payload — onto
//! [`rig::completion::CompletionResponse::raw`] before `try_into` normalizes
//! it. There is no opt-in and nothing about it reaches the wire; `raw` is
//! `None` only on a response constructed without a provider payload behind it
//! (hand-built, or persisted before the field existed), never because capture
//! "was not requested".
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
//! | 1 | `raw_roundtrips_interaction` | typed access | `Interaction::deserialize(&*raw)` re-serializes equal, and its `try_into` reproduces the normalized response | recorded |
//! | 2 | `raw_exposes_lifecycle_fields` | un-normalized fields | `object` / `status` spelling / `steps` == fixture, absent from the normalized response | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain non-streaming interactions route.
//!
//! Cell 1 also carries the "one story" contract: re-normalizing `raw` by hand
//! lands on the same choice / finish reason / model / usage / identity the
//! typed route reported, so `raw` and the normalized response can never
//! disagree about the turn they describe.
//!
//! The un-normalized fields of choice are the interaction's lifecycle
//! envelope: `object` (`"interaction"`), the wire spelling of `status`
//! (`"completed"`, which normalizes to rig's `Stop`), and the `steps` log —
//! the interaction `id` is normalized into `response_id` *and* scrubbed into
//! the fixture, so it cannot prove anything against the recorded bytes.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::prelude::*;
use rig::providers::gemini::interactions_api::{Interaction, InteractionsCompletionModel};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_interactions_cassette;

const PROVIDER: &str = "gemini";

/// The model the neighbouring interactions cassettes record against; the
/// Interactions API is served for the Gemini 3 family.
const MODEL: &str = "gemini-3-flash-preview";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

type Model = InteractionsCompletionModel<reqwest::Client>;

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

// ---------------------------------------------------------------------------
// 1: typed access is recoverable, and re-normalizes to the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_interaction() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/raw_roundtrips_interaction";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/raw_roundtrips_interaction",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("a provider-backed response always carries raw");

            let typed = Interaction::deserialize(raw)
                .expect("raw must deserialize into the Interactions API's Interaction");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "Interaction must round-trip through its own Serialize/Deserialize"
            );

            // The typed value agrees with the normalized fields next to it.
            assert_eq!(typed.model, response.model);
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            assert_eq!(
                typed
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.total_input_tokens),
                Some(response.usage.input_tokens)
            );

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
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("a provider-backed response always carries raw");
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized response provably lacks these: `object` and `steps`
            // have no normalized home, and `status` reaches it only as rig's
            // finish-reason vocabulary.
            let normalized = normalized_without_raw(&response);
            assert!(!contains_key(&normalized, "object"));
            assert!(!contains_key(&normalized, "steps"));
            assert!(!contains_key(&normalized, "status"));
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
