//! Feature matrix for opt-in raw provider response capture on the Gemini
//! Interactions API (`POST /v1beta/interactions`) unary seam.
//!
//! # The feature
//!
//! [`rig::completion::CompletionRequest::capture_raw_response`] is local
//! policy: when set, `InteractionsCompletionModel::completion` serializes the
//! value its inherent `raw_completion` would have returned — the API's own
//! [`Interaction`] payload — onto [`rig::completion::CompletionResponse::raw`]
//! before `try_into` normalizes it. Off (the default) it stays `None`, and the
//! flag never reaches the wire either way.
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
//! | 1 | `flag_off_leaves_raw_unset` | default (`false`) | `raw.is_none()` | recorded |
//! | 2 | `flag_on_roundtrips_interaction` | `true` → typed access | `Interaction::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_lifecycle_fields` | `true` → un-normalized fields | `object` / `status` spelling / `steps` == fixture, absent from the normalized response | recorded |
//! | 4 | `request_bytes_invariant_across_flag` | request boundary | recorded off/on request bodies byte-identical | recorded |
//! | 5 | `normalized_fields_invariant_across_flag` | normalized fields | choice / finish reason / model / prompt usage identical off vs on | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain non-streaming interactions route.
//!
//! Cells 4 and 5 record one scenario each with **two** interactions — the
//! flag-off request first, then the flag-on twin — because the invariant is
//! between the two; the harness replays interactions in order.
//!
//! The un-normalized fields of choice are the interaction's lifecycle
//! envelope: `object` (`"interaction"`), the wire spelling of `status`
//! (`"completed"`, which normalizes to rig's `Stop`), and the `steps` log —
//! the interaction `id` is normalized into `response_id` *and* scrubbed into
//! the fixture, so it cannot prove anything against the recorded bytes.

use rig::completion::{CompletionModel as _, FinishReason};
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

fn request(model: &Model, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .capture_raw_response(capture)
        .build()
}

/// The premise every cell rests on: the recorded body is a completed
/// interaction carrying the lifecycle envelope cell 3 reads.
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

fn assert_request_body_never_names_the_flag(scenario: &str, body: &str) {
    for spelling in ["capture_raw_response", "captureRawResponse"] {
        assert!(
            !body.contains(spelling),
            "{scenario}: the recorded request body must not carry {spelling:?}; the flag is \
             `#[serde(skip)]` local policy and must never reach Gemini"
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

// ---------------------------------------------------------------------------
// 1: default off
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_off_leaves_raw_unset() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/flag_off_leaves_raw_unset";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/flag_off_leaves_raw_unset",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, false))
                .await
                .expect("completion should succeed");

            assert!(
                response.raw.is_none(),
                "capture was not requested, so the normalized response must not carry raw"
            );
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            assert_eq!(response.model.as_deref(), Some(MODEL));
            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "the interaction id is normalized as the response id"
            );
        },
    )
    .await;

    assert_recorded_completed_interaction(SCENARIO);
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
async fn flag_on_roundtrips_interaction() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/flag_on_roundtrips_interaction";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/flag_on_roundtrips_interaction",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("capture was requested, so raw must be populated");

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
        },
    )
    .await;

    assert_recorded_completed_interaction(SCENARIO);
}

// ---------------------------------------------------------------------------
// 3: on → un-normalized lifecycle fields are readable and match the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_lifecycle_fields() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/flag_on_exposes_lifecycle_fields";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/flag_on_exposes_lifecycle_fields",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("capture was requested, so raw must be populated");
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

// ---------------------------------------------------------------------------
// 4: the request boundary never sees the flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_bytes_invariant_across_flag() {
    const SCENARIO: &str = "interactions_raw_capture_matrix/request_bytes_invariant_across_flag";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/request_bytes_invariant_across_flag",
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
         policy and never reaches Gemini"
    );
    assert_request_body_never_names_the_flag(SCENARIO, on_request);
}

// ---------------------------------------------------------------------------
// 5: normalized fields mean the same thing with or without capture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn normalized_fields_invariant_across_flag() {
    const SCENARIO: &str =
        "interactions_raw_capture_matrix/normalized_fields_invariant_across_flag";
    with_gemini_interactions_cassette(
        "interactions_raw_capture_matrix/normalized_fields_invariant_across_flag",
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

            assert_eq!(off.choice, on.choice);
            assert_eq!(off.finish_reason(), on.finish_reason());
            assert_eq!(off.finish_reason(), Some(FinishReason::Stop));
            assert_eq!(off.model, on.model);
            assert_eq!(off.provider, on.provider);
            assert_eq!(off.message_id, on.message_id);
            assert_eq!(off.provider_request_id, on.provider_request_id);
            // Identical request bytes tokenize identically; the output side (and
            // the thinking budget spent on it) is the model's to vary.
            assert_eq!(off.usage.input_tokens, on.usage.input_tokens);
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
            response.get("status"),
            Some(&Value::String("completed".to_string())),
            "{SCENARIO}: both recorded interactions should have completed"
        );
    }
}
