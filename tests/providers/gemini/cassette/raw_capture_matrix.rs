//! Feature matrix for opt-in raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! [`rig::completion::CompletionRequest::capture_raw_response`] is local
//! policy: when set, `CompletionModel::completion` serializes the value its
//! inherent `raw_completion` would have returned — here Gemini's own
//! [`GenerateContentResponse`] — onto [`rig::completion::CompletionResponse::raw`]
//! before `try_into` normalizes it. Off (the default) it stays `None`, and the
//! flag never reaches the wire either way.
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the normalized response. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns (record mode writes the fixture on the way out): a cell
//! whose recording lost the wire shape it claims to cover must fail loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `flag_off_leaves_raw_unset` | default (`false`) | `raw.is_none()` | recorded |
//! | 2 | `flag_on_roundtrips_generate_content_response` | `true` → typed access | `GenerateContentResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_prompt_tokens_details` | `true` → un-normalized field | `usageMetadata.promptTokensDetails` == fixture, absent from the normalized response | recorded |
//! | 4 | `request_bytes_invariant_across_flag` | request boundary | recorded off/on request bodies byte-identical | recorded |
//! | 5 | `normalized_fields_invariant_across_flag` | normalized fields | choice / finish reason / model / prompt usage identical off vs on | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cells 4 and 5 record one scenario each with **two** interactions — the
//! flag-off request first, then the flag-on twin — because the invariant is
//! between the two; the harness replays interactions in order.
//!
//! The un-normalized field of choice is `usageMetadata.promptTokensDetails`
//! (a per-modality token breakdown): `modelVersion` and `responseId` are
//! normalized into `model` / `response_id`, and `responseId` is scrubbed on
//! the way into the fixture, so neither would prove the raw value survives
//! against the recorded bytes.

use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_cassette;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small and the two turns of
/// a two-interaction cell answer identically at temperature 0.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

fn request(model: &gemini::CompletionModel, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .capture_raw_response(capture)
        .build()
}

/// The premise every cell rests on: the recorded body is a `generateContent`
/// answer whose first candidate stopped naturally and whose usage carries the
/// per-modality prompt breakdown cell 3 reads.
fn assert_recorded_generate_content_body(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "{scenario}: the recorded turn should have stopped naturally"
    );
    assert!(
        body.pointer("/usageMetadata/promptTokensDetails")
            .and_then(Value::as_array)
            .is_some_and(|details| !details.is_empty()),
        "{scenario}: the recorded usageMetadata should carry promptTokensDetails, the \
         un-normalized field this matrix reads through `raw`"
    );
    body
}

/// The on-wire proof that the flag is local policy: nothing in the recorded
/// request body names it, under either spelling a serializer could produce.
fn assert_request_body_never_names_the_flag(scenario: &str, body: &str) {
    for spelling in ["capture_raw_response", "captureRawResponse"] {
        assert!(
            !body.contains(spelling),
            "{scenario}: the recorded request body must not carry {spelling:?}; the flag is \
             `#[serde(skip)]` local policy and must never reach Gemini"
        );
    }
}

/// Serialize the normalized response with `raw` removed, so a cell can prove a
/// field is *not* reachable through the normalized surface (as opposed to
/// merely re-reading it out of `raw`).
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
    const SCENARIO: &str = "raw_capture_matrix/flag_off_leaves_raw_unset";
    with_gemini_cassette(
        "raw_capture_matrix/flag_off_leaves_raw_unset",
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
            // The normalized surface is untouched by the flag being off.
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            assert_eq!(response.model.as_deref(), Some(MODEL));
            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "Gemini reports a responseId on every generateContent answer"
            );
        },
    )
    .await;

    assert_recorded_generate_content_body(SCENARIO);
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
async fn flag_on_roundtrips_generate_content_response() {
    const SCENARIO: &str = "raw_capture_matrix/flag_on_roundtrips_generate_content_response";
    with_gemini_cassette(
        "raw_capture_matrix/flag_on_roundtrips_generate_content_response",
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

            // `raw` is the value `raw_completion` would have returned, serialized:
            // Gemini's own type reads it back, and re-serializing that typed value
            // reproduces `raw` exactly — nothing is lost through the escape hatch.
            let typed = GenerateContentResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's GenerateContentResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "GenerateContentResponse must round-trip through its own Serialize/Deserialize"
            );

            // And the typed value agrees with the normalized fields next to it.
            assert_eq!(typed.model_version.as_deref(), response.model.as_deref());
            assert_eq!(
                Some(typed.response_id.as_str()),
                response.response_id.as_deref()
            );
            assert_eq!(
                typed
                    .usage_metadata
                    .as_ref()
                    .map(|usage| usage.prompt_token_count as u64),
                Some(response.usage.input_tokens)
            );
        },
    )
    .await;

    assert_recorded_generate_content_body(SCENARIO);
}

// ---------------------------------------------------------------------------
// 3: on → an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_prompt_tokens_details() {
    const SCENARIO: &str = "raw_capture_matrix/flag_on_exposes_prompt_tokens_details";
    // The observed `raw` is compared against the fixture bytes only after the
    // wrapper returns, so it is carried out of the test body.
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_capture_matrix/flag_on_exposes_prompt_tokens_details",
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

            // The normalized response provably lacks the field: it is only
            // reachable through `raw`.
            assert!(
                !contains_key(&normalized_without_raw(&response), "promptTokensDetails"),
                "promptTokensDetails is not part of rig's normalized response — it is exactly \
             the kind of provider detail `raw` exists to expose"
            );
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_generate_content_body(SCENARIO);
    assert_eq!(
        raw.pointer("/usageMetadata/promptTokensDetails"),
        body.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry Gemini's promptTokensDetails exactly as the wire sent it"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        body.pointer("/candidates/0/finishReason"),
        "raw must keep Gemini's own finishReason spelling"
    );
    assert_eq!(
        raw.pointer("/usageMetadata/totalTokenCount"),
        body.pointer("/usageMetadata/totalTokenCount"),
        "raw must carry the wire's total token count untouched"
    );
}

// ---------------------------------------------------------------------------
// 4: the request boundary never sees the flag
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_bytes_invariant_across_flag() {
    const SCENARIO: &str = "raw_capture_matrix/request_bytes_invariant_across_flag";
    with_gemini_cassette(
        "raw_capture_matrix/request_bytes_invariant_across_flag",
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
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_invariant_across_flag";
    with_gemini_cassette(
        "raw_capture_matrix/normalized_fields_invariant_across_flag",
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

            // Same normalized meaning either way — `raw` is additive.
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
            // Two live turns get two responseIds; both are populated, neither is
            // shaped by the flag.
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
            response.pointer("/candidates/0/finishReason"),
            Some(&Value::String("STOP".to_string())),
            "{SCENARIO}: both recorded turns should have stopped naturally"
        );
    }
}
