//! Feature matrix for raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: `CompletionModel::completion` serializes the value
//! its inherent `raw_completion` returned — here Gemini's own
//! [`GenerateContentResponse`] — onto [`rig::completion::CompletionResponse::raw`]
//! before `try_into` normalizes it. There is no opt-in and nothing about it
//! reaches the wire; `raw` is `None` only on a response constructed without a
//! provider payload behind it (hand-built, or persisted before the field
//! existed), never because capture "was not requested".
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
//! | 1 | `raw_roundtrips_generate_content_response` | typed access | `GenerateContentResponse::deserialize(&*raw)` re-serializes equal, and its `try_into` reproduces the normalized response | recorded |
//! | 2 | `raw_exposes_prompt_tokens_details` | un-normalized field | `usageMetadata.promptTokensDetails` == fixture, absent from the normalized response | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cell 1 also carries the "one story" contract: re-normalizing `raw` by hand
//! lands on the same choice / finish reason / model / usage / identity the
//! typed route reported, so `raw` and the normalized response can never
//! disagree about the turn they describe.
//!
//! The un-normalized field of choice is `usageMetadata.promptTokensDetails`
//! (a per-modality token breakdown): `modelVersion` and `responseId` are
//! normalized into `model` / `response_id`, and `responseId` is scrubbed on
//! the way into the fixture, so neither would prove the raw value survives
//! against the recorded bytes.

use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::with_gemini_cassette;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

fn request(model: &gemini::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The premise every cell rests on: the recorded body is a `generateContent`
/// answer whose first candidate stopped naturally and whose usage carries the
/// per-modality prompt breakdown cell 2 reads.
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

/// Serialize the normalized response with `raw` removed, so a cell can prove a
/// field is *not* reachable through the normalized surface (as opposed to
/// merely re-reading it out of `raw`).
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
async fn raw_roundtrips_generate_content_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_generate_content_response";
    with_gemini_cassette(
        "raw_capture_matrix/raw_roundtrips_generate_content_response",
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

            // `raw` is the value `raw_completion` returned, serialized: Gemini's
            // own type reads it back, and re-serializing that typed value
            // reproduces `raw` exactly — nothing is lost through the escape hatch.
            let typed = GenerateContentResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's GenerateContentResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "GenerateContentResponse must round-trip through its own Serialize/Deserialize"
            );

            // The typed value agrees with the normalized fields next to it.
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

    assert_recorded_generate_content_body(SCENARIO);
}

// ---------------------------------------------------------------------------
// 2: an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_prompt_tokens_details() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_prompt_tokens_details";
    // The observed `raw` is compared against the fixture bytes only after the
    // wrapper returns, so it is carried out of the test body.
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_prompt_tokens_details",
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
