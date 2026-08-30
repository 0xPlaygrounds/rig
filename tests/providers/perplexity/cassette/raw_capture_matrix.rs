//! Raw provider response capture on Perplexity's blocking chat-completions
//! path.
//!
//! **The feature.** Every blocking completion attaches the value the model's
//! inherent `raw_completion` returned onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a
//! `Value::Null` only ever means a response built by hand with no provider
//! payload behind it. Perplexity reuses the shared
//! [`perplexity::CompletionResponse`] wire type, so the raw view is that type
//! serialized. That framing matters here more than for any other provider in
//! this family: Perplexity's wire also carries `citations` and
//! `search_results`, which the shared type does *not* model — and the
//! documented meaning of `raw` is "the response as rig's wire type parsed it",
//! so those are absent from `raw` by construction. The cells pin what the type
//! does model and the normalized response lacks (the `object` tag), and cell 2
//! pins the absence of the unmodeled fields against the fixture bytes so the
//! limitation stays documented rather than discovered.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_openai_type` | typed round trip | `raw` deserializes into `perplexity::CompletionResponse` and re-serializes equal | recorded |
//! | 2 | `raw_exposes_object_not_citations` | provider-only field | `raw.object` equals the fixture body; the fixture's `citations` are not in `raw` (unmodeled by the wire type) | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes and equals its own `raw` re-normalized | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the `object` tag and the
//! `citations` array out of the recorded body, and cell 3 checks the
//! normalized fields against the recorded body before comparing them with
//! the re-normalized `raw`, so a recording that stopped carrying a usage
//! block or a finish reason fails loudly instead of covering nothing.
//! Perplexity contracts no request-id header, so `provider_request_id` is
//! `None` on every turn here — a documented outcome, pinned as such.
//! Perplexity's models search the web on every turn, so the prompt is
//! deliberately trivial.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::perplexity;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{assert_matches_recorded_token, with_perplexity_cassette};

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &perplexity::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The single recorded interaction of `scenario` as `(request, response)` JSON.
fn recorded_json(scenario: &str) -> (Value, Value) {
    let interactions = crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario);
    assert_eq!(
        interactions.len(),
        1,
        "every cell here is a single completion turn"
    );
    let (request, response) = &interactions[0];
    (
        serde_json::from_str(request).expect("recorded request should be JSON"),
        serde_json::from_str(response).expect("recorded response should be JSON"),
    )
}

fn recorded_finish_reason(body: &Value) -> FinishReason {
    match body["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        other => panic!("recorded turn should finish on stop or length, got {other:?}"),
    }
}

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The normalized fields, checked against the wire bytes that produced them.
fn assert_reproduces_fixture(response: &CompletionResponse, body: &Value) {
    assert_eq!(response.provider, PROVIDER, "provider");
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        "response id",
    );
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.finish_reason(),
        Some(recorded_finish_reason(body)),
        "finish reason"
    );
    assert_eq!(
        response.usage.input_tokens,
        body["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt_tokens"),
        "input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens"),
        "output tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"]
            .as_u64()
            .expect("total_tokens"),
        "total tokens"
    );
    assert_eq!(
        text_of(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("recorded content"),
        "choice text"
    );
    // Perplexity contracts no request-id header, so `None` is the documented
    // outcome.
    assert_eq!(response.provider_request_id, None, "request id");
}

// ================================================================
// 1. raw round-trips the shared OpenAI type
// ================================================================

#[tokio::test]
async fn raw_round_trips_openai_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openai_type";
    with_perplexity_cassette(
        "raw_capture_matrix/raw_round_trips_openai_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("the turn should succeed");
            let raw = &response.raw;
            let typed = perplexity::CompletionResponse::deserialize(raw)
                .expect("raw is the shared OpenAI CompletionResponse Perplexity parses into");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed view serialized, nothing more"
            );
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
        },
    )
    .await;

    let (_, response_body) = recorded_json(SCENARIO);
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 2. What the wire type models is in raw; what it does not is not
// ================================================================

#[tokio::test]
async fn raw_exposes_object_not_citations() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_object_not_citations";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/raw_exposes_object_not_citations",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("the turn should succeed");
            *sink.lock().expect("observation lock") = Some(response);
        },
    )
    .await;

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = recorded_json(SCENARIO);
    let recorded_object = body["object"]
        .as_str()
        .expect("Perplexity tags every completion with an object");
    assert!(
        body["citations"].is_array(),
        "the recorded Perplexity turn carries citations: {body}"
    );

    let raw = &response.raw;
    assert_eq!(raw["object"], json!(recorded_object));
    // The normalized view has no slot for the tag.
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("object").is_none());
    // And `raw` is the wire *type* serialized, not the wire bytes: the shared
    // OpenAI type has no `citations` / `search_results` slot, so they are not
    // here — the documented limitation, pinned against a fixture that has
    // them.
    assert!(
        raw.get("citations").is_none() && raw.get("search_results").is_none(),
        "unmodeled fields are absent from raw by construction: {raw}"
    );
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("the turn should succeed");
            *sink.lock().expect("observation lock") = Some(response);
        },
    )
    .await;

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = recorded_json(SCENARIO);
    assert_reproduces_fixture(&response, &body);

    // The normalized fields are exactly what the response's own raw
    // re-normalizes to: capture adds a view, it never changes the mapping.
    let raw = &response.raw;
    let renormalized = perplexity::CompletionResponse::deserialize(raw)
        .expect("raw is the shared OpenAI type")
        .normalize(PROVIDER)
        .expect("raw normalizes")
        .with_optional_provider_request_id(response.provider_request_id.clone());
    assert_eq!(renormalized.identity(), response.identity());
    assert_eq!(renormalized.finish_reason(), response.finish_reason());
    assert_eq!(renormalized.model, response.model);
    assert_eq!(renormalized.usage, response.usage);
    assert_eq!(renormalized.choice, response.choice);
    assert!(
        renormalized.raw.is_null(),
        "normalizing a hand-fed typed value attaches no raw of its own"
    );
}
