//! Raw provider response capture on xAI's blocking path.
//!
//! **The feature.** Every blocking completion attaches the value the model's
//! inherent `raw_completion` returned onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. xAI speaks the OpenAI
//! Responses wire, so the raw view is the Responses [`CompletionResponse`]
//! serialized — a serialization that mirrors the wire body, which is why the
//! transport `provider_request_id` the typed value carries is *not* in `raw`
//! (it is on the normalized response instead). Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a `None`
//! only ever means a response built by hand with no provider payload behind
//! it. `raw` is a second view of the same response, never a substitute for a
//! normalized field. The Responses envelope carries a `status` and, on xAI,
//! a `service_tier` and a `metadata.system_fingerprint` the normalized
//! response has no slot for; those are the fields pinned here as reachable
//! only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_responses_type` | typed round trip | `raw` deserializes into the Responses `CompletionResponse` and re-serializes equal | recorded |
//! | 2 | `raw_exposes_status_and_service_tier` | provider-only field | `raw.status`, `raw.service_tier` and `raw.metadata.system_fingerprint` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes (including the `x-request-id` header) and equals its own `raw` re-normalized plus the id | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the status and tier out of the
//! recorded body rather than trusting what the typed view reports, and cell 3
//! checks the normalized fields against the recorded body and the recorded
//! `x-request-id` header before comparing them with the re-normalized `raw`,
//! so a recording that stopped carrying usage, a status, or the id header
//! fails loudly instead of covering nothing.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::xai;
use serde::Deserialize;
use serde_json::{Value, json};

use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_xai_cassette_result,
};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &xai::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).build()
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

/// The `x-request-id` the single recorded interaction carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_headers(scenario)[0]
        .iter()
        .find(|(name, _)| name == "x-request-id")
        .map(|(_, value)| value.clone())
}

/// The recorded `output` message item: its id and its output text.
fn recorded_message(body: &Value) -> (&str, String) {
    let item = body["output"]
        .as_array()
        .expect("output items")
        .iter()
        .find(|item| item["type"] == "message")
        .expect("the recorded turn carries a message item");
    let text = item["content"]
        .as_array()
        .expect("message content")
        .iter()
        .filter(|part| part["type"] == "output_text")
        .map(|part| part["text"].as_str().expect("output_text"))
        .collect();
    (item["id"].as_str().expect("message id"), text)
}

fn recorded_finish_reason(body: &Value) -> FinishReason {
    match body["status"].as_str() {
        Some("completed") => FinishReason::Stop,
        other => panic!("recorded turn should have completed, got {other:?}"),
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
fn assert_reproduces_fixture(
    response: &CompletionResponse,
    body: &Value,
    request_id: Option<&str>,
) {
    assert_eq!(response.provider, PROVIDER, "provider");
    let (message_id, text) = recorded_message(body);
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        "response id",
    );
    assert_matches_recorded_token(
        response.message_id.as_deref(),
        Some(message_id),
        "message id",
    );
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.finish_reason(),
        Some(recorded_finish_reason(body)),
        "finish reason"
    );
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            body["usage"]["input_tokens"].as_u64().expect("input"),
            body["usage"]["output_tokens"].as_u64().expect("output"),
            body["usage"]["total_tokens"].as_u64().expect("total"),
        ),
        "usage"
    );
    assert_eq!(text_of(&response.choice), text, "choice text");
    // xAI contracts `x-request-id`; the recorded header is the premise.
    assert!(
        request_id.is_some(),
        "the recorded response must carry x-request-id"
    );
    assert_matches_recorded_token(
        response.provider_request_id.as_deref(),
        request_id,
        "request id",
    );
}

// ================================================================
// 1. raw round-trips the Responses type
// ================================================================

#[tokio::test]
async fn raw_round_trips_responses_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_responses_type";
    with_xai_cassette_result(
        "raw_capture_matrix/raw_round_trips_responses_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model)).await?;
            let raw = response
                .raw
                .as_deref()
                .expect("every provider-backed response carries raw");
            let typed = xai::CompletionResponse::deserialize(raw)
                .expect("raw is the Responses CompletionResponse xAI parses into");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed view serialized, nothing more"
            );
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            // The transport id is not part of the wire body, so the mirrored
            // serialization never carries it — it lives on the normalized
            // response only.
            assert!(raw.get("provider_request_id").is_none());
            assert_eq!(typed.provider_request_id, None);
            assert!(response.provider_request_id.is_some());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_round_trips_responses_type should replay from its cassette");

    let (_, response_body) = recorded_json(SCENARIO);
    assert_eq!(response_body["object"], json!("response"));
}

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_status_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_status_and_service_tier";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_capture_matrix/raw_exposes_status_and_service_tier",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_exposes_status_and_service_tier should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = recorded_json(SCENARIO);
    let recorded_status = body["status"]
        .as_str()
        .expect("a Responses body carries a status");
    let recorded_tier = body["service_tier"]
        .as_str()
        .expect("xAI reports service_tier on its Responses body");
    let recorded_fingerprint = body["metadata"]["system_fingerprint"]
        .as_str()
        .expect("xAI reports metadata.system_fingerprint");

    let raw = response
        .raw
        .as_deref()
        .expect("every provider-backed response carries raw");
    assert_eq!(raw["status"], json!(recorded_status));
    assert_eq!(raw["service_tier"], json!(recorded_tier));
    assert_matches_recorded_token(
        raw["metadata"]["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // And the normalized view has no slot for any of them: the status is
    // folded into a finish reason, the rest has nowhere to go.
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("status").is_none());
    assert!(normalized.get("service_tier").is_none());
    assert!(normalized.get("metadata").is_none());
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = recorded_json(SCENARIO);
    assert_reproduces_fixture(&response, &body, recorded_request_id(SCENARIO).as_deref());

    // The normalized fields are exactly what the response's own raw
    // re-normalizes to (plus the transport id the mirrored body cannot
    // carry): capture adds a view, it never changes the mapping.
    let raw = response
        .raw
        .as_deref()
        .expect("every provider-backed response carries raw");
    let renormalized = xai::CompletionResponse::deserialize(raw)
        .expect("raw is the Responses type")
        .normalize(PROVIDER)
        .expect("raw normalizes")
        .with_optional_provider_request_id(response.provider_request_id.clone());
    assert_eq!(renormalized.identity(), response.identity());
    assert_eq!(renormalized.finish_reason(), response.finish_reason());
    assert_eq!(renormalized.model, response.model);
    assert_eq!(renormalized.usage, response.usage);
    assert_eq!(renormalized.choice, response.choice);
    assert!(
        renormalized.raw.is_none(),
        "normalizing a hand-fed typed value attaches no raw of its own"
    );
}
