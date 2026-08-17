//! Raw provider response capture on xAI's blocking path.
//!
//! **The feature.** `CompletionRequest::capture_raw_response` asks the model
//! to attach the value its inherent `raw_completion` would have returned onto
//! the normalized [`rig::completion::CompletionResponse::raw`]. xAI speaks
//! the OpenAI Responses wire, so the raw view is the Responses
//! [`CompletionResponse`] serialized — a serialization that mirrors the wire
//! body, which is why the transport `provider_request_id` the typed value
//! carries is *not* in `raw` (it is on the normalized response instead). Off
//! by default, never serialized into the request, never a substitute for a
//! normalized field. The Responses envelope carries a `status` and, on xAI,
//! a `service_tier` and a `metadata.system_fingerprint` the normalized
//! response has no slot for; those are the fields pinned here as reachable
//! only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_leaves_raw_none` | flag off (the default) | `raw == None` | recorded |
//! | 2 | `capture_on_round_trips_responses_type` | flag on | `raw` deserializes into the Responses `CompletionResponse` and re-serializes equal | recorded |
//! | 3 | `capture_on_exposes_status_and_service_tier` | provider-only field | `raw.status`, `raw.service_tier` and `raw.metadata.system_fingerprint` equal the fixture body | recorded |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | both responses reproduce their own fixture bytes (including the `x-request-id` header); the on-response equals its raw re-normalized plus the id | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 3 reads the status and tier out of the
//! recorded body rather than trusting what the typed view reports, and cells
//! 4 and 5 record the flag-off and flag-on turns as two interactions of one
//! scenario so the comparison is between bytes that crossed the wire in the
//! same session.

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

fn request(model: &xai::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .capture_raw_response(capture)
        .build()
}

/// Every recorded interaction of `scenario` as `(request, response)` JSON.
fn recorded_json(scenario: &str) -> Vec<(Value, Value)> {
    crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario)
        .into_iter()
        .map(|(request, response)| {
            (
                serde_json::from_str(&request).expect("recorded request should be JSON"),
                serde_json::from_str(&response).expect("recorded response should be JSON"),
            )
        })
        .collect()
}

/// The `x-request-id` the recorded interaction at `index` carried.
fn recorded_request_id(scenario: &str, index: usize) -> Option<String> {
    recorded_response_headers(scenario)[index]
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
    context: &str,
) {
    assert_eq!(response.provider, PROVIDER, "{context}: provider");
    let (message_id, text) = recorded_message(body);
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_matches_recorded_token(
        response.message_id.as_deref(),
        Some(message_id),
        &format!("{context}: message id"),
    );
    assert_eq!(
        response.model.as_deref(),
        body["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        response.finish_reason(),
        Some(recorded_finish_reason(body)),
        "{context}: finish reason"
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
        "{context}: usage"
    );
    assert_eq!(text_of(&response.choice), text, "{context}: choice text");
    // xAI contracts `x-request-id`; the recorded header is the premise.
    assert!(
        request_id.is_some(),
        "{context}: the recorded response must carry x-request-id"
    );
    assert_matches_recorded_token(
        response.provider_request_id.as_deref(),
        request_id,
        &format!("{context}: request id"),
    );
}

// ================================================================
// 1. Off leaves raw None
// ================================================================

#[tokio::test]
async fn capture_off_leaves_raw_none() {
    const SCENARIO: &str = "raw_capture_matrix/capture_off_leaves_raw_none";
    with_xai_cassette_result(
        "raw_capture_matrix/capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model, false)).await?;
            assert!(
                response.raw.is_none(),
                "raw must stay None unless asked for"
            );
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("capture_off_leaves_raw_none should replay from its cassette");

    let (request_body, _) = &recorded_json(SCENARIO)[0];
    assert!(
        request_body.get("capture_raw_response").is_none(),
        "the flag is local policy and must never reach the wire"
    );
}

// ================================================================
// 2. On round-trips the Responses type
// ================================================================

#[tokio::test]
async fn capture_on_round_trips_responses_type() {
    const SCENARIO: &str = "raw_capture_matrix/capture_on_round_trips_responses_type";
    with_xai_cassette_result(
        "raw_capture_matrix/capture_on_round_trips_responses_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model, true)).await?;
            let raw = response
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
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
    .expect("capture_on_round_trips_responses_type should replay from its cassette");

    let (request_body, response_body) = &recorded_json(SCENARIO)[0];
    assert!(request_body.get("capture_raw_response").is_none());
    assert_eq!(response_body["object"], json!("response"));
}

// ================================================================
// 3. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn capture_on_exposes_status_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/capture_on_exposes_status_and_service_tier";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_capture_matrix/capture_on_exposes_status_and_service_tier",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model.completion(request(&model, true)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("capture_on_exposes_status_and_service_tier should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = &recorded_json(SCENARIO)[0];
    let recorded_status = body["status"]
        .as_str()
        .expect("a Responses body carries a status");
    let recorded_tier = body["service_tier"]
        .as_str()
        .expect("xAI reports service_tier on its Responses body");
    let recorded_fingerprint = body["metadata"]["system_fingerprint"]
        .as_str()
        .expect("xAI reports metadata.system_fingerprint");

    let raw = response.raw.as_deref().expect("raw is captured");
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
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_capture_matrix/request_invariant_off_vs_on";
    with_xai_cassette_result(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model.completion(request(&model, false)).await?;
            let on = model.completion(request(&model, true)).await?;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("request_invariant_off_vs_on should replay from its cassette");

    let interactions = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "one flag-off and one flag-on turn");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "the flag-off and flag-on request bodies must be byte-identical"
    );
    assert!(!interactions[0].0.contains("capture_raw"));
}

// ================================================================
// 5. Normalized fields are the same either way
// ================================================================

#[tokio::test]
async fn normalized_fields_identical_off_vs_on() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model.completion(request(&model, false)).await?;
            let on = model.completion(request(&model, true)).await?;
            *sink.lock().expect("observation lock") = Some((off, on));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("normalized_fields_identical_off_vs_on should replay from its cassette");

    let (off, on) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe both responses");
    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2);
    assert!(off.raw.is_none());
    assert_reproduces_fixture(
        &off,
        &interactions[0].1,
        recorded_request_id(SCENARIO, 0).as_deref(),
        "flag off",
    );
    assert_reproduces_fixture(
        &on,
        &interactions[1].1,
        recorded_request_id(SCENARIO, 1).as_deref(),
        "flag on",
    );

    // The on-response's normalized fields are exactly what its own raw
    // re-normalizes to (plus the transport id the mirrored body cannot
    // carry): capture adds a view, it never changes the mapping.
    let raw = on.raw.as_deref().expect("raw is captured");
    let renormalized = xai::CompletionResponse::deserialize(raw)
        .expect("raw is the Responses type")
        .normalize(PROVIDER)
        .expect("raw normalizes")
        .with_optional_provider_request_id(on.provider_request_id.clone());
    assert_eq!(renormalized.identity(), on.identity());
    assert_eq!(renormalized.finish_reason(), on.finish_reason());
    assert_eq!(renormalized.model, on.model);
    assert_eq!(renormalized.usage, on.usage);
    assert_eq!(renormalized.choice, on.choice);
    assert!(
        renormalized.raw.is_none(),
        "normalizing raw does not re-capture"
    );
}
