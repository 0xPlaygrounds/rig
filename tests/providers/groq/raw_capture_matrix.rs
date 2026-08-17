//! Raw provider response capture on Groq's blocking chat-completions path.
//!
//! **The feature.** `CompletionRequest::capture_raw_response` asks the model
//! to attach the value its inherent `raw_completion` would have returned onto
//! the normalized [`rig::completion::CompletionResponse::raw`]. Groq reuses
//! the shared [`openai::CompletionResponse`] wire type rather than declaring
//! its own, so the raw view is that type serialized — and it is what makes
//! Groq's timing accounting (`usage.queue_time`, `prompt_time`,
//! `completion_time`, `total_time`) reachable: the shared usage type models
//! them, the normalized `Usage` has no slot for any of them. Off by default,
//! never serialized into the request, never a substitute for a normalized
//! field.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_leaves_raw_none` | flag off (the default) | `raw == None` | recorded |
//! | 2 | `capture_on_round_trips_openai_type` | flag on | `raw` deserializes into `openai::CompletionResponse` and re-serializes equal | recorded |
//! | 3 | `capture_on_exposes_queue_time` | provider-only field | `raw.usage.queue_time` and `raw.system_fingerprint` equal the fixture body | recorded |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | both responses reproduce their own fixture bytes; the on-response equals its raw re-normalized | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 3 reads the queue time out of the recorded
//! body rather than trusting the number the typed view reports, and cells 4
//! and 5 record the flag-off and flag-on turns as two interactions of one
//! scenario so the comparison is between bytes that crossed the wire in the
//! same session. Groq's `x_groq` envelope is *not* on the shared wire type,
//! so it is absent from `raw` by design (the docs say so: fields the wire
//! type does not model are not there); the cells pin the fields the type does
//! model.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::{groq, openai};
use serde::Deserialize;
use serde_json::{Value, json};

use super::RAW_CAPTURE_MODEL;
use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_groq_cassette_result,
};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &groq::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
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
fn assert_reproduces_fixture(
    response: &CompletionResponse,
    body: &Value,
    request_id: Option<&str>,
    context: &str,
) {
    assert_eq!(response.provider, PROVIDER, "{context}: provider");
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{context}: response id"),
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
        response.usage.input_tokens,
        body["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt_tokens"),
        "{context}: input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens"),
        "{context}: output tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"]
            .as_u64()
            .expect("total_tokens"),
        "{context}: total tokens"
    );
    assert_eq!(
        text_of(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("recorded content"),
        "{context}: choice text"
    );
    // Groq contracts `x-request-id`; the recorded header is the premise.
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
    with_groq_cassette_result(
        "raw_capture_matrix/capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
// 2. On round-trips the shared OpenAI type
// ================================================================

#[tokio::test]
async fn capture_on_round_trips_openai_type() {
    const SCENARIO: &str = "raw_capture_matrix/capture_on_round_trips_openai_type";
    with_groq_cassette_result(
        "raw_capture_matrix/capture_on_round_trips_openai_type",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
            let response = model.completion(request(&model, true)).await?;
            let raw = response
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
            let typed = openai::CompletionResponse::deserialize(raw)
                .expect("raw is the shared OpenAI CompletionResponse Groq parses into");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed view serialized, nothing more"
            );
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("capture_on_round_trips_openai_type should replay from its cassette");

    let (request_body, response_body) = &recorded_json(SCENARIO)[0];
    assert!(request_body.get("capture_raw_response").is_none());
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 3. A field the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn capture_on_exposes_queue_time() {
    const SCENARIO: &str = "raw_capture_matrix/capture_on_exposes_queue_time";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_groq_cassette_result(
        "raw_capture_matrix/capture_on_exposes_queue_time",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
            let response = model.completion(request(&model, true)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("capture_on_exposes_queue_time should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = &recorded_json(SCENARIO)[0];
    let recorded_queue_time = body["usage"]["queue_time"]
        .as_f64()
        .expect("Groq reports usage.queue_time on every response");
    let recorded_prompt_time = body["usage"]["prompt_time"]
        .as_f64()
        .expect("Groq reports usage.prompt_time on every response");
    let recorded_fingerprint = body["system_fingerprint"]
        .as_str()
        .expect("Groq reports a system_fingerprint");

    let raw = response.raw.as_deref().expect("raw is captured");
    assert_eq!(raw["usage"]["queue_time"], json!(recorded_queue_time));
    assert_eq!(raw["usage"]["prompt_time"], json!(recorded_prompt_time));
    assert_matches_recorded_token(
        raw["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // And the normalized view has no slot for any of them.
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("queue_time").is_none()
            && normalized_usage.get("prompt_time").is_none(),
        "the normalized usage has no timing slots: {normalized_usage}"
    );
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("system_fingerprint").is_none());
}

// ================================================================
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_capture_matrix/request_invariant_off_vs_on";
    with_groq_cassette_result(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
    with_groq_cassette_result(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
    // re-normalizes to: capture adds a view, it never changes the mapping.
    let raw = on.raw.as_deref().expect("raw is captured");
    let renormalized = openai::CompletionResponse::deserialize(raw)
        .expect("raw is the shared OpenAI type")
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
