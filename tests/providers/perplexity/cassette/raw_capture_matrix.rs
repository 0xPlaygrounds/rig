//! Raw provider response capture on Perplexity's blocking chat-completions
//! path.
//!
//! **The feature.** `CompletionRequest::capture_raw_response` asks the model
//! to attach the value its inherent `raw_completion` would have returned onto
//! the normalized [`rig::completion::CompletionResponse::raw`]. Perplexity
//! reuses the shared [`openai::CompletionResponse`] wire type, so the raw
//! view is that type serialized. That framing matters here more than for any
//! other provider in this family: Perplexity's wire also carries `citations`
//! and `search_results`, which the shared type does *not* model — and the
//! documented meaning of `raw` is "the response as rig's wire type parsed
//! it", so those are absent from `raw` by construction. The cells pin what
//! the type does model and the normalized response lacks (the `object` tag),
//! and cell 3 pins the absence of the unmodeled fields against the fixture
//! bytes so the limitation stays documented rather than discovered.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `capture_off_leaves_raw_none` | flag off (the default) | `raw == None` | recorded |
//! | 2 | `capture_on_round_trips_openai_type` | flag on | `raw` deserializes into `openai::CompletionResponse` and re-serializes equal | recorded |
//! | 3 | `capture_on_exposes_object_not_citations` | provider-only field | `raw.object` equals the fixture body; the fixture's `citations` are not in `raw` (unmodeled by the wire type) | recorded |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | both responses reproduce their own fixture bytes; the on-response equals its raw re-normalized | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns, and cells 4 and 5 record the flag-off and
//! flag-on turns as two interactions of one scenario so the comparison is
//! between bytes that crossed the wire in the same session. Perplexity
//! contracts no request-id header, so `provider_request_id` is `None` on
//! every turn here — a documented outcome, pinned as such. Perplexity's
//! models search the web on every turn, so the prompt is deliberately
//! trivial.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::{openai, perplexity};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{assert_matches_recorded_token, with_perplexity_cassette};

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &perplexity::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .capture_raw_response(capture)
        .build()
}

/// Perplexity rate-limits back-to-back requests on this key (`429
/// request_rate_limit_exceeded` on the second turn of a two-turn scenario).
/// The pause exists only while recording — replay serves the fixture — and
/// leaves no trace in the fixture.
async fn pause_between_live_turns() {
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Record {
        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
    }
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
fn assert_reproduces_fixture(response: &CompletionResponse, body: &Value, context: &str) {
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
    // Perplexity contracts no request-id header, so `None` is the documented
    // outcome.
    assert_eq!(response.provider_request_id, None, "{context}: request id");
}

// ================================================================
// 1. Off leaves raw None
// ================================================================

#[tokio::test]
async fn capture_off_leaves_raw_none() {
    const SCENARIO: &str = "raw_capture_matrix/capture_off_leaves_raw_none";
    with_perplexity_cassette(
        "raw_capture_matrix/capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, false))
                .await
                .expect("the turn should succeed");
            assert!(
                response.raw.is_none(),
                "raw must stay None unless asked for"
            );
            assert!(!response.choice.is_empty());
        },
    )
    .await;

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
    with_perplexity_cassette(
        "raw_capture_matrix/capture_on_round_trips_openai_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("the turn should succeed");
            let raw = response
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
            let typed = openai::CompletionResponse::deserialize(raw)
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

    let (request_body, response_body) = &recorded_json(SCENARIO)[0];
    assert!(request_body.get("capture_raw_response").is_none());
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 3. What the wire type models is in raw; what it does not is not
// ================================================================

#[tokio::test]
async fn capture_on_exposes_object_not_citations() {
    const SCENARIO: &str = "raw_capture_matrix/capture_on_exposes_object_not_citations";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/capture_on_exposes_object_not_citations",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
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
    let (_, body) = &recorded_json(SCENARIO)[0];
    let recorded_object = body["object"]
        .as_str()
        .expect("Perplexity tags every completion with an object");
    assert!(
        body["citations"].is_array(),
        "the recorded Perplexity turn carries citations: {body}"
    );

    let raw = response.raw.as_deref().expect("raw is captured");
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
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_capture_matrix/request_invariant_off_vs_on";
    with_perplexity_cassette(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("the turn should succeed");
            pause_between_live_turns().await;
            let on = model
                .completion(request(&model, true))
                .await
                .expect("the turn should succeed");
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
        },
    )
    .await;

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
    with_perplexity_cassette(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("the turn should succeed");
            pause_between_live_turns().await;
            let on = model
                .completion(request(&model, true))
                .await
                .expect("the turn should succeed");
            *sink.lock().expect("observation lock") = Some((off, on));
        },
    )
    .await;

    let (off, on) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe both responses");
    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2);
    assert!(off.raw.is_none());
    assert_reproduces_fixture(&off, &interactions[0].1, "flag off");
    assert_reproduces_fixture(&on, &interactions[1].1, "flag on");

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
