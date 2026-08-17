//! Raw provider response capture on Venice's blocking chat-completions path.
//!
//! **The feature.** Every blocking completion attaches the value the model's
//! inherent `raw_completion` returned — Venice's own
//! [`venice::CompletionResponse`], serialized — onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. Capture is always on: there
//! is no flag to request it, nothing about it reaches the wire, and a `None`
//! only ever means a response built by hand with no provider payload behind
//! it. `raw` is a second view of the same response, never a substitute for a
//! normalized field. Venice's payload is OpenAI's plus the resolved
//! `venice_parameters` echo and the request's `cost`; neither has a slot on
//! the normalized response, so they are the fields pinned here as reachable
//! only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_venice_type` | typed round trip | `raw` deserializes into `venice::CompletionResponse` and re-serializes equal | recorded |
//! | 2 | `raw_exposes_venice_parameters_and_cost` | provider-only field | `raw.venice_parameters.disable_thinking` and `raw.cost.usd` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes and equals its own `raw` re-normalized | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the echo and the cost out of the
//! recorded body rather than trusting what the typed view reports, and cell 3
//! checks the normalized fields against the recorded body before comparing
//! them with the re-normalized `raw`, so a recording that stopped carrying a
//! usage block or a finish reason fails loudly instead of covering nothing.
//! Venice contracts no request-id header, so `provider_request_id` is `None`
//! on every turn here — a documented outcome, pinned as such. Thinking is
//! disabled through `venice_parameters` so the small reasoning model answers
//! in plain text within the token budget.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::venice;
use rig::providers::venice::completion::VeniceParameters;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::DEFAULT_MODEL;
use super::super::support::{assert_matches_recorded_token, with_venice_cassette_result};

const PROVIDER: &str = "venice";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &venice::CompletionModel) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .additional_params(
            VeniceParameters::new()
                .disable_thinking(true)
                .into_additional_params(),
        )
        .build()
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
    // Venice contracts no request-id header, so `None` is the documented
    // outcome.
    assert_eq!(response.provider_request_id, None, "request id");
}

// ================================================================
// 1. raw round-trips Venice's own type
// ================================================================

#[tokio::test]
async fn raw_round_trips_venice_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_venice_type";
    with_venice_cassette_result(
        "raw_capture_matrix/raw_round_trips_venice_type",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model.completion(request(&model)).await?;
            let raw = response
                .raw
                .as_deref()
                .expect("every provider-backed response carries raw");
            let typed = venice::CompletionResponse::deserialize(raw)
                .expect("raw is Venice's own CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed view serialized, nothing more"
            );
            assert_eq!(
                Some(typed.openai.id.as_str()),
                response.response_id.as_deref()
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_round_trips_venice_type should replay from its cassette");

    let (_, response_body) = recorded_json(SCENARIO);
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_venice_parameters_and_cost() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_venice_parameters_and_cost";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_venice_cassette_result(
        "raw_capture_matrix/raw_exposes_venice_parameters_and_cost",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model.completion(request(&model)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_exposes_venice_parameters_and_cost should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (request_body, body) = recorded_json(SCENARIO);
    assert_eq!(
        request_body["venice_parameters"]["disable_thinking"],
        json!(true),
        "the request asked Venice to disable thinking"
    );
    let recorded_echo = body["venice_parameters"]["disable_thinking"]
        .as_bool()
        .expect("Venice echoes the resolved venice_parameters block");
    let recorded_cost = body["cost"]["usd"]
        .as_f64()
        .expect("Venice reports what the request cost");

    let raw = response
        .raw
        .as_deref()
        .expect("every provider-backed response carries raw");
    assert_eq!(
        raw["venice_parameters"]["disable_thinking"],
        json!(recorded_echo)
    );
    assert_eq!(raw["cost"]["usd"], json!(recorded_cost));
    // And the normalized view has no slot for either.
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("venice_parameters").is_none());
    assert!(normalized.get("cost").is_none());
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_venice_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
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
    assert_reproduces_fixture(&response, &body);

    // The normalized fields are exactly what the response's own raw
    // re-normalizes to: capture adds a view, it never changes the mapping.
    let raw = response
        .raw
        .as_deref()
        .expect("every provider-backed response carries raw");
    let renormalized = venice::CompletionResponse::deserialize(raw)
        .expect("raw is Venice's own type")
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
