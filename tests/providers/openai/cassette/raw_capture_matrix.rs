//! Raw provider response capture on OpenAI's unary seams
//! (`CompletionResponse::raw`).
//!
//! # What this pins
//!
//! Every `completion()` on either OpenAI route carries `raw`: the value the
//! model's inherent `raw_completion` would have returned, serialized — so it
//! round-trips into the route's own wire type and re-serializes to the same
//! value. There is no switch behind it; `raw` is `None` only on a response
//! constructed without a provider response behind it, never on one that came
//! off the wire. Because capture is unconditional it must be an escape hatch,
//! not a second source of truth: re-normalizing `raw` yields the same
//! `identity()`, `finish_reason()`, `model`, `usage` and `choice` the typed
//! route reported, so `raw` and the normalized response tell one story. Both
//! routes are covered because they have different wire types: Chat
//! Completions' `openai::CompletionResponse` is a derived
//! `Serialize`/`Deserialize` pair, while the Responses API's
//! `openai::responses_api::CompletionResponse` carries a *manual*
//! `Serialize` that mirrors the wire body — so its re-serialization equality
//! is a load-bearing check, not a tautology.
//!
//! The provider-specific field cells read something rig does not normalize —
//! Chat `service_tier` (and `system_fingerprint`), Responses `service_tier`
//! and `store` — off `raw`, equal to the fixture body, and prove the
//! normalized response has no such key.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_round_trips_typed` | chat, unary | `openai::CompletionResponse` round trip; re-normalized `raw` ≡ typed route | recorded |
//! | 2 | `chat_raw_exposes_service_tier` | chat, provider-only field | `raw["service_tier"]` = fixture | recorded |
//! | 3 | `responses_raw_round_trips_typed` | Responses, unary | manual-`Serialize` type round trip; re-normalized `raw` ≡ typed route | recorded |
//! | 4 | `responses_raw_exposes_service_tier_and_store` | Responses, provider-only fields | `raw["service_tier"]`, `raw["store"]` = fixture | recorded |
//!
//! Every cell is recorded; none is unit-only. Each cell re-derives its premise
//! from its own fixture after the wrapper returns: the recorded response is a
//! completed turn whose body carries the field the cell reads.

use std::future::Future;
use std::pin::Pin;

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse as _,
};
use rig::prelude::*;
use rig::providers::openai;
use serde::Deserialize as _;
use serde_json::Value;

use super::super::support::{assert_matches_recorded_token, with_openai_cassette};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

type Observed = std::sync::Arc<std::sync::Mutex<Option<CompletionResponse>>>;

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

/// One `completion()` on the chat route, saved onto `sink`.
fn chat_body(sink: Observed) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completions_api().completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("chat completion should succeed");
            *sink.lock().expect("observation mutex") = Some(response);
        })
    })
}

fn responses_body(sink: Observed) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("responses completion should succeed");
            *sink.lock().expect("observation mutex") = Some(response);
        })
    })
}

fn take(observed: &Observed) -> CompletionResponse {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// The premise shared by every cell: the recorded turn completed with a text
/// choice, and the normalized response reflects that fixture.
fn assert_chat_fixture_premise(scenario: &str, response: &CompletionResponse, body: &Value) {
    assert_eq!(body["object"], "chat.completion", "{scenario}: chat body");
    assert_eq!(
        body["choices"][0]["finish_reason"], "stop",
        "{scenario}: completed turn"
    );
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{scenario}: response_id"),
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        Some(response.usage.input_tokens),
        body["usage"]["prompt_tokens"].as_u64()
    );
}

fn assert_responses_fixture_premise(scenario: &str, response: &CompletionResponse, body: &Value) {
    assert_eq!(body["object"], "response", "{scenario}: responses body");
    assert_eq!(body["status"], "completed", "{scenario}: completed turn");
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{scenario}: response_id"),
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        Some(response.usage.input_tokens),
        body["usage"]["input_tokens"].as_u64()
    );
}

/// The `raw` a wire response must carry — `None` is reserved for values built
/// without a provider response behind them, which a `completion()` result
/// never is.
fn captured_raw<'a>(scenario: &str, response: &'a CompletionResponse) -> &'a Value {
    response
        .raw
        .as_deref()
        .unwrap_or_else(|| panic!("{scenario}: a response off the wire always carries `raw`"))
}

/// `raw` and the typed route tell one story: normalizing the captured value
/// again reproduces every field the typed route reported. The transport id is
/// the one exception by construction — it lives in a response header, not the
/// body `raw` mirrors — so the re-normalized identity is compared without it.
fn assert_raw_renormalizes_to(
    scenario: &str,
    typed: &CompletionResponse,
    renormalized: &CompletionResponse,
) {
    assert_eq!(typed.choice, renormalized.choice, "{scenario}: choice");
    assert_eq!(typed.usage, renormalized.usage, "{scenario}: usage");
    assert_eq!(typed.model, renormalized.model, "{scenario}: model");
    assert_eq!(
        typed.provider, renormalized.provider,
        "{scenario}: provider"
    );
    assert_eq!(
        typed.finish_reason(),
        renormalized.finish_reason(),
        "{scenario}: finish reason"
    );
    let typed_identity = typed.identity();
    let renormalized_identity = renormalized.identity();
    assert_eq!(
        typed_identity.response_id, renormalized_identity.response_id,
        "{scenario}: response id"
    );
    assert_eq!(
        typed_identity.message_id, renormalized_identity.message_id,
        "{scenario}: message id"
    );
    assert!(
        typed_identity.provider_request_id.is_some(),
        "{scenario}: the typed route reports the transport id"
    );
    assert_eq!(
        renormalized_identity.provider_request_id, None,
        "{scenario}: the transport id is a header, so re-normalizing the body has none"
    );
}

/// The normalized response's own serialization has no key by that name — the
/// field is reachable through `raw` alone.
fn assert_normalized_lacks_key(scenario: &str, response: &CompletionResponse, key: &str) {
    let normalized = serde_json::to_value(response).expect("normalized response serializes");
    assert!(
        normalized.get(key).is_none(),
        "{scenario}: the normalized response must not model `{key}` — that is what \
         makes it a provider-only field"
    );
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/chat_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_raw_round_trips_typed",
        chat_body(observed.clone()),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &response, &body);

    let raw = captured_raw(SCENARIO, &response);
    let typed = openai::CompletionResponse::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the chat wire type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed response serializes"),
        *raw,
        "{SCENARIO}: the typed round trip must re-serialize to the captured value"
    );
    // The captured value is *this* response: id, model, choice, usage.
    assert_matches_recorded_token(
        Some(typed.id.as_str()),
        body["id"].as_str(),
        &format!("{SCENARIO}: raw id"),
    );
    assert_eq!(Some(typed.model.as_str()), body["model"].as_str());
    assert_eq!(
        raw["choices"][0]["finish_reason"],
        body["choices"][0]["finish_reason"]
    );
    // The captured value is the response *as rig's wire type parsed it*: the
    // chat type models assistant content as parts, so a wire string comes back
    // as one text part. Same text, typed shape.
    let raw_text: String = match &raw["choices"][0]["message"]["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect(),
        other => panic!("{SCENARIO}: unexpected raw content shape {other}"),
    };
    assert_eq!(
        Some(raw_text.as_str()),
        body["choices"][0]["message"]["content"].as_str(),
        "{SCENARIO}: raw content text equals the fixture's"
    );
    assert_eq!(
        raw["usage"]["prompt_tokens"],
        body["usage"]["prompt_tokens"]
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        body["usage"]["completion_tokens"]
    );
    // One story: the typed value re-normalizes to what `completion()` reported.
    let renormalized = typed
        .normalize(PROVIDER)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must normalize again: {err}"));
    assert_raw_renormalizes_to(SCENARIO, &response, &renormalized);
}

#[tokio::test]
async fn chat_raw_exposes_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/chat_raw_exposes_service_tier";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_raw_exposes_service_tier",
        chat_body(observed.clone()),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &response, &body);
    // Premise: the recorded body reports a service tier at all.
    let recorded_tier = body["service_tier"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded chat body must carry a string `service_tier`")
    });

    let raw = captured_raw(SCENARIO, &response);
    assert_eq!(
        raw["service_tier"].as_str(),
        Some(recorded_tier),
        "{SCENARIO}: `service_tier` is readable off raw and equals the fixture"
    );
    assert_matches_recorded_token(
        raw["system_fingerprint"].as_str(),
        body["system_fingerprint"].as_str(),
        &format!("{SCENARIO}: `system_fingerprint` off raw vs the fixture"),
    );
    assert_normalized_lacks_key(SCENARIO, &response, "service_tier");
    assert_normalized_lacks_key(SCENARIO, &response, "system_fingerprint");
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// The Responses wire type's `Serialize` is hand-written (it mirrors the wire
/// body and folds three reasoning surfaces back into one `reasoning` key), so
/// `to_value(&typed) == raw` here is a genuine check that the manual impl and
/// the `Deserialize` impl agree on the shape they exchange.
#[tokio::test]
async fn responses_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/responses_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_raw_round_trips_typed",
        responses_body(observed.clone()),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &response, &body);

    let raw = captured_raw(SCENARIO, &response);
    let typed = openai::responses_api::CompletionResponse::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the Responses wire type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed response serializes"),
        *raw,
        "{SCENARIO}: the manual-Serialize round trip must re-serialize to the captured value"
    );
    assert_matches_recorded_token(
        Some(typed.id.as_str()),
        body["id"].as_str(),
        &format!("{SCENARIO}: raw id"),
    );
    assert_eq!(Some(typed.model.as_str()), body["model"].as_str());
    assert_eq!(raw["status"], body["status"]);
    assert_eq!(raw["usage"]["input_tokens"], body["usage"]["input_tokens"]);
    assert_eq!(
        raw["usage"]["output_tokens"],
        body["usage"]["output_tokens"]
    );
    // The transport id is not part of the wire body, so the captured value —
    // which mirrors the wire body — must not carry it, while the normalized
    // response does.
    assert!(
        raw.get("provider_request_id").is_none(),
        "{SCENARIO}: the Responses raw value mirrors the wire body"
    );
    assert!(
        response.provider_request_id.is_some(),
        "{SCENARIO}: the normalized response still reports the transport id"
    );
    // One story: the typed value re-normalizes to what `completion()` reported.
    let renormalized = typed
        .normalize(PROVIDER)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must normalize again: {err}"));
    assert_raw_renormalizes_to(SCENARIO, &response, &renormalized);
}

#[tokio::test]
async fn responses_raw_exposes_service_tier_and_store() {
    const SCENARIO: &str = "raw_capture_matrix/responses_raw_exposes_service_tier_and_store";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_raw_exposes_service_tier_and_store",
        responses_body(observed.clone()),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &response, &body);
    let recorded_tier = body["service_tier"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded Responses body must carry a string `service_tier`")
    });
    let recorded_store = body["store"].as_bool().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded Responses body must carry a boolean `store`")
    });

    let raw = captured_raw(SCENARIO, &response);
    assert_eq!(
        raw["service_tier"].as_str(),
        Some(recorded_tier),
        "{SCENARIO}: `service_tier` is readable off raw and equals the fixture"
    );
    assert_eq!(
        raw["store"].as_bool(),
        Some(recorded_store),
        "{SCENARIO}: `store` is readable off raw and equals the fixture"
    );
    assert_normalized_lacks_key(SCENARIO, &response, "service_tier");
    assert_normalized_lacks_key(SCENARIO, &response, "store");
}
