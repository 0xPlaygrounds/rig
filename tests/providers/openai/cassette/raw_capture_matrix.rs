//! Opt-in raw provider response capture on OpenAI's unary seams
//! (`CompletionRequest::capture_raw_response` → `CompletionResponse::raw`).
//!
//! # What this pins
//!
//! `raw` is `None` unless the request opted in; when it did, it is the value
//! the model's inherent `raw_completion` would have returned, serialized — so
//! it round-trips into the route's own wire type and re-serializes to the same
//! value. The flag is local policy: the request the provider receives is
//! byte-identical with and without it, and every normalized field means the
//! same thing either way. Both routes are covered because they have different
//! wire types: Chat Completions' `openai::CompletionResponse` is a derived
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
//! | 1 | `chat_off_is_none` | chat, flag off | `raw.is_none()` | recorded |
//! | 2 | `chat_on_round_trips_typed` | chat, flag on | `openai::CompletionResponse` round trip | recorded |
//! | 3 | `chat_on_exposes_service_tier` | chat, provider-only field | `raw["service_tier"]` = fixture | recorded |
//! | 4 | `chat_off_on_request_invariant` | chat, off then on | identical request bytes, identical normalized fields | recorded |
//! | 5 | `responses_off_is_none` | Responses, flag off | `raw.is_none()` | recorded |
//! | 6 | `responses_on_round_trips_typed` | Responses, flag on | manual-`Serialize` type round trip | recorded |
//! | 7 | `responses_on_exposes_service_tier_and_store` | Responses, provider-only fields | `raw["service_tier"]`, `raw["store"]` = fixture | recorded |
//! | 8 | `responses_off_on_request_invariant` | Responses, off then on | identical request bytes, identical normalized fields | recorded |
//!
//! Every cell is recorded; none is unit-only. Each cell re-derives its premise
//! from its own fixture after the wrapper returns: the recorded response is a
//! completed turn whose body carries the field the cell reads.

use std::future::Future;
use std::pin::Pin;

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse, FinishReason};
use rig::prelude::*;
use rig::providers::openai;
use serde::Deserialize as _;
use serde_json::Value;

use super::super::support::{assert_matches_recorded_token, with_openai_cassette};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &(impl CompletionModel + Clone), capture_raw: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .capture_raw_response(capture_raw)
        .build()
}

type Observed = std::sync::Arc<std::sync::Mutex<Vec<CompletionResponse>>>;

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

/// One `completion()` per flag on the chat route, in one scenario (one
/// interaction per flag), pushed onto `sink` in order.
fn chat_body(sink: Observed, flags: &'static [bool]) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completions_api().completion_model(MODEL);
            for &capture_raw in flags {
                let response = model
                    .completion(request(&model, capture_raw))
                    .await
                    .expect("chat completion should succeed");
                sink.lock().expect("observation mutex").push(response);
            }
        })
    })
}

fn responses_body(sink: Observed, flags: &'static [bool]) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completion_model(MODEL);
            for &capture_raw in flags {
                let response = model
                    .completion(request(&model, capture_raw))
                    .await
                    .expect("responses completion should succeed");
                sink.lock().expect("observation mutex").push(response);
            }
        })
    })
}

fn take(observed: &Observed, scenario: &str, expected: usize) -> Vec<CompletionResponse> {
    let responses = std::mem::take(&mut *observed.lock().expect("observation mutex"));
    assert_eq!(
        responses.len(),
        expected,
        "{scenario}: one response per flag"
    );
    responses
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

/// The normalized fields the flag must not perturb.
fn assert_normalized_fields_equal(
    scenario: &str,
    off: &CompletionResponse,
    on: &CompletionResponse,
) {
    assert_eq!(off.choice, on.choice, "{scenario}: choice");
    assert_eq!(off.usage, on.usage, "{scenario}: usage");
    assert_eq!(off.model, on.model, "{scenario}: model");
    assert_eq!(off.provider, on.provider, "{scenario}: provider");
    assert_eq!(
        off.finish_reason(),
        on.finish_reason(),
        "{scenario}: finish reason"
    );
    assert_eq!(
        off.response_id.is_some(),
        on.response_id.is_some(),
        "{scenario}: response id presence"
    );
    assert_eq!(
        off.provider_request_id.is_some(),
        on.provider_request_id.is_some(),
        "{scenario}: request id presence"
    );
    assert_eq!(
        off.message_id.is_some(),
        on.message_id.is_some(),
        "{scenario}: message id presence"
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
async fn chat_off_is_none() {
    const SCENARIO: &str = "raw_capture_matrix/chat_off_is_none";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_off_is_none",
        chat_body(observed.clone(), &[false]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &responses[0], &body);
    assert!(
        responses[0].raw.is_none(),
        "{SCENARIO}: capture was not requested, so `raw` must be None"
    );
}

#[tokio::test]
async fn chat_on_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/chat_on_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_on_round_trips_typed",
        chat_body(observed.clone(), &[true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &responses[0], &body);

    let raw = responses[0]
        .raw
        .as_deref()
        .unwrap_or_else(|| panic!("{SCENARIO}: capture was requested, so `raw` must be Some"));
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
}

#[tokio::test]
async fn chat_on_exposes_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/chat_on_exposes_service_tier";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_on_exposes_service_tier",
        chat_body(observed.clone(), &[true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &responses[0], &body);
    // Premise: the recorded body reports a service tier at all.
    let recorded_tier = body["service_tier"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded chat body must carry a string `service_tier`")
    });

    let raw = responses[0]
        .raw
        .as_deref()
        .unwrap_or_else(|| panic!("{SCENARIO}: capture was requested, so `raw` must be Some"));
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
    assert_normalized_lacks_key(SCENARIO, &responses[0], "service_tier");
    assert_normalized_lacks_key(SCENARIO, &responses[0], "system_fingerprint");
}

#[tokio::test]
async fn chat_off_on_request_invariant() {
    const SCENARIO: &str = "raw_capture_matrix/chat_off_on_request_invariant";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_off_on_request_invariant",
        chat_body(observed.clone(), &[false, true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 2);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(bodies.len(), 2, "{SCENARIO}: one interaction per flag");
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "{SCENARIO}: the flag is local policy — the request the provider received must be \
         byte-identical with it off and on"
    );
    let request: Value = serde_json::from_str(&bodies[0].0).expect("request should be JSON");
    assert!(
        request.get("capture_raw_response").is_none(),
        "{SCENARIO}: the flag never serializes onto the wire"
    );
    let off_body: Value = serde_json::from_str(&bodies[0].1).expect("body should be JSON");
    let on_body: Value = serde_json::from_str(&bodies[1].1).expect("body should be JSON");
    assert_chat_fixture_premise(SCENARIO, &responses[0], &off_body);
    assert_chat_fixture_premise(SCENARIO, &responses[1], &on_body);
    assert!(responses[0].raw.is_none(), "{SCENARIO}: off → None");
    assert!(responses[1].raw.is_some(), "{SCENARIO}: on → Some");
    assert_normalized_fields_equal(SCENARIO, &responses[0], &responses[1]);
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_off_is_none() {
    const SCENARIO: &str = "raw_capture_matrix/responses_off_is_none";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_off_is_none",
        responses_body(observed.clone(), &[false]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &responses[0], &body);
    assert!(
        responses[0].raw.is_none(),
        "{SCENARIO}: capture was not requested, so `raw` must be None"
    );
}

/// The Responses wire type's `Serialize` is hand-written (it mirrors the wire
/// body and folds three reasoning surfaces back into one `reasoning` key), so
/// `to_value(&typed) == raw` here is a genuine check that the manual impl and
/// the `Deserialize` impl agree on the shape they exchange.
#[tokio::test]
async fn responses_on_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/responses_on_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_on_round_trips_typed",
        responses_body(observed.clone(), &[true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &responses[0], &body);

    let raw = responses[0]
        .raw
        .as_deref()
        .unwrap_or_else(|| panic!("{SCENARIO}: capture was requested, so `raw` must be Some"));
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
        responses[0].provider_request_id.is_some(),
        "{SCENARIO}: the normalized response still reports the transport id"
    );
}

#[tokio::test]
async fn responses_on_exposes_service_tier_and_store() {
    const SCENARIO: &str = "raw_capture_matrix/responses_on_exposes_service_tier_and_store";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_on_exposes_service_tier_and_store",
        responses_body(observed.clone(), &[true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 1);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &responses[0], &body);
    let recorded_tier = body["service_tier"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded Responses body must carry a string `service_tier`")
    });
    let recorded_store = body["store"].as_bool().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded Responses body must carry a boolean `store`")
    });

    let raw = responses[0]
        .raw
        .as_deref()
        .unwrap_or_else(|| panic!("{SCENARIO}: capture was requested, so `raw` must be Some"));
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
    assert_normalized_lacks_key(SCENARIO, &responses[0], "service_tier");
    assert_normalized_lacks_key(SCENARIO, &responses[0], "store");
}

#[tokio::test]
async fn responses_off_on_request_invariant() {
    const SCENARIO: &str = "raw_capture_matrix/responses_off_on_request_invariant";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_off_on_request_invariant",
        responses_body(observed.clone(), &[false, true]),
    )
    .await;
    let responses = take(&observed, SCENARIO, 2);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(bodies.len(), 2, "{SCENARIO}: one interaction per flag");
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "{SCENARIO}: the flag is local policy — the request the provider received must be \
         byte-identical with it off and on"
    );
    let request: Value = serde_json::from_str(&bodies[0].0).expect("request should be JSON");
    assert!(
        request.get("capture_raw_response").is_none(),
        "{SCENARIO}: the flag never serializes onto the wire"
    );
    let off_body: Value = serde_json::from_str(&bodies[0].1).expect("body should be JSON");
    let on_body: Value = serde_json::from_str(&bodies[1].1).expect("body should be JSON");
    assert_responses_fixture_premise(SCENARIO, &responses[0], &off_body);
    assert_responses_fixture_premise(SCENARIO, &responses[1], &on_body);
    assert!(responses[0].raw.is_none(), "{SCENARIO}: off → None");
    assert!(responses[1].raw.is_some(), "{SCENARIO}: on → Some");
    assert_normalized_fields_equal(SCENARIO, &responses[0], &responses[1]);
}
