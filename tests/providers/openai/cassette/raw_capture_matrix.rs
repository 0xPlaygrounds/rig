//! Raw provider response capture on OpenAI's unary seams
//! (`CompletionResponse::raw`).
//!
//! # What this pins
//!
//! Every `completion()` on either OpenAI route carries `raw`: the value the
//! model's inherent `raw_completion` would have returned, serialized — so it
//! round-trips into the route's own wire type and re-serializes to the same
//! value. There is no switch behind it; `raw` is `Value::Null` only on a
//! response constructed without a provider response behind it, never on one
//! that came off the wire. Because capture is unconditional it must be an escape hatch,
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
//! Text turns are the easy case. The wire shapes most likely to break the
//! round trip are the ones the typed routes rewrite on the way in: a
//! Responses reasoning turn (an `output[]` item of `type: "reasoning"` with
//! `encrypted_content` and `summary`, folded by the manual `Serialize` back
//! under one `reasoning` key), a Chat tool call (whose
//! `function.arguments` is a JSON *string* on the wire and stays one on
//! `raw`, while the normalized `finish_reason()` is `ToolCalls` and the raw
//! spelling is OpenAI's own `"tool_calls"`), and a Chat structured-output
//! turn (`response_format: json_schema`, whose message carries a `refusal`
//! sibling and whose body carries `system_fingerprint`). Cells 5–7 record
//! each of those and hold the same round-trip / one-story bar.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_round_trips_typed` | chat, unary | `openai::CompletionResponse` round trip; re-normalized `raw` ≡ typed route | recorded |
//! | 2 | `chat_raw_exposes_service_tier` | chat, provider-only field | `raw["service_tier"]` = fixture | recorded |
//! | 3 | `responses_raw_round_trips_typed` | Responses, unary | manual-`Serialize` type round trip; re-normalized `raw` ≡ typed route | recorded |
//! | 4 | `responses_raw_exposes_service_tier_and_store` | Responses, provider-only fields | `raw["service_tier"]`, `raw["store"]` = fixture | recorded |
//! | 5 | `responses_reasoning_raw_round_trips_typed` | Responses, reasoning turn (`reasoning: { effort, summary }`) | round trip; `raw["output"][i].type == "reasoning"` with `encrypted_content` + `summary` = fixture; `raw["reasoning"]` = fixture | recorded |
//! | 6 | `chat_tool_call_raw_round_trips_typed` | chat, forced tool call (`tool_choice: required`) | round trip; `tool_calls[0].function.arguments` is a JSON string = fixture; `finish_reason() == ToolCalls` while `raw` spells `"tool_calls"` | recorded |
//! | 7 | `chat_structured_output_raw_exposes_system_fingerprint` | chat, `response_format: json_schema` | round trip; `raw["system_fingerprint"]` = fixture; fixture message carries `refusal`; content parses under the schema | recorded |
//!
//! Every cell is recorded; none is unit-only. Each cell re-derives its premise
//! from its own fixture after the wrapper returns: the recorded response is a
//! completed turn whose body carries the field the cell reads — for cell 5 a
//! `reasoning` output item with a string `encrypted_content` (and the recorded
//! request asked for it via `include`), for cell 6 a `tool_calls` entry with
//! `finish_reason: "tool_calls"`, for cell 7 a request carrying
//! `response_format.type == "json_schema"` and a body with a string
//! `system_fingerprint`.

use std::future::Future;
use std::pin::Pin;

use rig::completion::{
    AssistantContent, CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse as _, ToolDefinition,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::Deserialize as _;
use serde_json::{Value, json};

use super::super::support::{assert_matches_recorded_token, with_openai_cassette};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
/// The reasoning cell needs a model that emits `reasoning` output items; the
/// same one the `reasoning_roundtrip` module records against.
const REASONING_MODEL: &str = openai::GPT_5_2;
const PROMPT: &str = "Reply with exactly the single word: pong";
/// A problem small enough to answer in a few tokens but not so trivial that
/// the model skips reasoning: at `effort: "low"` a one-step multiplication
/// came back with no `reasoning` output item at all, and a cell that cannot
/// find one fails on its premise.
const REASONING_PROMPT: &str = "A train leaves at 09:30 and travels 150 km at 60 km/h. \
     At what time does it arrive? Reply with only the time in HH:MM.";
const TOOL_PROMPT: &str = "Call ping exactly once with no arguments.";
const STRUCTURED_PROMPT: &str = "Put the single word pong in the `word` field.";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

/// The reasoning request shape the `reasoning_roundtrip` module uses
/// (`effort: "medium"`), with a summary asked for so the reasoning item
/// carries `summary` as well as the `encrypted_content` the provider adds to
/// `include` on every reasoning request.
fn reasoning_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(REASONING_PROMPT)
        .additional_params(json!({
            "reasoning": { "effort": "medium", "summary": "auto" }
        }))
        .build()
}

fn ping_tool() -> ToolDefinition {
    ToolDefinition {
        name: "ping".to_owned(),
        description: "Matrix tool ping".to_owned(),
        parameters: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
    }
}

/// The forced tool call `raw_completion_parity_matrix` records: `required`
/// leaves the model no text-only exit.
fn tool_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .tool(ping_tool())
        .tool_choice(ToolChoice::Required)
        .temperature(0.0)
        .max_tokens(64)
        .build()
}

/// The structured-output schema: one required string field, so the chat
/// route maps it to `response_format: { type: "json_schema", … }`.
#[derive(Debug, serde::Deserialize, JsonSchema)]
struct Answer {
    word: String,
}

fn structured_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(STRUCTURED_PROMPT)
        .output_schema(schemars::schema_for!(Answer))
        .temperature(0.0)
        .max_tokens(32)
        .build()
}

type Observed = std::sync::Arc<std::sync::Mutex<Option<CompletionResponse>>>;

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

/// One `completion()` on the chat route with the request `build` makes for
/// the model, saved onto `sink`.
fn chat_body_with(
    sink: Observed,
    build: impl FnOnce(&openai::CompletionModel) -> CompletionRequest + 'static,
) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completions_api().completion_model(MODEL);
            let response = model
                .completion(build(&model))
                .await
                .expect("chat completion should succeed");
            *sink.lock().expect("observation mutex") = Some(response);
        })
    })
}

/// The text turn every original chat cell records.
fn chat_body(sink: Observed) -> Body {
    chat_body_with(sink, request)
}

fn responses_body_with(
    sink: Observed,
    model_name: &'static str,
    build: impl FnOnce(&openai::ResponsesCompletionModel) -> CompletionRequest + 'static,
) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completion_model(model_name);
            let response = model
                .completion(build(&model))
                .await
                .expect("responses completion should succeed");
            *sink.lock().expect("observation mutex") = Some(response);
        })
    })
}

fn responses_body(sink: Observed) -> Body {
    responses_body_with(sink, MODEL, request)
}

fn take(observed: &Observed) -> CompletionResponse {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// The premise shared by every chat cell: the recorded turn completed the way
/// the cell expects — `wire_finish` is OpenAI's own spelling in the fixture,
/// `finish` the normalized reading of it — and the normalized response
/// reflects that fixture.
fn assert_chat_fixture_premise(
    scenario: &str,
    response: &CompletionResponse,
    body: &Value,
    wire_finish: &str,
    finish: FinishReason,
) {
    assert_eq!(body["object"], "chat.completion", "{scenario}: chat body");
    assert_eq!(
        body["choices"][0]["finish_reason"], wire_finish,
        "{scenario}: the recorded turn finished as the cell expects"
    );
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{scenario}: response_id"),
    );
    assert_eq!(
        response.finish_reason(),
        Some(finish),
        "{scenario}: normalized finish reason"
    );
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

/// The `raw` a wire response must carry — `Value::Null` is reserved for
/// values built without a provider response behind them, which a
/// `completion()` result never is.
fn captured_raw<'a>(scenario: &str, response: &'a CompletionResponse) -> &'a Value {
    assert!(
        !response.raw.is_null(),
        "{scenario}: a response off the wire always carries `raw`"
    );
    &response.raw
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
    assert_chat_fixture_premise(SCENARIO, &response, &body, "stop", FinishReason::Stop);

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
    assert_chat_fixture_premise(SCENARIO, &response, &body, "stop", FinishReason::Stop);
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

// ---------------------------------------------------------------------------
// Reasoning, tool-call and structured-output turns
// ---------------------------------------------------------------------------

/// The `output[]` item of `type: "reasoning"` in a Responses body, with the
/// premise that it exists and carries a string `encrypted_content`.
fn reasoning_output_item<'a>(scenario: &str, body: &'a Value, what: &str) -> &'a Value {
    let item = body["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "reasoning"))
        .unwrap_or_else(|| panic!("{scenario}: {what} must carry a `reasoning` output item"));
    assert!(
        item["encrypted_content"].is_string(),
        "{scenario}: {what}'s reasoning item must carry a string `encrypted_content`"
    );
    let summary = item["summary"].as_array().unwrap_or_else(|| {
        panic!("{scenario}: {what}'s reasoning item must carry a `summary` array")
    });
    assert!(
        summary
            .iter()
            .any(|entry| entry["type"] == "summary_text" && entry["text"].is_string()),
        "{scenario}: {what}'s reasoning item must carry a `summary_text` entry — that is \
         what `summary: \"auto\"` asked for"
    );
    item
}

/// A Responses reasoning turn: the wire's `reasoning` output item — the one
/// shape the manual `Serialize` has to rebuild rather than pass through — is
/// on `raw` verbatim (its `encrypted_content`, its `summary`), the top-level
/// `reasoning` echo equals the fixture's, and the normalized response carries
/// the same reasoning as content blocks, not as an `output` array.
#[tokio::test]
async fn responses_reasoning_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/responses_reasoning_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/responses_reasoning_raw_round_trips_typed",
        responses_body_with(observed.clone(), REASONING_MODEL, reasoning_request),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_responses_fixture_premise(SCENARIO, &response, &body);
    // Premise: the recorded request was a reasoning request that asked for
    // the encrypted content and a summary, and the recorded body answered
    // with a reasoning item carrying both.
    let request = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(
        request["reasoning"],
        json!({ "effort": "medium", "summary": "auto" }),
        "{SCENARIO}: the recorded request carries the reasoning shape"
    );
    assert_eq!(
        request["include"],
        json!(["reasoning.encrypted_content"]),
        "{SCENARIO}: the provider asks for encrypted reasoning on every reasoning request"
    );
    let recorded_item = reasoning_output_item(SCENARIO, &body, "the recorded body");
    assert!(
        body["reasoning"].is_object(),
        "{SCENARIO}: the recorded body echoes the reasoning configuration as an object"
    );

    let raw = captured_raw(SCENARIO, &response);
    let typed = openai::responses_api::CompletionResponse::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the Responses wire type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed response serializes"),
        *raw,
        "{SCENARIO}: the manual-Serialize round trip must re-serialize to the captured value"
    );
    // The reasoning item is on raw as the wire item.
    let raw_item = reasoning_output_item(SCENARIO, raw, "raw");
    assert_matches_recorded_token(
        raw_item["encrypted_content"].as_str(),
        recorded_item["encrypted_content"].as_str(),
        &format!("{SCENARIO}: `encrypted_content` off raw vs the fixture"),
    );
    assert_matches_recorded_token(
        raw_item["id"].as_str(),
        recorded_item["id"].as_str(),
        &format!("{SCENARIO}: reasoning item id off raw vs the fixture"),
    );
    assert_eq!(
        raw_item["summary"], recorded_item["summary"],
        "{SCENARIO}: `summary` off raw equals the fixture's"
    );
    assert_eq!(
        raw["reasoning"], body["reasoning"],
        "{SCENARIO}: the top-level `reasoning` echo is folded back to the wire's object"
    );
    assert_eq!(
        raw["usage"]["output_tokens_details"]["reasoning_tokens"],
        body["usage"]["output_tokens_details"]["reasoning_tokens"],
        "{SCENARIO}: reasoning token usage off raw equals the fixture's"
    );
    // The normalized response has no `output` array; its reasoning is a
    // content block carrying the same encrypted payload.
    assert_normalized_lacks_key(SCENARIO, &response, "output");
    let encrypted_blocks: Vec<&str> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .flat_map(|reasoning| reasoning.content.iter())
        .filter_map(|block| match block {
            rig::message::ReasoningContent::Encrypted(data) => Some(data.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        encrypted_blocks,
        vec![
            raw_item["encrypted_content"]
                .as_str()
                .expect("premise: string")
        ],
        "{SCENARIO}: the normalized reasoning block carries raw's encrypted content"
    );
    // One story: the typed value re-normalizes to what `completion()` reported.
    let renormalized = typed
        .normalize(PROVIDER)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must normalize again: {err}"));
    assert_raw_renormalizes_to(SCENARIO, &response, &renormalized);
}

/// A forced Chat tool call: `raw` keeps the wire's representation — a
/// `tool_calls` entry whose `function.arguments` is a JSON *string* and a
/// `finish_reason` spelled `"tool_calls"` — while the normalized response
/// reports `FinishReason::ToolCalls` and a parsed tool call.
#[tokio::test]
async fn chat_tool_call_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_capture_matrix/chat_tool_call_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_tool_call_raw_round_trips_typed",
        chat_body_with(observed.clone(), tool_request),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(
        SCENARIO,
        &response,
        &body,
        "tool_calls",
        FinishReason::ToolCalls,
    );
    // Premise: the recorded request forced the tool and the recorded body
    // carries exactly one call to it, with string arguments.
    let request = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(
        request["tool_choice"], "required",
        "{SCENARIO}: forced tool call"
    );
    assert_eq!(request["tools"][0]["function"]["name"], "ping");
    let recorded_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("{SCENARIO}: the recorded body must carry `tool_calls`"));
    assert_eq!(
        recorded_calls.len(),
        1,
        "{SCENARIO}: one recorded tool call"
    );
    let recorded_arguments = recorded_calls[0]["function"]["arguments"]
        .as_str()
        .unwrap_or_else(|| {
            panic!("{SCENARIO}: the wire spells `function.arguments` as a JSON string")
        });
    assert_eq!(recorded_calls[0]["function"]["name"], "ping");

    let raw = captured_raw(SCENARIO, &response);
    let typed = openai::CompletionResponse::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the chat wire type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed response serializes"),
        *raw,
        "{SCENARIO}: the typed round trip must re-serialize to the captured value"
    );
    // The wire's tool-call representation survives on raw: arguments as a
    // JSON string (equal to the fixture's, as JSON), the provider's own
    // finish-reason spelling.
    let raw_calls = raw["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("{SCENARIO}: raw must carry `tool_calls`"));
    assert_eq!(raw_calls.len(), 1, "{SCENARIO}: one tool call on raw");
    let raw_arguments = raw_calls[0]["function"]["arguments"]
        .as_str()
        .unwrap_or_else(|| panic!("{SCENARIO}: `function.arguments` stays a JSON string on raw"));
    assert_eq!(
        serde_json::from_str::<Value>(raw_arguments).expect("raw arguments are JSON"),
        serde_json::from_str::<Value>(recorded_arguments).expect("recorded arguments are JSON"),
        "{SCENARIO}: raw arguments equal the fixture's"
    );
    assert_eq!(raw_calls[0]["function"]["name"], "ping");
    assert_matches_recorded_token(
        raw_calls[0]["id"].as_str(),
        recorded_calls[0]["id"].as_str(),
        &format!("{SCENARIO}: tool call id off raw vs the fixture"),
    );
    assert_eq!(
        raw["choices"][0]["finish_reason"], "tool_calls",
        "{SCENARIO}: raw keeps OpenAI's own spelling"
    );
    assert_eq!(
        raw["choices"][0]["finish_reason"],
        body["choices"][0]["finish_reason"]
    );
    // The normalized side parsed the same call.
    let normalized_calls: Vec<(&str, &Value)> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => {
                Some((call.function.name.as_str(), &call.function.arguments))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        normalized_calls,
        vec![(
            "ping",
            &serde_json::from_str::<Value>(raw_arguments).expect("raw arguments are JSON")
        )],
        "{SCENARIO}: the normalized tool call is raw's, parsed"
    );
    // One story: the typed value re-normalizes to what `completion()` reported.
    let renormalized = typed
        .normalize(PROVIDER)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must normalize again: {err}"));
    assert_raw_renormalizes_to(SCENARIO, &response, &renormalized);
}

/// A Chat structured-output turn (`response_format: json_schema` via the
/// builder's `output_schema`): `raw` round-trips, its message content is the
/// schema-shaped JSON the fixture carries, and `system_fingerprint` — which
/// the normalized response does not model — is readable off `raw` and equals
/// the fixture's. The wire's `refusal` sibling on the message is proven
/// present in the fixture (`null` on a compliant turn); rig's chat type
/// omits an absent refusal on the way back out, so it is a fixture-level
/// fact, not a `raw` one.
#[tokio::test]
async fn chat_structured_output_raw_exposes_system_fingerprint() {
    const SCENARIO: &str =
        "raw_capture_matrix/chat_structured_output_raw_exposes_system_fingerprint";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_matrix/chat_structured_output_raw_exposes_system_fingerprint",
        chat_body_with(observed.clone(), structured_request),
    )
    .await;
    let response = take(&observed);
    let body = crate::cassettes::recorded_json_response(PROVIDER, SCENARIO);
    assert_chat_fixture_premise(SCENARIO, &response, &body, "stop", FinishReason::Stop);
    // Premise: the recorded request asked for a strict JSON schema, and the
    // recorded body carries the fields the cell reads.
    let request = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(
        request["response_format"]["type"], "json_schema",
        "{SCENARIO}: the recorded request carries `response_format: json_schema`"
    );
    assert_eq!(request["response_format"]["json_schema"]["name"], "Answer");
    assert_eq!(request["response_format"]["json_schema"]["strict"], true);
    let recorded_fingerprint = body["system_fingerprint"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the recorded chat body must carry a string `system_fingerprint`")
    });
    let recorded_message = body["choices"][0]["message"]
        .as_object()
        .unwrap_or_else(|| panic!("{SCENARIO}: the recorded body carries a message object"));
    assert!(
        recorded_message.contains_key("refusal"),
        "{SCENARIO}: the wire's structured-output message carries a `refusal` sibling"
    );
    assert_eq!(
        recorded_message["refusal"],
        Value::Null,
        "{SCENARIO}: a compliant turn's `refusal` is null"
    );
    let recorded_content = recorded_message["content"]
        .as_str()
        .unwrap_or_else(|| panic!("{SCENARIO}: the recorded content is a JSON string"));
    let recorded_answer: Answer = serde_json::from_str(recorded_content).unwrap_or_else(|err| {
        panic!("{SCENARIO}: recorded content parses under the schema: {err}")
    });

    let raw = captured_raw(SCENARIO, &response);
    let typed = openai::CompletionResponse::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the chat wire type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed response serializes"),
        *raw,
        "{SCENARIO}: the typed round trip must re-serialize to the captured value"
    );
    // The fingerprint is a scrubbed token in the fixture (`fp_REDACTED_n`),
    // so replay compares it exactly and record mode compares presence.
    assert_matches_recorded_token(
        raw["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        &format!("{SCENARIO}: `system_fingerprint` is readable off raw and equals the fixture"),
    );
    assert_normalized_lacks_key(SCENARIO, &response, "system_fingerprint");
    // The structured content is on raw as the chat type parsed it: one text
    // part holding the schema-shaped JSON.
    let raw_text: String = match &raw["choices"][0]["message"]["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect(),
        other => panic!("{SCENARIO}: unexpected raw content shape {other}"),
    };
    assert_eq!(
        raw_text, recorded_content,
        "{SCENARIO}: raw content text equals the fixture's"
    );
    let raw_answer: Answer = serde_json::from_str(&raw_text)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw content parses under the schema: {err}"));
    assert_eq!(raw_answer.word, recorded_answer.word);
    assert!(
        !raw_answer.word.trim().is_empty(),
        "{SCENARIO}: the schema's `word` field is filled"
    );
    // One story: the typed value re-normalizes to what `completion()` reported.
    let renormalized = typed
        .normalize(PROVIDER)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must normalize again: {err}"));
    assert_raw_renormalizes_to(SCENARIO, &response, &renormalized);
}
