//! Raw provider response capture on Mistral's blocking chat-completions path.
//!
//! **The feature.** Every blocking completion attaches the value the model's
//! inherent `raw_completion` returned — Mistral's own
//! [`mistral::CompletionResponse`], serialized — onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a
//! `Value::Null` only ever means a response built by hand with no provider
//! payload behind it. `raw` is a second view of the same response, never a
//! substitute for a normalized field. Mistral's wire carries envelope metadata
//! the normalized response has no slot for — the `object` tag and the capacity
//! tier Mistral reports inside `usage` (`usage.service_tier`) — so those are
//! the fields pinned here as reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_mistral_type` | typed round trip | `raw` deserializes into `mistral::CompletionResponse` and re-serializes equal | recorded |
//! | 2 | `raw_exposes_object_and_service_tier` | provider-only field | `raw.object` and `raw.usage.service_tier` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes (including the `mistral-correlation-id` header) and equals its own `raw` re-normalized | recorded |
//! | 4 | `tool_call_raw_round_trips_and_exposes_wire_tool_call` | forced tool call | a `tool_choice: any` turn's `raw` round-trips into `mistral::CompletionResponse`; `raw.choices[0].message.tool_calls[0].function.arguments` is a JSON *string* that parses to the fixture's arguments while the normalized call carries an object; `finish_reason()` is `ToolCalls` while `raw.choices[0].finish_reason` is the wire's `"tool_calls"` | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the tier out of the recorded body
//! rather than trusting the string the typed view reports, cell 3 checks
//! the normalized fields against the recorded body and headers before
//! comparing them with the re-normalized `raw`, and cell 4 reads the tool
//! call (id, name, stringified arguments) and the `"tool_calls"` finish out
//! of the recorded body, so a recording that stopped carrying a usage block,
//! a finish reason, the correlation-id header, or a tool call fails loudly
//! instead of covering nothing.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse, ToolDefinition,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::mistral;
use serde::Deserialize;
use serde_json::{Value, json};

use super::DEFAULT_MODEL;
use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_mistral_cassette_result,
};

const PROVIDER: &str = "mistral";
const PROMPT: &str = "Reply with the single word: pong";
/// The forced-call request shape the tool-lifecycle matrix uses: a preamble
/// that forbids prose, `tool_choice: any`, and a prompt naming the one call.
const TOOL_PREAMBLE: &str =
    "Follow the user's tool-call instruction exactly. Do not answer in prose.";
const TOOL_PROMPT: &str = "Call lookup_city exactly once with city Paris.";
const TOOL_NAME: &str = "lookup_city";

fn request(model: &mistral::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

fn lookup_city_tool() -> ToolDefinition {
    ToolDefinition {
        name: TOOL_NAME.to_owned(),
        description: "Look up a city by name.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

fn tool_request(model: &mistral::CompletionModel) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .preamble(TOOL_PREAMBLE.to_owned())
        .tool(lookup_city_tool())
        .additional_params(json!({ "tool_choice": "any", "parallel_tool_calls": false }))
        .max_tokens(128)
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

/// The `mistral-correlation-id` the recorded interaction carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_headers(scenario)[0]
        .iter()
        .find(|(name, _)| name == "mistral-correlation-id")
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
) {
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
    // Mistral contracts `mistral-correlation-id`; the recorded header is the
    // premise.
    assert!(
        request_id.is_some(),
        "the recorded response must carry mistral-correlation-id"
    );
    assert_matches_recorded_token(
        response.provider_request_id.as_deref(),
        request_id,
        "request id",
    );
}

// ================================================================
// 1. raw round-trips Mistral's own type
// ================================================================

#[tokio::test]
async fn raw_round_trips_mistral_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_mistral_type";
    with_mistral_cassette_result(
        "raw_capture_matrix/raw_round_trips_mistral_type",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model.completion(request(&model)).await?;
            let raw = &response.raw;
            let typed = mistral::CompletionResponse::deserialize(raw)
                .expect("raw is Mistral's own CompletionResponse");
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
    .expect("raw_round_trips_mistral_type should replay from its cassette");

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
async fn raw_exposes_object_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_object_and_service_tier";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_mistral_cassette_result(
        "raw_capture_matrix/raw_exposes_object_and_service_tier",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model.completion(request(&model)).await?;
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_exposes_object_and_service_tier should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (_, body) = recorded_json(SCENARIO);
    let recorded_object = body["object"]
        .as_str()
        .expect("Mistral tags every completion with an object");
    let recorded_tier = body["usage"]["service_tier"]
        .as_str()
        .expect("Mistral reports usage.service_tier on the live chat wire");

    let raw = &response.raw;
    assert_eq!(raw["object"], json!(recorded_object));
    assert_eq!(raw["usage"]["service_tier"], json!(recorded_tier));
    // And the normalized view has no slot for either.
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("service_tier").is_none(),
        "the normalized usage has no tier slot: {normalized_usage}"
    );
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("object").is_none());
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_mistral_cassette_result(
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
    assert_reproduces_fixture(&response, &body, recorded_request_id(SCENARIO).as_deref());

    // The normalized fields are exactly what the response's own raw
    // re-normalizes to: capture adds a view, it never changes the mapping.
    let raw = &response.raw;
    let renormalized = mistral::CompletionResponse::deserialize(raw)
        .expect("raw is Mistral's own type")
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

// ================================================================
// 4. A forced tool call: raw round-trips and keeps the wire's spelling
// ================================================================

#[tokio::test]
async fn tool_call_raw_round_trips_and_exposes_wire_tool_call() {
    const SCENARIO: &str =
        "raw_capture_matrix/tool_call_raw_round_trips_and_exposes_wire_tool_call";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_mistral_cassette_result(
        "raw_capture_matrix/tool_call_raw_round_trips_and_exposes_wire_tool_call",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let response = model.completion(tool_request(&model)).await?;
            let raw = &response.raw;
            let typed = mistral::CompletionResponse::deserialize(raw)
                .expect("raw is Mistral's own CompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed view serialized, nothing more"
            );
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            *sink.lock().expect("observation lock") = Some(response);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("tool_call_raw_round_trips_and_exposes_wire_tool_call should replay from its cassette");

    let response = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a response");
    let (request_body, body) = recorded_json(SCENARIO);
    // Premise, from the bytes: the call was forced and the recorded turn is
    // one tool call to `lookup_city`, finishing on the wire's `tool_calls`.
    assert_eq!(request_body["tool_choice"], json!("any"));
    assert_eq!(
        request_body["tools"][0]["function"]["name"],
        json!(TOOL_NAME)
    );
    assert_eq!(body["choices"][0]["finish_reason"], json!("tool_calls"));
    let recorded_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("a forced turn carries message.tool_calls");
    assert_eq!(recorded_calls.len(), 1, "exactly one recorded call");
    let recorded_call = &recorded_calls[0];
    assert_eq!(recorded_call["function"]["name"], json!(TOOL_NAME));
    let recorded_arguments = recorded_call["function"]["arguments"]
        .as_str()
        .expect("Mistral spells tool-call arguments as a JSON string");
    let recorded_arguments: Value =
        serde_json::from_str(recorded_arguments).expect("recorded arguments parse as JSON");
    assert_eq!(recorded_arguments["city"], json!("Paris"));

    // raw keeps the wire's representation: `arguments` is a JSON string
    // (the typed view re-serializes it compactly, so it is compared parsed,
    // not byte-for-byte), and the finish reason is the wire's own spelling.
    let raw = &response.raw;
    assert_eq!(raw["choices"][0]["finish_reason"], json!("tool_calls"));
    let raw_call = &raw["choices"][0]["message"]["tool_calls"][0];
    assert_matches_recorded_token(
        raw_call["id"].as_str(),
        recorded_call["id"].as_str(),
        "tool call id",
    );
    assert_eq!(raw_call["function"]["name"], json!(TOOL_NAME));
    let raw_arguments = raw_call["function"]["arguments"]
        .as_str()
        .expect("raw keeps arguments as the wire's JSON string");
    assert_eq!(
        serde_json::from_str::<Value>(raw_arguments).expect("raw arguments parse as JSON"),
        recorded_arguments,
        "raw's stringified arguments parse to the recorded arguments"
    );

    // The normalized view maps both: an object for the arguments and
    // `ToolCalls` for the finish reason.
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    let normalized_calls = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(normalized_calls.len(), 1, "one normalized tool call");
    let normalized_call = normalized_calls[0];
    assert_eq!(normalized_call.function.name, TOOL_NAME);
    assert!(
        normalized_call.function.arguments.is_object(),
        "the normalized call carries arguments as an object: {}",
        normalized_call.function.arguments
    );
    assert_eq!(normalized_call.function.arguments, recorded_arguments);
    assert_matches_recorded_token(
        Some(normalized_call.id.as_str()),
        recorded_call["id"].as_str(),
        "normalized tool call id",
    );
}
