//! Raw provider response capture on Mistral's blocking chat-completions path.
//!
//! **The feature.** Every blocking completion attaches the provider's reply
//! document — the bytes Mistral sent, parsed as JSON — onto the normalized
//! [`rig::completion::CompletionResponse::raw`], where Mistral's own
//! [`mistral::CompletionResponse`] reads it back. Capture is always on: there is
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
//! | 1 | `raw_round_trips_mistral_type` | typed round trip | `raw` deserializes into `mistral::CompletionResponse`, whose provider-native fields are the normalized response's | recorded |
//! | 2 | `raw_exposes_object_and_service_tier` | provider-only field | `raw.object` and `raw.usage.service_tier` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes (including the `mistral-correlation-id` header), and the typed view of its own `raw` agrees with it field for field | recorded |
//! | 4 | `tool_call_raw_round_trips_and_exposes_wire_tool_call` | forced tool call | a `tool_choice: any` turn's `raw` round-trips into `mistral::CompletionResponse`; `raw.choices[0].message.tool_calls[0].function.arguments` is a JSON *string* that parses to the fixture's arguments while the normalized call carries an object; `finish_reason()` is `ToolCalls` while `raw.choices[0].finish_reason` is the wire's `"tool_calls"` | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the tier out of the recorded body
//! rather than trusting the string the typed view reports, cell 3 checks
//! the normalized fields against the recorded body and headers before
//! comparing them with the typed view of its own `raw`, and cell 4 reads the
//! call (id, name, stringified arguments) and the `"tool_calls"` finish out
//! of the recorded body, so a recording that stopped carrying a usage block,
//! a finish reason, the correlation-id header, or a tool call fails loudly
//! instead of covering nothing.
//!
//! The format-wide half of cell 3 — "the normalized fields are the recorded
//! reply's" — is [`chat::assert_reproduces_body`], shared by every
//! chat-completions dialect. The provider-native half stays here, because
//! Mistral's decoder maps from [`mistral::CompletionResponse`] rather than
//! the shared chat-completions type the format helper compares against.

use rig::completion::{CompletionModel, CompletionRequest, FinishReason, ToolDefinition};
use rig::message::AssistantContent;
use rig::providers::mistral;
use serde::Deserialize;
use serde_json::{Value, json};

use super::DEFAULT_MODEL;
use super::support::with_mistral_cassette_result;
use crate::cassettes::{recorded_json_turn, recorded_response_header};
use crate::raw_capture::{
    assert_contracted_request_id, assert_normalized_lacks, capture_completion, chat,
};
use crate::support::{
    Observed, assert_matches_recorded_token, assistant_text, normalized_without_raw,
};

const PROVIDER: &str = "mistral";
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "mistral-correlation-id";
/// The forced-call request shape the tool-lifecycle matrix uses: a preamble
/// that forbids prose, `tool_choice: any`, and a prompt naming the one call.
const TOOL_PREAMBLE: &str =
    "Follow the user's tool-call instruction exactly. Do not answer in prose.";
const TOOL_PROMPT: &str = "Call lookup_city exactly once with city Paris.";
const TOOL_NAME: &str = "lookup_city";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
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

fn tool_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .preamble(TOOL_PREAMBLE.to_owned())
        .tool(lookup_city_tool())
        .additional_params(json!({ "tool_choice": "any", "parallel_tool_calls": false }))
        .max_tokens(128)
        .build()
}

/// The `mistral-correlation-id` the recorded interaction carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_header(PROVIDER, scenario, 0, REQUEST_ID_HEADER)
}

/// The assistant text of a provider-native choice, as Mistral spells it.
fn provider_text(choice: &mistral::Choice) -> &str {
    match &choice.message {
        mistral::Message::Assistant { content, .. } => content.as_str(),
        other => panic!("a completion choice carries an assistant message, got {other:?}"),
    }
}

// ================================================================
// 1. raw round-trips Mistral's own type
// ================================================================

#[tokio::test]
async fn raw_round_trips_mistral_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_mistral_type";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_capture_matrix/raw_round_trips_mistral_type",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, observed.clone()),
    )
    .await
    .expect("raw_round_trips_mistral_type should replay from its cassette");

    let response = observed.take();
    let (_, response_body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );

    // `raw` is the reply document, so Mistral's own response type reads it
    // back — the documented escape hatch — and its provider-native fields
    // are the normalized response's fields.
    let typed = mistral::CompletionResponse::deserialize(&response.raw)
        .expect("raw is Mistral's own CompletionResponse");
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    let usage = typed.usage.as_ref().expect("Mistral reports usage");
    assert_eq!(
        Some(usage.total_tokens as u64),
        response.usage.total_tokens,
        "one reply, one token count"
    );
}

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_object_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_object_and_service_tier";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_capture_matrix/raw_exposes_object_and_service_tier",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, observed.clone()),
    )
    .await
    .expect("raw_exposes_object_and_service_tier should replay from its cassette");

    let response = observed.take();
    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_object = body["object"]
        .as_str()
        .expect("Mistral tags every completion with an object");
    let recorded_tier = body["usage"]["service_tier"]
        .as_str()
        .expect("Mistral reports usage.service_tier on the live chat wire");

    let raw = &response.raw;
    assert_eq!(raw["object"], json!(recorded_object));
    assert_eq!(raw["usage"]["service_tier"], json!(recorded_tier));
    // And the normalized view has no slot for either — checked against the
    // response with its capture cleared, so `raw` cannot satisfy the claim it
    // is the counterexample to.
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert_normalized_lacks(&normalized_usage, &["service_tier"]);
    assert_normalized_lacks(&normalized_without_raw(response), &["object"]);
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, observed.clone()),
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");

    let response = observed.take();
    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // Mistral contracts `mistral-correlation-id`; the recorded header is the
    // premise.
    assert_contracted_request_id(
        response.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );

    // One reply, two views. `raw` is the document Mistral sent; its own type
    // reads that back, and every provider-native field it exposes must be
    // the normalized response's. There is one decoder and one mapping now,
    // so this pins that mapping instead of comparing it with a second one —
    // which is why the native view here is Mistral's own type rather than
    // the shared chat-completions one.
    let typed =
        mistral::CompletionResponse::deserialize(&response.raw).expect("raw is Mistral's own type");
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    assert_eq!(
        typed.choices.len(),
        1,
        "the recorded turn has one candidate"
    );
    let typed_choice = &typed.choices[0];
    assert_eq!(
        Some(typed_choice.finish_reason.as_str()),
        body["choices"][0]["finish_reason"].as_str(),
        "the typed view keeps the wire's own finish spelling"
    );
    assert_eq!(
        response.finish_reason(),
        Some(chat::recorded_chat_finish_reason(&response.raw)),
        "and the normalized reason is that same spelling, mapped"
    );
    assert_eq!(
        provider_text(typed_choice),
        assistant_text(&response.choice)
    );
    let typed_usage = typed.usage.as_ref().expect("Mistral reports usage");
    assert_eq!(
        Some(typed_usage.prompt_tokens as u64),
        response.usage.input_tokens
    );
    assert_eq!(
        Some(typed_usage.completion_tokens as u64),
        response.usage.output_tokens
    );
    assert_eq!(
        Some(typed_usage.total_tokens as u64),
        response.usage.total_tokens
    );
}

// ================================================================
// 4. A forced tool call: raw round-trips and keeps the wire's spelling
// ================================================================

#[tokio::test]
async fn tool_call_raw_round_trips_and_exposes_wire_tool_call() {
    const SCENARIO: &str =
        "raw_capture_matrix/tool_call_raw_round_trips_and_exposes_wire_tool_call";
    let observed = Observed::default();
    with_mistral_cassette_result(
        "raw_capture_matrix/tool_call_raw_round_trips_and_exposes_wire_tool_call",
        |client| {
            capture_completion(
                client.completion(DEFAULT_MODEL),
                tool_request,
                observed.clone(),
            )
        },
    )
    .await
    .expect("tool_call_raw_round_trips_and_exposes_wire_tool_call should replay from its cassette");

    let response = observed.take();
    let typed = mistral::CompletionResponse::deserialize(&response.raw)
        .expect("raw is Mistral's own CompletionResponse");
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());

    let (request_body, body) = recorded_json_turn(PROVIDER, SCENARIO);
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
        normalized_call.id.explicit(),
        recorded_call["id"].as_str(),
        "normalized tool call id",
    );
}
