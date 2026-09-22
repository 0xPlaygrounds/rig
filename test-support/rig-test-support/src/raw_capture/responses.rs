//! The OpenAI Responses format contract.
//!
//! A Responses reply is an envelope with a `status` and a list of `output`
//! items, so its identity, finish reason and accounting live in different
//! places than the chat-completions wire's — a separate contract, not a
//! parameterization of [`super::chat`]'s. What stays at the call site: the
//! envelope fields a particular provider adds (`service_tier`,
//! `metadata.system_fingerprint`), and its transport id contract.

use rig_core::completion::{CompletionResponse, FinishReason, Usage};
use rig_core::providers::openai::responses_api;
use rig_core::streaming::StreamFinal;
use serde::Deserialize as _;
use serde_json::Value;

use crate::support::{assert_matches_recorded_token, assistant_text};

/// The provider-native terminal record a Responses stream's
/// [`StreamFinal::raw`] holds.
pub type Terminal = responses_api::streaming::StreamingCompletionResponse;

/// The terminal record `raw` holds, read back and required to be exactly the
/// decoder's record serialized.
///
/// A streamed reply is many events and no single one of them is the answer,
/// so what rides along is the record the decoder assembled from the
/// `response.completed` event — which makes an exact typed round trip the
/// right claim here, unlike the blocking path where `raw` is the reply
/// document. The returned typed record is the cell's handle on whatever its
/// dialect keeps beside the shared fields.
pub fn assert_terminal_round_trips(terminal: &StreamFinal) -> Terminal {
    let raw = &terminal.raw;
    let typed =
        Terminal::deserialize(raw).expect("raw is the Responses terminal record, serialized");
    assert_eq!(
        serde_json::to_value(&typed).expect("typed serializes"),
        *raw,
        "the captured value is the typed terminal serialized, nothing more"
    );
    assert_eq!(typed.response_id, terminal.response_id, "response id");
    assert_eq!(typed.message_id, terminal.message_id, "message id");
    assert_eq!(typed.model, terminal.model, "model");
    assert_eq!(
        Usage::from(&typed),
        terminal.usage,
        "the normalized usage is the terminal event's accounting, normalized"
    );
    typed
}

/// The recorded `output` message item's id and its concatenated output text.
pub fn recorded_message(body: &Value) -> (&str, String) {
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

/// The finish reason a recorded Responses envelope's `status` reports.
///
/// Mapped by hand from the wire word, so the expectation is independent of
/// the decoder that produced the normalized reason it is compared against.
pub fn recorded_finish_reason(body: &Value) -> FinishReason {
    match body["status"].as_str() {
        Some("completed") => FinishReason::Stop,
        other => panic!("recorded turn should have completed, got {other:?}"),
    }
}

/// The normalized response's fields, checked against the recorded reply bytes
/// that produced them.
///
/// The transport request id is not here: whether a dialect contracts an id
/// header or sends none are different contracts, so each cell states its own
/// with [`assert_contracted_request_id`](super::assert_contracted_request_id)
/// or [`assert_no_request_id`](super::assert_no_request_id).
pub fn assert_reproduces_body(
    response: &CompletionResponse,
    provider: &str,
    body: &Value,
    context: &str,
) {
    assert_eq!(response.provider, provider, "{context}: provider");
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
            body["usage"]["input_tokens"].as_u64(),
            body["usage"]["output_tokens"].as_u64(),
            body["usage"]["total_tokens"].as_u64(),
        ),
        "{context}: usage"
    );
    assert_eq!(
        assistant_text(&response.choice),
        text,
        "{context}: choice text"
    );
}

/// The provider-native view of the captured reply, beside the normalized
/// fields the decoder mapped from it.
///
/// `native` is read out of the response's own `raw` by the cell, so this pins
/// the mapping one decoder performed rather than a second copy of it. The
/// `provider_request_id` claim is part of the contract rather than an aside:
/// this type has a slot for it and a reply *document* never fills it, because
/// the id arrives on a header.
pub fn assert_native_matches_normalized(
    response: &CompletionResponse,
    native: &responses_api::CompletionResponse,
    context: &str,
) {
    assert_eq!(
        response.response_id.as_deref(),
        Some(native.id.as_str()),
        "{context}: native response id"
    );
    assert_eq!(
        response.model.as_deref(),
        Some(native.model.as_str()),
        "{context}: native model"
    );
    assert_eq!(
        native.status,
        responses_api::ResponseStatus::Completed,
        "{context}: native status"
    );
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::Stop),
        "{context}: the normalized reason is that status"
    );
    let native_usage = native.usage.as_ref().expect("the reply reports usage");
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            Some(native_usage.input_tokens),
            Some(native_usage.output_tokens),
            Some(native_usage.total_tokens),
        ),
        "{context}: the normalized counters are the native ones"
    );
    assert_eq!(
        native.provider_request_id, None,
        "{context}: a reply document has no slot for a response header"
    );
    let message = native
        .output
        .iter()
        .find_map(|item| match item {
            responses_api::Output::Message(message) => Some(message),
            _ => None,
        })
        .expect("the reply carries a message item");
    assert_eq!(
        response.message_id.as_deref(),
        Some(message.id.as_str()),
        "{context}: native message id"
    );
}
