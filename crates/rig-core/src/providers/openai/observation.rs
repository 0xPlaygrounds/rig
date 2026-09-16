//! Responses metadata projected before normalization can discard it: the
//! verdict, the model, the response id, the usage and any error envelope, on
//! the unary reply and on every stream frame.
//!
//! The Chat Completions projector lives beside its wire, in
//! [`wire::observation`](super::wire).

use crate::observe::{
    AdapterAttempt, AdapterContext, AdapterErrorEnvelope, AdapterEvent, AdapterUsage,
    AdapterVerdict, PayloadObserver,
};
use serde::Deserialize;

/// Attach `context` to a Responses request, with the Responses projector.
pub(crate) fn attach_responses<B>(
    context: AdapterContext,
    request: &mut http::Request<B>,
    route: &'static str,
) {
    context.attach(request, route);
    request
        .extensions_mut()
        .insert(PayloadObserver(responses_payload));
}

fn count<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

#[derive(Default, Deserialize)]
struct TokenDetails {
    #[serde(default, deserialize_with = "count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    reasoning_tokens: Option<u64>,
}

/// The error envelope both wires share: `{"error": {code, message, type}}`;
/// the Responses stream's `error` event carries the same fields at the top.
#[derive(Deserialize)]
struct Envelope {
    code: Option<serde_json::Value>,
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

fn envelope(attempt: &mut AdapterAttempt, error: Envelope) {
    let code = error.code.map(|code| match code {
        serde_json::Value::String(code) => attempt.text(&code),
        serde_json::Value::Number(code) => code.to_string(),
        _ => "[invalid]".to_owned(),
    });
    attempt.emit(AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code,
            status: error.kind.map(|value| attempt.text(&value)),
            message: error.message.map(|value| attempt.text(&value)),
        },
    });
}

// Responses: the unary reply is the response object; a stream event wraps
// the object under `response` (`response.created`, `.completed`, `.failed`,
// `.incomplete`) or, for `error`, carries the envelope's fields itself.
#[derive(Deserialize)]
struct ResponsesUsage {
    #[serde(default, deserialize_with = "count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<TokenDetails>,
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseObject {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<ResponsesUsage>,
    error: Option<Envelope>,
}

#[derive(Deserialize)]
struct ResponsesPayload {
    #[serde(rename = "type")]
    kind: Option<String>,
    response: Option<ResponseObject>,
    // The unary reply's own fields, and the stream `error` event's.
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<ResponsesUsage>,
    error: Option<Envelope>,
    code: Option<serde_json::Value>,
    message: Option<String>,
}

fn responses_payload(bytes: &[u8], attempt: &mut AdapterAttempt) {
    let Ok(payload) = serde_json::from_slice::<ResponsesPayload>(bytes) else {
        return;
    };
    if payload.kind.as_deref() == Some("error") {
        // The event carries its envelope either nested under `error` or as
        // its own top-level fields; the nested form names the error type.
        let error = payload.error.unwrap_or(Envelope {
            code: payload.code,
            kind: None,
            message: payload.message,
        });
        envelope(attempt, error);
        return;
    }
    let object = payload.response.unwrap_or(ResponseObject {
        id: payload.id,
        model: payload.model,
        status: payload.status,
        incomplete_details: payload.incomplete_details,
        usage: payload.usage,
        error: payload.error,
    });
    if let Some(usage) = object.usage {
        attempt.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage.input_tokens_details.and_then(|d| d.cached_tokens),
                reasoning_tokens: usage.output_tokens_details.and_then(|d| d.reasoning_tokens),
                tool_input_tokens: None,
            },
        });
    }
    // `status` is the provider's verdict; `in_progress` on a stream's
    // opening event is not one yet, so it is left out of the projection.
    let finish_reason = object
        .status
        .filter(|status| status != "in_progress" && status != "queued");
    let verdict = AdapterVerdict {
        finish_reason: finish_reason.map(|v| attempt.text(&v)),
        block_reason: None,
        detail: object
            .incomplete_details
            .and_then(|details| details.reason)
            .map(|v| attempt.text(&v)),
        model: object.model.map(|v| attempt.text(&v)),
    };
    let response_id = object.id.map(|v| attempt.text(&v));
    attempt.provider(verdict, response_id);
    if let Some(error) = object.error {
        envelope(attempt, error);
    }
}
