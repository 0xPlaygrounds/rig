//! Chat Completions and Responses metadata projected before normalization
//! can discard it: the verdict, the model, the response id, the usage and
//! any error envelope, on the unary reply and on every stream frame.

use crate::observe::{
    AdapterAttempt, AdapterContext, AdapterErrorEnvelope, AdapterEvent, AdapterUsage,
    AdapterVerdict, PayloadObserver,
};
use serde::Deserialize;

/// Attach `context` to a Chat Completions request, with the chat projector.
pub(crate) fn attach_chat<B>(
    context: AdapterContext,
    request: &mut http::Request<B>,
    route: &'static str,
) {
    context.attach(request, route);
    request
        .extensions_mut()
        .insert(PayloadObserver(chat_payload));
}

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

// Chat Completions: one object for the unary reply and each stream chunk.
// Every field is optional: a chunk carries a delta, the last chunk (or the
// reply) carries the usage, and `[DONE]` is not JSON at all.
#[derive(Deserialize)]
struct ChatUsage {
    #[serde(default, deserialize_with = "count")]
    prompt_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    completion_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<TokenDetails>,
}

#[derive(Default, Deserialize)]
struct ChatChoice {
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatPayload {
    id: Option<String>,
    model: Option<String>,
    usage: Option<ChatUsage>,
    #[serde(default)]
    choices: Vec<ChatChoice>,
    error: Option<Envelope>,
}

fn chat_payload(bytes: &[u8], attempt: &mut AdapterAttempt) {
    let Ok(payload) = serde_json::from_slice::<ChatPayload>(bytes) else {
        return;
    };
    if let Some(usage) = payload.usage {
        attempt.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|d| d.reasoning_tokens),
                tool_input_tokens: None,
            },
        });
    }
    // Every chunk names the model; only the chunk that carries the finish
    // reason is a verdict, so the model rides with it rather than on each
    // delta. The id still lands on the terminal verdict or the closure.
    let choice = payload.choices.into_iter().next().unwrap_or_default();
    let verdict = match choice.finish_reason {
        Some(reason) => AdapterVerdict {
            finish_reason: Some(attempt.text(&reason)),
            block_reason: None,
            detail: None,
            model: payload.model.map(|v| attempt.text(&v)),
        },
        None => AdapterVerdict::default(),
    };
    let response_id = payload.id.map(|v| attempt.text(&v));
    attempt.provider(verdict, response_id);
    if let Some(error) = payload.error {
        envelope(attempt, error);
    }
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

#[cfg(test)]
mod tests;
