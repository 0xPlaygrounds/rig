//! Messages metadata projected before normalization can discard it: the
//! stop reason, the model, the message id, the usage and any error
//! envelope, on the unary reply and on the stream's `message_start`,
//! `message_delta` and `error` events.

use crate::observe::{
    AdapterAttempt, AdapterContext, AdapterErrorEnvelope, AdapterEvent, AdapterUsage,
    AdapterVerdict, PayloadObserver,
};
use serde::Deserialize;

/// Attach `context` to a Messages request, with the Messages projector.
pub(crate) fn attach<B>(
    context: AdapterContext,
    request: &mut http::Request<B>,
    route: &'static str,
) {
    context.attach(request, route);
    request.extensions_mut().insert(PayloadObserver(payload));
}

fn count<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, deserialize_with = "count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    cache_read_input_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    stop_reason: Option<String>,
}

/// One object covers the unary reply and every stream event: a
/// `message_start` nests the message, a `message_delta` carries the stop
/// reason under `delta` and the cumulative output usage beside it.
#[derive(Deserialize)]
struct Payload {
    id: Option<String>,
    model: Option<String>,
    stop_reason: Option<String>,
    usage: Option<Usage>,
    message: Option<Box<Payload>>,
    delta: Option<Delta>,
    error: Option<Envelope>,
}

fn payload(bytes: &[u8], attempt: &mut AdapterAttempt) {
    let Ok(payload) = serde_json::from_slice::<Payload>(bytes) else {
        return;
    };
    let usage = payload.usage;
    let (id, model, stop_reason, nested_usage) = match payload.message {
        Some(message) => (
            message.id,
            message.model,
            message.stop_reason,
            message.usage,
        ),
        None => (payload.id, payload.model, payload.stop_reason, None),
    };
    // Anthropic reports the prompt on `message_start` and the answer's
    // running total on each `message_delta`: each is a snapshot of what it
    // knows, never a sum.
    if let Some(usage) = usage.or(nested_usage) {
        attempt.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: None,
                cached_input_tokens: usage.cache_read_input_tokens,
                reasoning_tokens: None,
                tool_input_tokens: None,
            },
        });
    }
    let stop_reason = stop_reason.or(payload.delta.and_then(|delta| delta.stop_reason));
    let verdict = AdapterVerdict {
        finish_reason: stop_reason.map(|v| attempt.text(&v)),
        block_reason: None,
        detail: None,
        model: model.map(|v| attempt.text(&v)),
    };
    let response_id = id.map(|v| attempt.text(&v));
    attempt.provider(verdict, response_id);
    if let Some(error) = payload.error {
        attempt.emit(AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code: None,
                status: error.kind.map(|v| attempt.text(&v)),
                message: error.message.map(|v| attempt.text(&v)),
            },
        });
    }
}

#[cfg(test)]
mod tests;
