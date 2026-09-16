//! The chat wire's observation projection.
//!
//! Verdict, model, response id, usage and error envelope, read off a raw
//! payload before normalization discards them. This is the body of the
//! `PayloadObserver` the client layer registered per request; as
//! [`Decoder::project`](crate::wire::Decoder::project) it runs for the unary
//! reply and for every stream frame without anyone having to attach it.

use serde::Deserialize;

use crate::wire::{AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, ObservationSink};

/// A counter the provider may send as a number, a string, or `null`.
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

/// The error envelope this wire sends: `{"error": {code, message, type}}`.
#[derive(Deserialize)]
struct Envelope {
    code: Option<serde_json::Value>,
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

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

/// One object for the unary reply and each stream chunk. Every field is
/// optional: a chunk carries a delta, the last chunk (or the reply) carries
/// the usage, and `[DONE]` is not JSON at all.
#[derive(Deserialize)]
struct ChatPayload {
    id: Option<String>,
    model: Option<String>,
    usage: Option<ChatUsage>,
    #[serde(default)]
    choices: Vec<ChatChoice>,
    error: Option<Envelope>,
}

/// Project one chat-completions payload's boundary facts.
pub(super) fn project_chat(payload: &[u8], sink: &mut dyn ObservationSink) {
    let Ok(payload) = serde_json::from_slice::<ChatPayload>(payload) else {
        return;
    };
    if let Some(usage) = payload.usage {
        sink.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .and_then(|details| details.cached_tokens),
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|details| details.reasoning_tokens),
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
            finish_reason: Some(sink.scrub(&reason)),
            block_reason: None,
            detail: None,
            model: payload.model.map(|value| sink.scrub(&value)),
        },
        None => AdapterVerdict::default(),
    };
    let response_id = payload.id.map(|value| sink.scrub(&value));
    sink.provider(verdict, response_id);
    if let Some(error) = payload.error {
        let code = error.code.map(|code| match code {
            serde_json::Value::String(code) => sink.scrub(&code),
            serde_json::Value::Number(code) => code.to_string(),
            _ => "[invalid]".to_owned(),
        });
        sink.emit(AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code,
                status: error.kind.map(|value| sink.scrub(&value)),
                message: error.message.map(|value| sink.scrub(&value)),
            },
        });
    }
}
