//! GenerateContent metadata projected before normalization can discard it.

use crate::observe::{
    AdapterAttempt, AdapterContext, AdapterErrorEnvelope, AdapterEvent, AdapterUsage,
    AdapterVerdict, PayloadObserver,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Projection {
    #[serde(rename = "usageMetadata")]
    usage: Option<UsageProjection>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Metadata {
    #[serde(default)]
    candidates: Vec<Candidate>,
    prompt_feedback: Option<Feedback>,
    model_version: Option<String>,
    response_id: Option<String>,
    error: Option<Envelope>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    finish_reason: Option<String>,
    finish_message: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Feedback {
    block_reason: Option<String>,
}

#[derive(Deserialize)]
struct Envelope {
    code: Option<serde_json::Value>,
    status: Option<String>,
    message: Option<String>,
}

// Ignore all unrelated response fields rather than allocating another tree
// containing the completion text, tools, signatures and media.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageProjection {
    #[serde(default, deserialize_with = "count")]
    prompt_token_count: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    candidates_token_count: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    total_token_count: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    cached_content_token_count: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    thoughts_token_count: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    tool_use_prompt_token_count: Option<u64>,
}

fn count<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

pub(super) fn attach<B>(
    context: AdapterContext,
    request: &mut http::Request<B>,
    route: &'static str,
) {
    context.attach(request, route);
    request.extensions_mut().insert(PayloadObserver(payload));
}

fn payload(bytes: &[u8], attempt: &mut AdapterAttempt) {
    // The observation projection must not inherit native response defaults:
    // omitted prompt/total counts in UsageMetadata otherwise become zero.
    // Parsing failure has no effect on the provider's authoritative decoder.
    if let Ok(Projection { usage: Some(usage) }) = serde_json::from_slice::<Projection>(bytes) {
        attempt.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count,
                total_tokens: usage.total_token_count,
                cached_input_tokens: usage.cached_content_token_count,
                reasoning_tokens: usage.thoughts_token_count,
                tool_input_tokens: usage.tool_use_prompt_token_count,
            },
        });
    }
    // Project metadata independently so malformed candidate fields cannot
    // erase an otherwise valid usage report from a rejected response.
    let Ok(metadata) = serde_json::from_slice::<Metadata>(bytes) else {
        return;
    };
    let candidate = metadata.candidates.into_iter().next().unwrap_or_default();
    let scrub = |value: String| attempt.text(&value);
    let verdict = AdapterVerdict {
        finish_reason: candidate.finish_reason.map(scrub),
        block_reason: metadata
            .prompt_feedback
            .and_then(|f| f.block_reason)
            .map(scrub),
        detail: candidate.finish_message.map(scrub),
        model: metadata.model_version.map(scrub),
    };
    let response_id = metadata.response_id.map(scrub);
    attempt.provider(verdict, response_id);
    if let Some(error) = metadata.error {
        let code = error.code.map(|code| match code {
            serde_json::Value::String(code) => attempt.text(&code),
            serde_json::Value::Number(code) => code.to_string(),
            _ => "[invalid]".to_owned(),
        });
        attempt.emit(AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code,
                status: error.status.map(|value| attempt.text(&value)),
                message: error.message.map(|value| attempt.text(&value)),
            },
        });
    }
}
