//! Chat Completions error detection and finish vocabularies, shared by the
//! Chat decoder and its dialects.

use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::providers::openai::wire::Quirks;

/// The wire's in-band provider error envelope, when this frame is one.
///
/// Delivered with a 200 status, so it is not an HTTP failure: the frame is
/// this wire's own terminal failure and the decoder models it as an event.
/// A `null` or empty `error` can accompany valid terminal usage, and only a
/// populated `choices` array establishes content that outranks the error.
pub(crate) fn provider_error_envelope(data: &str) -> Option<ProviderError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let error = value
        .get("error")
        .filter(|error| error.is_object() || error.as_str().is_some_and(|s| !s.is_empty()))?;
    if value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
    {
        return None;
    }
    if let Some(message) = error.get("message").and_then(serde_json::Value::as_str) {
        tracing::warn!(message, "provider returned a streaming error event");
    }
    Some(ProviderError::from_provider_body(data))
}

/// The finish reasons every Chat dialect shares: OpenAI's, the legacy
/// `function_call`, and the spellings compatible servers use for the same
/// endings (`end`, `eos`, `end_turn`, `stop_sequence`, `max_tokens`,
/// Mistral's `model_length`).
pub(crate) const CHAT_FINISHES: &[(&str, FinishReason)] = &[
    ("stop", FinishReason::Stop),
    ("end", FinishReason::Stop),
    ("eos", FinishReason::Stop),
    ("end_turn", FinishReason::Stop),
    ("stop_sequence", FinishReason::Stop),
    ("length", FinishReason::Length),
    ("max_tokens", FinishReason::Length),
    ("model_length", FinishReason::Length),
    ("tool_calls", FinishReason::ToolCalls),
    ("function_call", FinishReason::ToolCalls),
    ("content_filter", FinishReason::ContentFilter),
];

/// `reason` in the dialect's documented vocabulary
/// ([`Quirks::finishes`]) and the shared one; anything else is
/// [`FinishReason::Other`], which fails the turn.
pub(crate) fn finish_reason(reason: &str, quirks: &Quirks) -> FinishReason {
    quirks
        .finishes
        .iter()
        .chain(CHAT_FINISHES)
        .find(|(name, _)| *name == reason)
        .map_or_else(
            || FinishReason::Other(reason.to_owned()),
            |(_, finish)| finish.clone(),
        )
}

/// The upstream providers' own reasons a gateway forwards beside its
/// normalized one, compared case-insensitively.
const NATIVE_FINISHES: &[(&str, FinishReason)] = &[
    ("stop", FinishReason::Stop),
    ("end_turn", FinishReason::Stop),
    ("stop_sequence", FinishReason::Stop),
    ("complete", FinishReason::Stop),
    ("completed", FinishReason::Stop),
    ("length", FinishReason::Length),
    ("max_tokens", FinishReason::Length),
    ("max_output_tokens", FinishReason::Length),
    ("model_length", FinishReason::Length),
    ("tool_calls", FinishReason::ToolCalls),
    ("function_call", FinishReason::ToolCalls),
    ("tool_use", FinishReason::ToolCalls),
    ("content_filter", FinishReason::ContentFilter),
    ("safety", FinishReason::ContentFilter),
    ("blocklist", FinishReason::ContentFilter),
    ("prohibited_content", FinishReason::ContentFilter),
    ("spii", FinishReason::ContentFilter),
];

/// A gateway's upstream-native finish reason, lowercased; an unknown one
/// is [`FinishReason::Other`].
pub(crate) fn native_finish_reason(reason: &str) -> FinishReason {
    let reason = reason.to_ascii_lowercase();
    NATIVE_FINISHES
        .iter()
        .find(|(name, _)| *name == reason)
        .map_or_else(
            || FinishReason::Other(reason.clone()),
            |(_, finish)| finish.clone(),
        )
}

#[cfg(test)]
pub(crate) mod tests;
