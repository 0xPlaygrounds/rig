//! Shared Chat Completions error detection, finish-reason normalization, and
//! truncated tool-call handling for decoders and typed response views.

use serde::{Deserialize, Deserializer};

use crate::completion::FinishReason;
use crate::error::ProviderError;

/// The wire's in-band provider error envelope, when this frame is one.
///
/// Delivered with a 200 status, so it is not an HTTP failure: the frame is
/// this wire's own terminal failure and the decoder models it as an event.
pub(crate) fn provider_error_envelope(data: &str) -> Option<ProviderError> {
    provider_response_from_compatible_sse_data(data)
}

fn provider_response_from_compatible_sse_data(data: &str) -> Option<ProviderError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    // Null or empty-string error fields can accompany valid terminal usage.
    let error = value
        .get("error")
        .filter(|error| error.is_object() || error.as_str().is_some_and(|s| !s.is_empty()))?;
    // Only populated choices establish content; empty choices must not mask
    // an error and let a later terminator commit a failed turn as successful.
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

    Some(crate::error::ProviderError::from_provider_body(data))
}

/// Map an OpenAI Chat Completions-style `finish_reason` string onto the
/// normalized vocabulary, preserving anything unrecognized verbatim.
///
/// Shared by the unary and streaming paths so both agree, and so a gateway
/// inventing a new reason surfaces it rather than reading as a natural stop.
pub(crate) fn map_openai_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        // Context-window exhaustion and output-budget exhaustion both truncate.
        "length" | "max_tokens" | "model_length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Normalize a gateway's upstream-native finish reason case-insensitively.
/// Unknown values are returned lowercased as [`FinishReason::Other`].
pub(crate) fn map_native_finish_reason(reason: &str) -> FinishReason {
    match reason.to_ascii_lowercase().as_str() {
        "stop" | "end_turn" | "stop_sequence" | "complete" | "completed" => FinishReason::Stop,
        "length" | "max_tokens" | "max_output_tokens" | "model_length" => FinishReason::Length,
        "tool_calls" | "function_call" | "tool_use" => FinishReason::ToolCalls,
        "content_filter" | "safety" | "blocklist" | "prohibited_content" | "spii" => {
            FinishReason::ContentFilter
        }
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Deserialize choices, dropping incomplete tool calls only for length finishes.
/// Dropping requires a copy with arguments repaired to `{}` to deserialize as `T`.
/// Other defects remain deserialization errors.
pub(crate) fn deserialize_choices_dropping_incomplete_tool_calls<'de, D, T>(
    deserializer: D,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    deserialize_choices_dropping_incomplete_tool_calls_when(deserializer, |choice| {
        choice
            .get("finish_reason")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|reason| matches!(map_openai_finish_reason(reason), FinishReason::Length))
    })
}

/// Deserialize choices using `is_output_length` to select truncated turns.
/// Applies [`deserialize_choices_dropping_incomplete_tool_calls`]'s repair check
/// before dropping incomplete calls.
pub(crate) fn deserialize_choices_dropping_incomplete_tool_calls_when<'de, D, T, F>(
    deserializer: D,
    is_output_length: F,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
    F: Fn(&serde_json::Value) -> bool,
{
    Vec::<serde_json::Value>::deserialize(deserializer)?
        .into_iter()
        .map(|mut choice| {
            if is_output_length(&choice) {
                let dropped = drop_tool_calls_cut_by_budget::<T>(&mut choice);
                if dropped > 0 {
                    tracing::debug!(
                        dropped,
                        "dropping tool calls incomplete under an output-length finish reason"
                    );
                }
            }

            serde_json::from_value(choice).map_err(serde::de::Error::custom)
        })
        .collect()
}

/// Drop incomplete argument strings from a choice and return the number removed.
/// The caller must establish an output-length finish. Returns zero unchanged if
/// repairing the arguments to `{}` does not make the choice deserialize as `T`.
pub(crate) fn drop_tool_calls_cut_by_budget<T>(choice: &mut serde_json::Value) -> usize
where
    T: serde::de::DeserializeOwned,
{
    let mut probe = choice.clone();
    if !repair_incomplete_arguments(&mut probe) || serde_json::from_value::<T>(probe).is_err() {
        return 0;
    }
    drop_incomplete_arguments(choice)
}

/// Whether arguments are an empty or unparseable string; nonstrings return false.
/// Empty strings count as incomplete because truncation can precede the first token.
fn incomplete_arguments(call: &serde_json::Value) -> bool {
    call.get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|raw| {
            raw.trim().is_empty() || crate::json_utils::parse_tool_arguments(raw).is_err()
        })
}

fn message_tool_calls_mut(choice: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    choice
        .get_mut("message")
        .and_then(|message| message.get_mut("tool_calls"))
        .and_then(serde_json::Value::as_array_mut)
}

fn repair_incomplete_arguments(choice: &mut serde_json::Value) -> bool {
    let Some(tool_calls) = message_tool_calls_mut(choice) else {
        return false;
    };

    let mut repaired = false;
    for call in tool_calls {
        if !incomplete_arguments(call) {
            continue;
        }
        let Some(arguments) = call
            .get_mut("function")
            .and_then(|function| function.get_mut("arguments"))
        else {
            continue;
        };
        *arguments = serde_json::Value::String("{}".to_owned());
        repaired = true;
    }
    repaired
}

fn drop_incomplete_arguments(choice: &mut serde_json::Value) -> usize {
    let Some(tool_calls) = message_tool_calls_mut(choice) else {
        return 0;
    };

    let before = tool_calls.len();
    tool_calls.retain(|call| !incomplete_arguments(call));
    before - tool_calls.len()
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests;
