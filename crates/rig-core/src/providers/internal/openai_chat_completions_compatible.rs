//! Shared pieces of the OpenAI Chat Completions wire, for the providers that
//! speak it.
//!
//! What is left here is what more than one wire's `Decoder` needs and no
//! single provider owns: the in-band provider-error frame test, the
//! `finish_reason` vocabulary, the decode-time policy for a tool call the
//! provider truncated, and the streamed tool-call fragment shape with the
//! eviction rule that tells two distinct calls apart. Frame splitting,
//! triage, assembly and telemetry all belong to the driver.

use serde::{Deserialize, Deserializer};

use super::tool_call_bridge::ToolCallSlot;
use crate::completion::{CompletionError, FinishReason};

/// The wire's in-band provider error envelope, when this frame is one.
///
/// Delivered with a 200 status, so it is not an HTTP failure: the frame is
/// this wire's own terminal failure and the decoder models it as an event.
pub(crate) fn provider_error_envelope(data: &str) -> Option<CompletionError> {
    provider_response_from_compatible_sse_data(data)
}

fn provider_response_from_compatible_sse_data(data: &str) -> Option<CompletionError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    // Treat the chunk as an error only when `error` is present AND carries a
    // payload: either an object (`{"error":{...}}`, the canonical OpenAI-compatible
    // error event) or a non-empty string (`{"error":"oops"}`, used by some
    // gateways). A `{"error":null}` or `{"error":""}` chunk — which some providers
    // send alongside the terminal usage event — must not terminate the stream.
    let error = value
        .get("error")
        .filter(|error| error.is_object() || error.as_str().is_some_and(|s| !s.is_empty()))?;
    // Only a chunk actually carrying choices is a content chunk that happens
    // to mention an error field. Mere *presence* of `choices` — including
    // `[]` and `null`, which error bodies like
    // `{"error":{"message":"rate limited"},"choices":[]}` carry — must not
    // mask the error: a masked one classifies as a normal chunk and a
    // following `[DONE]` commits a failed turn to history as a successful
    // usage-less completion (introduced in #1944; #2258 B6).
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

    Some(crate::provider_response::completion_error_from_body(data))
}

/// Map an OpenAI Chat Completions-style `finish_reason` string onto the
/// normalized vocabulary, preserving anything unrecognized verbatim.
///
/// Shared by the unary and streaming paths so both agree, and so a gateway
/// inventing a new reason surfaces it rather than reading as a natural stop.
pub(crate) fn map_openai_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        // `model_length` is Mistral's spelling for generation stopped because
        // the *context window* was exhausted rather than `max_tokens`. Both are
        // truncation, so both are `Length` — the distinction is which limit was
        // hit, not whether the turn finished. OpenRouter's own mapper already
        // folds the same spelling in (`openrouter/completion.rs`).
        "length" | "max_tokens" | "model_length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Map a gateway's upstream-native finish reason (OpenRouter's
/// `native_finish_reason`).
///
/// Its vocabulary is the union of its upstreams' — Anthropic's `end_turn`,
/// Gemini's `STOP`, the OpenAI-compatible spellings — so it is wider than
/// the normalized one and cannot be read through [`map_openai_finish_reason`].
/// Matched case-insensitively because the upstreams disagree on casing.
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

/// Deserialize OpenAI-compatible choices while tolerating only tool calls
/// that the provider cut off under an output-length finish reason.
///
/// The outer choice owns the evidence that the turn was truncated. Keeping
/// the policy here prevents an ordinary `tool_calls` turn with malformed JSON
/// arguments from being silently rewritten as though the provider had never
/// returned the call. Before dropping a candidate, a copy with only its
/// arguments repaired to `{}` must deserialize successfully; compound defects
/// such as a missing id or unknown tool type therefore remain loud.
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

/// Provider-aware form of
/// [`deserialize_choices_dropping_incomplete_tool_calls`].
///
/// Most compatible providers have one normalized `finish_reason`. Gateways
/// such as OpenRouter can expose a second upstream-native reason with explicit
/// precedence rules; their response type supplies that effective-length
/// predicate here while reusing the same compound-safe repair/drop policy.
pub(crate) fn deserialize_choices_dropping_incomplete_tool_calls_when<'de, D, T, F>(
    deserializer: D,
    is_output_length: F,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
    F: Fn(&serde_json::Value) -> bool,
{
    fn incomplete_arguments(call: &serde_json::Value) -> bool {
        call.get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|raw| {
                raw.trim().is_empty() || crate::json_utils::parse_tool_arguments(raw).is_err()
            })
    }

    fn repair_incomplete_arguments(choice: &mut serde_json::Value) -> bool {
        let Some(tool_calls) = choice
            .get_mut("message")
            .and_then(|message| message.get_mut("tool_calls"))
            .and_then(serde_json::Value::as_array_mut)
        else {
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
        let Some(tool_calls) = choice
            .get_mut("message")
            .and_then(|message| message.get_mut("tool_calls"))
            .and_then(serde_json::Value::as_array_mut)
        else {
            return 0;
        };

        let before = tool_calls.len();
        tool_calls.retain(|call| !incomplete_arguments(call));
        before - tool_calls.len()
    }

    Vec::<serde_json::Value>::deserialize(deserializer)?
        .into_iter()
        .map(|mut choice| {
            if is_output_length(&choice) {
                let mut repaired = choice.clone();
                if repair_incomplete_arguments(&mut repaired)
                    && serde_json::from_value::<T>(repaired).is_ok()
                {
                    let dropped = drop_incomplete_arguments(&mut choice);
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

/// A chunk's terminal reason, as reported by an OpenAI-compatible provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompatibleFinishReason {
    /// The chunk reported a terminal reason, normalized.
    Reported(FinishReason),
    /// The chunk carried no `finish_reason` field.
    Absent,
}

impl CompatibleFinishReason {
    /// Whether the provider explicitly ended the turn to call tools.
    pub(crate) fn is_tool_calls(&self) -> bool {
        matches!(self, Self::Reported(FinishReason::ToolCalls))
    }

    /// The normalized reason, when the provider reported one.
    pub(crate) fn reported(&self) -> Option<FinishReason> {
        match self {
            Self::Reported(reason) => Some(reason.clone()),
            Self::Absent => None,
        }
    }
}

/// The terminal state a chat-completions stream reached, from which a wire
/// builds its own provider-native terminal record.
#[derive(Debug, Clone)]
pub(crate) struct CompatibleTerminal<U> {
    /// Provider-native usage payload from the terminal event; `None` when the
    /// stream never carried one.
    pub(crate) usage: Option<U>,
    /// Normalized finish reason, when the stream reported one.
    pub(crate) finish_reason: Option<FinishReason>,
    /// Provider-assigned response identifier, when emitted.
    pub(crate) response_id: Option<String>,
    /// Provider-reported model identifier, when emitted.
    pub(crate) model: Option<String>,
    /// Per-chunk primary-choice log probabilities, deep-merged in arrival
    /// order so token arrays retain the exact streamed sequence.
    pub(crate) logprobs: Option<crate::message::AdditionalParams>,
    /// Provider-specific top-level chunk metadata, deep-merged in arrival
    /// order so the raw terminal record does not lose additive wire fields.
    pub(crate) additional_params: Option<crate::message::AdditionalParams>,
}

#[derive(Debug, Clone)]
pub(crate) struct CompatibleToolCallChunk {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

impl CompatibleToolCallChunk {
    fn has_nonempty_name(&self) -> bool {
        self.name.as_ref().is_some_and(|name| !name.is_empty())
    }

    fn has_nonempty_arguments(&self) -> bool {
        self.arguments
            .as_ref()
            .is_some_and(|arguments| !arguments.is_empty())
    }

    fn starts_new_tool_call(&self) -> bool {
        self.has_nonempty_name()
            && self
                .arguments
                .as_ref()
                .is_none_or(std::string::String::is_empty)
    }

    /// Whether this one fragment carries a whole call — the shape
    /// llama.cpp-based servers emit.
    pub(crate) fn is_complete_single_chunk(&self) -> bool {
        self.has_nonempty_name() && self.has_nonempty_arguments()
    }
}

pub(crate) fn should_evict_distinct_named_tool_call(
    existing: &ToolCallSlot,
    incoming: &CompatibleToolCallChunk,
) -> bool {
    if let Some(new_id) = &incoming.id
        && !new_id.is_empty()
        && let Some(new_name) = &incoming.name
        && incoming.has_nonempty_name()
        && !existing.id.is_empty()
        && existing.id != *new_id
        && !existing.name.is_empty()
    {
        return existing.name != *new_name || incoming.starts_new_tool_call();
    }

    false
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests;
