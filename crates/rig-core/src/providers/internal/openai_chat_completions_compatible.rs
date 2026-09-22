//! Shared pieces of the OpenAI Chat Completions wire, for the providers that
//! speak it.
//!
//! What is left here is what the chat decoder and the dialects' typed reply
//! views both need: the in-band provider-error frame test, the
//! `finish_reason` vocabulary, and the policy for a tool call the provider
//! truncated under an output-length finish reason. Frame splitting, triage,
//! assembly and telemetry all belong to the driver.

use serde::{Deserialize, Deserializer};

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

/// Drop from one raw output-length choice every tool call whose `arguments`
/// string the budget cut short, returning how many were dropped.
///
/// Before anything is dropped, a copy with those arguments stubbed to `{}`
/// must decode as `T`: a choice that is also broken elsewhere (a call
/// missing its id, an unknown tool type) keeps its original error rather
/// than having the evidence deleted underneath it. Shared by the typed reply
/// views (which apply it on decode) and the chat decoder (which applies it
/// to the raw body before classification), so the two cannot disagree about
/// which calls a truncated turn still carries.
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

/// Whether a raw tool call's `arguments` string is unusable as tool input.
///
/// Empty counts as unusable alongside unparseable: `parse_tool_arguments`
/// maps an empty string onto `{}` so a genuine zero-argument tool works, and
/// a call cut before its first argument token is exactly what that
/// normalization would disguise. Arguments a dialect sent as a raw JSON value
/// rather than a string are never unusable: there is no half-written string.
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
