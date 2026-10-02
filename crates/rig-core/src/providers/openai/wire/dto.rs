//! Chat Completions reply shapes for unary messages and streamed deltas,
//! and the merge that assembles a streamed message from its deltas.

use serde::{Deserialize, Serialize};

use crate::json_utils;
use crate::providers::openai::completion::Usage;

/// A streamed tool-call fragment's function half.
#[derive(Default, Deserialize, Debug, Clone)]
pub(crate) struct StreamingFunction {
    pub(crate) name: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_utils::deserialize_json_string_or_value"
    )]
    pub(crate) arguments: Option<String>,
}

/// One streamed tool-call fragment.
#[derive(Deserialize, Debug, Clone)]
pub(crate) struct StreamingToolCall {
    // Optional in several compatible dialects (e.g. Mistral); missing means
    // a single in-flight tool call.
    #[serde(default)]
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) function: StreamingFunction,
}

impl StreamingToolCall {
    fn has_nonempty_name(&self) -> bool {
        self.function
            .name
            .as_ref()
            .is_some_and(|name| !name.is_empty())
    }

    fn starts_new_tool_call(&self) -> bool {
        self.has_nonempty_name()
            && self
                .function
                .arguments
                .as_ref()
                .is_none_or(String::is_empty)
    }

    /// Whether this one fragment carries a whole call: the shape
    /// llama.cpp-based servers emit.
    pub(crate) fn is_complete_single_chunk(&self) -> bool {
        self.has_nonempty_name()
            && self
                .function
                .arguments
                .as_ref()
                .is_some_and(|arguments| !arguments.is_empty())
    }

    /// Whether this fragment belongs to a different call than the one open
    /// at its index. Some gateways stream two distinct calls under one
    /// `index`: a new id plus either a different name or an argument-less
    /// opening fragment is a second call; anything else continues the call
    /// already open.
    pub(crate) fn evicts(&self, existing_id: &str, existing_name: &str) -> bool {
        if let Some(new_id) = &self.id
            && !new_id.is_empty()
            && let Some(new_name) = &self.function.name
            && self.has_nonempty_name()
            && !existing_id.is_empty()
            && existing_id != *new_id
            && !existing_name.is_empty()
        {
            return existing_name != *new_name || self.starts_new_tool_call();
        }

        false
    }
}

/// The keys whose string fragments concatenate when a provider streams them:
/// text and reasoning, a call's arguments, audio's transcript and data, a
/// reasoning detail's text and summary. Every other string is an identifier,
/// a tag or a signature a later fragment restates.
const FRAGMENT_KEYS: [&str; 12] = [
    "content",
    "refusal",
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "thinking",
    "tool_plan",
    "transcript",
    "data",
    "arguments",
    "text",
    "summary",
];

/// Merge one streamed fragment of a provider object into what arrived so
/// far: fragment strings ([`FRAGMENT_KEYS`]) append, arrays extend, objects
/// merge key by key, and anything else replaces. A `null` or empty string
/// never erases a value, and a literal `null` argument placeholder gives
/// way to the first real fragment.
pub(crate) fn merge_fields(
    target: &mut serde_json::Map<String, serde_json::Value>,
    delta: &serde_json::Map<String, serde_json::Value>,
) {
    use serde_json::Value;
    for (key, value) in delta {
        let fragment = FRAGMENT_KEYS.contains(&key.as_str());
        match (target.get_mut(key), value) {
            (Some(_), Value::Null) => {}
            (Some(Value::String(existing)), Value::String(more)) if fragment => {
                if existing.trim() == "null" && !more.trim().is_empty() {
                    existing.clear();
                }
                existing.push_str(more);
            }
            (Some(Value::String(existing)), Value::String(more))
                if more.is_empty() && !existing.is_empty() => {}
            (Some(Value::Array(existing)), Value::Array(more)) => {
                existing.extend(more.iter().cloned());
            }
            (Some(Value::Object(existing)), Value::Object(more)) => merge_fields(existing, more),
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Open the block `slot` names, once: its writer index.
pub(crate) fn open_once(
    slot: &mut Option<usize>,
    block: crate::operation::Block,
    out: &mut crate::wire::Out<'_, crate::operation::Completion>,
) -> Result<usize, crate::error::ProviderError> {
    if let Some(index) = *slot {
        return Ok(index);
    }
    let index = out.fresh_index();
    out.open(index, block, serde_json::Value::Null)?;
    *slot = Some(index);
    Ok(index)
}

/// The text a message or delta carries: its `content` string, or the text
/// and refusal parts of a content array, falling back to the sibling
/// `refusal` when there is no content.
pub(crate) fn delta_text(delta: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    use serde_json::Value;
    let content = match delta.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| {
                matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("text" | "refusal")
                )
            })
            .filter_map(|part| {
                part.get("text")
                    .or_else(|| part.get("refusal"))
                    .and_then(Value::as_str)
            })
            .collect(),
        _ => String::new(),
    };
    if !content.is_empty() {
        return Some(content);
    }
    delta
        .get("refusal")
        .and_then(Value::as_str)
        .filter(|refusal| !refusal.is_empty())
        .map(str::to_owned)
}

/// A chat-completions terminal reason, in the wire's own vocabulary.
#[derive(Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model ended the turn to call tools.
    ToolCalls,
    /// The model stopped naturally.
    Stop,
    /// The provider's content filter ended the turn.
    ContentFilter,
    /// The output cap ended the turn.
    Length,
    /// Anything else the wire sent, preserved verbatim (including the
    /// deprecated `function_call`).
    #[serde(untagged)]
    Other(String),
}

impl FinishReason {
    /// Return the provider's wire spelling, preserving unknown values.
    pub(crate) fn as_wire(&self) -> &str {
        match self {
            Self::ToolCalls => "tool_calls",
            Self::Stop => "stop",
            Self::ContentFilter => "content_filter",
            Self::Length => "length",
            Self::Other(other) => other,
        }
    }
}

/// Chat Completions accounting with dialect-specific fields preserved for
/// the response's `raw`.
// Serde derives a U: Default bound for StreamingCompletionResponse<U>.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatUsage {
    /// The OpenAI-compatible accounting.
    #[serde(flatten)]
    pub openai: Usage,
    /// Fields this dialect adds.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatUsage {
    /// A `u64` counter from the dialect's extra fields.
    fn extra_count(&self, key: &str) -> Option<u64> {
        self.extra.get(key).and_then(serde_json::Value::as_u64)
    }

    /// Normalize this accounting.
    ///
    /// `cached_input_tokens` falls back to DeepSeek's `prompt_cache_hit_tokens`,
    /// which reports cache activity outside `prompt_tokens_details`; reading
    /// only the OpenAI spelling would report no cache hit on a turn that was
    /// entirely served from cache. (Mistral's `num_cached_tokens` is a typed
    /// field of [`Usage`] and handled by its own `to_normalized`.)
    pub fn to_normalized(&self) -> crate::completion::Usage {
        let mut usage = self.openai.to_normalized();
        if usage.cached_input_tokens.is_none() {
            usage.cached_input_tokens = self.extra_count("prompt_cache_hit_tokens");
        }
        usage
    }

    /// Normalize this accounting for a dialect with `quirks`, as the chat
    /// wire does: [`Self::to_normalized`], with the reasoning count left
    /// unreported where the dialect's count cannot be trusted
    /// ([`Quirks::reliable_reasoning_count`](super::Quirks::reliable_reasoning_count)).
    pub fn to_normalized_for(&self, quirks: &super::Quirks) -> crate::completion::Usage {
        let mut usage = self.to_normalized();
        if !quirks.reliable_reasoning_count {
            usage.reasoning_tokens = None;
        }
        usage
    }
}

impl From<ChatUsage> for crate::completion::Usage {
    fn from(value: ChatUsage) -> Self {
        value.to_normalized()
    }
}

/// One choice of a chat-completions frame, in either reply's shape.
#[derive(Deserialize, Debug)]
pub struct ChatChoice {
    /// The streamed shape's fragment, kept as the provider sent it. Defaulted
    /// because a choice on the wire is not guaranteed to carry one: Azure
    /// prepends a `prompt_filter_results` chunk (delta-less choice) to every
    /// stream when content filtering is enabled.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) delta: serde_json::Map<String, serde_json::Value>,
    /// The unary shape's whole assistant message, as the provider sent it.
    /// Absent on a streamed frame; present exactly when this frame is the
    /// unary reply.
    #[serde(default)]
    pub(crate) message: Option<serde_json::Map<String, serde_json::Value>>,
    pub(crate) finish_reason: Option<FinishReason>,
    /// Upstream provider spelling forwarded by gateways such as OpenRouter.
    /// Direct providers omit it.
    #[serde(default)]
    pub(crate) native_finish_reason: Option<String>,
    /// Which candidate this belongs to when the caller asked for `n > 1`.
    /// Optional because providers streaming a single candidate may omit it;
    /// absent is read as candidate 0.
    #[serde(default)]
    pub(crate) index: Option<usize>,
    /// Per-token probabilities, kept as the provider sent them: compatible
    /// services extend the object independently.
    #[serde(default)]
    pub(crate) logprobs: Option<serde_json::Value>,
}

/// One frame of the chat-completions wire.
#[derive(Deserialize, Debug)]
pub struct ChatFrame {
    pub(crate) id: Option<String>,
    pub(crate) model: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) choices: Vec<ChatChoice>,
    pub(crate) usage: Option<ChatUsage>,
    /// Provider-specific top-level fields. Chat-completions-compatible
    /// services add fields independently (`service_tier`, `provider`,
    /// `system_fingerprint`), and the terminal record must not erase them
    /// merely because the shared wire shape does not know their names yet.
    #[serde(flatten)]
    pub(crate) additional_params: serde_json::Map<String, serde_json::Value>,
}

impl ChatFrame {
    /// Whether this frame is the unary `chat.completion` body.
    ///
    /// The `object` tag decides it when the dialect sends one; a choice
    /// carrying a whole `message` decides it when the dialect does not. Both
    /// are needed: `object` is the authoritative tag, and several gateways
    /// omit it entirely.
    pub(crate) fn is_whole(&self) -> bool {
        match self
            .additional_params
            .get("object")
            .and_then(serde_json::Value::as_str)
        {
            // Stream chunks may include whole messages, so an explicit tag wins.
            Some(object) => object == "chat.completion",
            None => self.choices.iter().any(|choice| choice.message.is_some()),
        }
    }
}

/// The provider's own terminal record for one streamed chat-completions
/// reply: the response's `raw`, so a caller reaches every provider field rig
/// does not normalize.
///
/// `U` is the accounting: [`ChatUsage`] on the wire path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse<U = Usage> {
    /// Usage reported on the reply's terminal event; `None` when the reply
    /// never carried one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<U>,
    /// Why the model stopped generating, when the provider reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned response identifier, when the reply emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-reported model identifier, when the reply emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Token log probabilities accumulated from all primary-choice chunks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Provider-specific top-level fields accumulated from the reply, such
    /// as OpenAI's `service_tier` and `system_fingerprint` or OpenRouter's
    /// routed `provider`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Map<String, serde_json::Value>>,
}

impl<U> StreamingCompletionResponse<U>
where
    U: Into<crate::completion::Usage>,
{
    /// The provider's end of the reply: normalized usage and terminal
    /// metadata.
    pub fn into_finish(self) -> crate::operation::Finish {
        crate::operation::Finish {
            usage: self.usage.map(Into::into).unwrap_or_default(),
            reason: self.finish_reason,
            response_id: self.response_id,
            model: self.model,
            ..crate::operation::Finish::default()
        }
    }
}

#[cfg(test)]
mod tests;
