//! Chat Completions reply shapes for unary messages and streamed deltas.
//! Unary messages are converted to delta events during classification.

use serde::{Deserialize, Serialize};

use crate::json_utils;
use crate::providers::internal::tool_call_bridge::ToolCallSlot;
use crate::providers::openai::completion::{Message, Usage, joined_text_parts};
use crate::streaming::StreamFinal;

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
    pub(crate) fn evicts(&self, existing: &ToolCallSlot) -> bool {
        if let Some(new_id) = &self.id
            && !new_id.is_empty()
            && let Some(new_name) = &self.function.name
            && self.has_nonempty_name()
            && !existing.id.is_empty()
            && existing.id != *new_id
            && !existing.name.is_empty()
        {
            return existing.name != *new_name || self.starts_new_tool_call();
        }

        false
    }
}

fn deserialize_delta_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Some compatible providers (e.g. Mistral's reasoning models) stream
    // delta content as an array of content parts rather than a string.
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match value {
        serde_json::Value::String(text) => Some(text),
        serde_json::Value::Array(parts) => {
            let text = joined_text_parts(&parts);
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }))
}

/// One streamed choice's delta.
#[derive(Deserialize, Debug, Default, Clone)]
pub(crate) struct StreamingDelta {
    #[serde(default, deserialize_with = "deserialize_delta_content")]
    pub(crate) content: Option<String>,
    /// Refusal text used when content is absent or empty; see [`delta_text`].
    #[serde(default)]
    pub(crate) refusal: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    // Not part of the official OpenAI API; some compatible providers (e.g.
    // Groq) send the same payload under `reasoning`. A separate field rather
    // than a serde alias so a delta carrying BOTH keys is not a
    // duplicate-field error that drops the whole chunk.
    #[serde(default)]
    pub(crate) reasoning: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) tool_calls: Vec<StreamingToolCall>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) reasoning_details: Vec<serde_json::Value>,
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

/// Return nonempty content, falling back to nonempty refusal text.
/// Preserve empty content when no nonempty refusal is available.
pub(crate) fn delta_text(delta: &StreamingDelta) -> Option<String> {
    match delta.content.as_deref() {
        Some(content) if !content.is_empty() => delta.content.clone(),
        content => delta
            .refusal
            .clone()
            .filter(|refusal| !refusal.is_empty())
            .or_else(|| content.map(str::to_owned)),
    }
}

/// Chat Completions accounting with dialect-specific fields preserved for
/// [`StreamFinal::raw`].
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
}

impl From<ChatUsage> for crate::completion::Usage {
    fn from(value: ChatUsage) -> Self {
        value.to_normalized()
    }
}

/// One choice of a chat-completions frame, in either reply's shape.
#[derive(Deserialize, Debug)]
pub struct ChatChoice {
    /// The streamed shape's fragment. Defaulted because a choice on the wire
    /// is not guaranteed to carry one: Azure prepends a
    /// `prompt_filter_results` chunk (delta-less choice) to every stream when
    /// content filtering is enabled.
    #[serde(default)]
    pub(crate) delta: StreamingDelta,
    /// The unary shape's whole assistant message. Absent on a streamed
    /// frame; present exactly when this frame is the unary reply.
    #[serde(default)]
    pub(crate) message: Option<Message>,
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
    /// Per-token probabilities. Kept as provider metadata: compatible
    /// services extend the object independently, and the raw terminal record
    /// must retain every chunk rather than pick a token schema here.
    #[serde(
        default,
        deserialize_with = "crate::message::optional_additional_params"
    )]
    pub(crate) logprobs: Option<crate::message::AdditionalParams>,
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
        match self.object() {
            // Stream chunks may include whole messages, so an explicit tag wins.
            Some(object) => object == "chat.completion",
            // No tag: several gateways omit it, and then a choice carrying a
            // whole `message` rather than a `delta` is the unary body.
            None => self.choices.iter().any(|choice| choice.message.is_some()),
        }
    }

    /// Borrow the `object` tag without removing it from terminal metadata.
    pub(crate) fn object(&self) -> Option<&str> {
        self.additional_params
            .get("object")
            .and_then(serde_json::Value::as_str)
    }

    /// The primary candidate.
    ///
    /// `n > 1` streams as interleaved chunks distinguished only by
    /// `choices[].index`. Taking each frame's *first* choice would
    /// concatenate every candidate into one garbled answer, while the unary
    /// reply is normalized from candidate 0 alone; selecting by index keeps
    /// the two agreeing.
    pub(crate) fn primary(&self) -> Option<&ChatChoice> {
        self.choices
            .iter()
            .find(|choice| choice.index.is_none_or(|index| index == 0))
    }

    /// The primary candidate, taken out of the frame.
    pub(crate) fn into_primary(self) -> Option<ChatChoice> {
        self.choices
            .into_iter()
            .find(|choice| choice.index.is_none_or(|index| index == 0))
    }
}

/// The provider's own terminal record for one chat-completions reply.
///
/// `U` is the accounting: [`ChatUsage`] on the wire path. This is what the
/// decoder serializes onto [`StreamFinal::raw`], so a caller reaches every
/// provider field rig does not normalize.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse<U = Usage> {
    /// Usage reported on the reply's terminal event; `None` when the reply
    /// never carried one (a compatible service that ignores
    /// `stream_options.include_usage`, or a `usage: null` terminal chunk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<U>,
    /// Why the model stopped generating, when the provider reported it.
    ///
    /// Normalized out of the OpenAI-compatible `finish_reason` vocabulary,
    /// with unrecognized values preserved verbatim. The `Stop` -> `ToolCalls`
    /// upgrade is deliberately *not* applied here: it belongs to
    /// [`StreamingCompletionResponse`](crate::streaming::StreamingCompletionResponse),
    /// the only place that sees which tool calls the reply actually emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned response identifier, when the reply emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-reported model identifier, when the reply emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Driver-supplied transport request ID from `x-request-id`, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Token log probabilities accumulated from all primary-choice chunks.
    ///
    /// This stays provider-native: normalized completions do not model log
    /// probabilities, just as the unary path omits `Choice::logprobs` while
    /// its raw response retains them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Provider-specific top-level fields accumulated from the reply, such
    /// as OpenAI's `service_tier` and `system_fingerprint` or OpenRouter's
    /// routed `provider`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::message::optional_additional_params"
    )]
    pub additional_params: Option<crate::message::AdditionalParams>,
}

impl<U> StreamingCompletionResponse<U> {
    /// Create a terminal record carrying `usage`; the optional metadata
    /// starts unset.
    pub fn new(usage: Option<U>) -> Self {
        Self {
            usage,
            finish_reason: None,
            response_id: None,
            model: None,
            provider_request_id: None,
            logprobs: None,
            additional_params: None,
        }
    }
}

impl<U> StreamingCompletionResponse<U>
where
    U: Into<crate::completion::Usage>,
{
    /// Normalize usage and terminal metadata under `provider`, retaining `raw`.
    pub fn into_stream_final(self, provider: &str, raw: serde_json::Value) -> StreamFinal {
        StreamFinal::new(
            provider,
            self.usage.map(Into::into).unwrap_or_default(),
            raw,
        )
        .with_optional_finish_reason(self.finish_reason)
        .with_optional_response_id(self.response_id)
        .with_optional_provider_request_id(self.provider_request_id)
        .with_optional_model(self.model)
    }
}

#[cfg(test)]
mod tests;
