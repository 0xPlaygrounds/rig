//! The chat-completions wire's reply shapes.
//!
//! One frame type serves both replies. A streamed frame is a
//! `chat.completion.chunk` whose choice carries a `delta`; the unary reply is
//! a `chat.completion` whose choice carries a whole `message`. Modelling them
//! as one [`ChatFrame`] with two choice shapes means the typed decode happens
//! once per frame and the unary body is *converted to the stream's shape in
//! `classify`* — there is no second content mapping to drift from the first.

use serde::{Deserialize, Serialize};

use crate::json_utils;
use crate::providers::internal::openai_chat_completions_compatible::{
    CompatibleTerminal, CompatibleToolCallChunk,
};
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

impl From<&StreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &StreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
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
    /// A structured-output refusal streams here, on its own key, with
    /// `content` held at `null` for the whole turn — the same sibling-of-
    /// `content` spelling the unary body uses. Its deltas are the turn's
    /// visible text, so they join the text stream (see [`delta_text`]).
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
    /// This reason in the provider's own wire spelling.
    ///
    /// Round-tripping through the wire form keeps `map_openai_finish_reason`
    /// the single place the OpenAI-compatible vocabulary is interpreted, so
    /// the streaming and unary paths cannot drift — including on the
    /// deprecated `function_call` spelling, which this enum captures in
    /// [`FinishReason::Other`].
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

/// The visible text a delta carries: its `content`, or — when `content` has
/// none — its `refusal`.
///
/// A refusal turn streams `"content": null` beside the refusal deltas (and
/// opens with an empty `"refusal": ""`), so preferring non-empty content
/// keeps ordinary turns byte-identical while letting a refusal reach the
/// caller instead of vanishing. An empty `content` string with no refusal to
/// fall back on stays exactly as it was.
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

/// The accounting a chat-completions reply reports.
///
/// The OpenAI-compatible fields every dialect on this wire sends, plus
/// whatever else the dialect added: a gateway's `cost`, DeepSeek's
/// `prompt_cache_hit_tokens`, llama.cpp's `timings`. The extras ride along so
/// [`StreamFinal::raw`] loses nothing, which is what the typed escape hatch
/// used to provide through a per-provider `StreamingUsage` type.
// `Default` is load-bearing, not decoration: `StreamingCompletionResponse`
// declares `#[serde(default)] usage: Option<U>`, and serde's derive
// propagates that as a `U: Default` bound on the generated `Deserialize`.
// Without it a caller cannot name `ChatUsage` through the very record this
// type's docs advertise as the typed escape hatch.
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
            // The tag is authoritative when the dialect sends one. It has to
            // be: Perplexity streams a full `message` on every chunk beside
            // its `delta`, and its terminator is tagged
            // `chat.completion.done`, so keying on "a choice carries a
            // message" read every one of its stream frames as the whole
            // reply — the first frame emitted a terminal, the driver stopped,
            // and the rest of the turn was dropped.
            Some(object) => object == "chat.completion",
            // No tag: several gateways omit it, and then a choice carrying a
            // whole `message` rather than a `delta` is the unary body.
            None => self.choices.iter().any(|choice| choice.message.is_some()),
        }
    }

    /// The frame's `object` tag, when the dialect sends one.
    ///
    /// Read out of the flattened metadata rather than declared as a field on
    /// purpose: a named field would *consume* the key, and `object` would
    /// then be missing from the terminal record's `additional_params` while
    /// its neighbours (`service_tier`, `system_fingerprint`) survived — the
    /// old adapter accumulated all of them.
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
    /// The transport request id from the reply's `x-request-id` header — not
    /// part of any frame; stamped by the driver. `None` when the provider
    /// did not report one.
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

    /// Build the terminal record from the decoder's terminal state.
    pub(crate) fn from_terminal(terminal: CompatibleTerminal<U>) -> Self {
        Self {
            usage: terminal.usage,
            finish_reason: terminal.finish_reason,
            response_id: terminal.response_id,
            model: terminal.model,
            // Stamped by the driver; the decoder never sees connection
            // headers.
            provider_request_id: None,
            logprobs: terminal.logprobs.map(Into::into),
            additional_params: terminal.additional_params,
        }
    }
}

impl<U> StreamingCompletionResponse<U>
where
    U: Into<crate::completion::Usage>,
{
    /// Normalize this terminal record, attributed to `provider`.
    ///
    /// The provider descriptor name is an *input* rather than a constant:
    /// this record is shared by every dialect on the wire, so baking in
    /// `"openai"` would mislabel Groq, Together, DeepSeek and the rest.
    pub fn into_stream_final(self, provider: &str) -> StreamFinal {
        StreamFinal::new(provider, self.usage.map(Into::into).unwrap_or_default())
            .with_optional_finish_reason(self.finish_reason)
            .with_optional_response_id(self.response_id)
            .with_optional_provider_request_id(self.provider_request_id)
            .with_optional_model(self.model)
    }
}
