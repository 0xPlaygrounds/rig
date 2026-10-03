//! Chat Completions reply shapes for unary messages and streamed deltas,
//! and the merge that assembles a streamed message from its deltas.

use serde::{Deserialize, Serialize};

use crate::json_utils;
use crate::providers::openai::completion::Usage;

/// A streamed tool-call fragment's function half.
#[derive(Default, Debug, Clone)]
pub(crate) struct StreamingFunction {
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

/// One streamed tool-call fragment.
#[derive(Debug, Clone)]
pub(crate) struct StreamingToolCall {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) function: StreamingFunction,
}

impl StreamingToolCall {
    /// The fragment `call` states, read leniently: an `index` that is not
    /// an integer is a single in-flight call's 0, a numeric `id` is its
    /// digits, and arguments that are not a string are their JSON text.
    pub(crate) fn read(call: &serde_json::Value) -> Self {
        use serde_json::Value;
        Self {
            index: call
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .unwrap_or(0),
            id: match call.get("id") {
                Some(Value::String(id)) => Some(id.clone()),
                Some(id @ Value::Number(_)) => Some(id.to_string()),
                _ => None,
            },
            function: StreamingFunction {
                name: call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                arguments: match call.pointer("/function/arguments") {
                    None | Some(Value::Null) => None,
                    Some(arguments) => Some(json_utils::value_to_json_string(arguments)),
                },
            },
        }
    }

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
/// text and reasoning, a call's arguments or custom input, audio's transcript
/// and data, a reasoning detail's text and summary. Every other string is an identifier,
/// a tag or a signature a later fragment restates.
const FRAGMENT_KEYS: [&str; 13] = [
    "content",
    "input",
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
/// merge key by key, and content parts (a message's `content`, a thinking
/// part's `thinking`) go through [`merge_content`]. A
/// value never replaces one of another JSON type, a `null` or empty string
/// never erases a value, and a literal `null` argument placeholder gives way
/// to the first real fragment.
pub(crate) fn merge_fields(
    target: &mut serde_json::Map<String, serde_json::Value>,
    delta: &serde_json::Map<String, serde_json::Value>,
) {
    use serde_json::Value;
    for (key, value) in delta {
        let fragment = FRAGMENT_KEYS.contains(&key.as_str());
        match (target.get_mut(key), value) {
            (Some(_), Value::Null) => {}
            (Some(existing), more)
                if matches!(key.as_str(), "content" | "thinking")
                    && (existing.is_array() || more.is_array()) =>
            {
                merge_content(existing, more);
            }
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
            (Some(existing), more)
                if !existing.is_null()
                    && std::mem::discriminant(existing) != std::mem::discriminant(more) => {}
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Merge streamed message content into what arrived so far, as content
/// parts once either side is a part array: a string is a text part, and a
/// text or thinking part continues the last part of its type.
fn merge_content(existing: &mut serde_json::Value, more: &serde_json::Value) {
    use serde_json::Value;
    fn parts(value: Value) -> Vec<Value> {
        match value {
            Value::Array(parts) => parts,
            Value::String(text) if !text.is_empty() => {
                vec![serde_json::json!({"type": "text", "text": text})]
            }
            _ => Vec::new(),
        }
    }
    let mut merged = parts(std::mem::take(existing));
    for part in parts(more.clone()) {
        let kind = part.get("type").and_then(Value::as_str).map(str::to_owned);
        let kind = kind.as_deref();
        match (merged.last_mut(), part) {
            (Some(Value::Object(last)), Value::Object(next))
                if matches!(kind, Some("text" | "thinking"))
                    && last.get("type").and_then(Value::as_str) == kind =>
            {
                merge_fields(last, &next);
            }
            (_, part) => merged.push(part),
        }
    }
    *existing = Value::Array(merged);
}

/// The text a message or delta carries as a string: its `content`, falling
/// back to the sibling `refusal` when there is no content. Content-part
/// arrays go through the decoder's part dispatcher instead.
pub(crate) fn delta_text(delta: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    ["content", "refusal"].into_iter().find_map(|key| {
        delta
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    })
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
}

impl ChatUsage {
    /// This accounting normalized for a dialect with `quirks`, as the chat
    /// wire reads it (`UsageCounts`).
    pub fn to_normalized_for(&self, quirks: &super::Quirks) -> crate::completion::Usage {
        UsageCounts::read(&serde_json::to_value(self).unwrap_or_default()).normalized(quirks)
    }
}

impl From<ChatUsage> for crate::completion::Usage {
    fn from(value: ChatUsage) -> Self {
        value.to_normalized()
    }
}

/// The counters of a Chat usage object, read leniently: a counter that is
/// absent, `null` or not a non-negative integer is unreported, so no usage
/// shape fails a reply. One reader for the decoder and the observation
/// projection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UsageCounts {
    pub(crate) prompt: Option<u64>,
    pub(crate) completion: Option<u64>,
    pub(crate) total: Option<u64>,
    /// `prompt_tokens_details.cached_tokens`.
    pub(crate) cached: Option<u64>,
    /// `completion_tokens_details.reasoning_tokens`.
    pub(crate) reasoning: Option<u64>,
    audio: Option<u64>,
    cache_write: Option<u64>,
    /// Whether the usage has the details objects the counts above sit in.
    prompt_details: bool,
    completion_details: bool,
    /// Mistral's `num_cached_tokens` and DeepSeek's `prompt_cache_hit_tokens`.
    num_cached: Option<u64>,
    cache_hit: Option<u64>,
}

impl UsageCounts {
    /// The counters of `usage`.
    pub(crate) fn read(usage: &serde_json::Value) -> Self {
        let count = |pointer: &str| usage.pointer(pointer).and_then(serde_json::Value::as_u64);
        let object = |key: &str| usage.get(key).is_some_and(serde_json::Value::is_object);
        Self {
            prompt: count("/prompt_tokens"),
            completion: count("/completion_tokens"),
            total: count("/total_tokens"),
            cached: count("/prompt_tokens_details/cached_tokens"),
            reasoning: count("/completion_tokens_details/reasoning_tokens"),
            audio: count("/prompt_tokens_details/audio_tokens"),
            cache_write: count("/prompt_tokens_details/cache_write_tokens"),
            prompt_details: object("prompt_tokens_details"),
            completion_details: object("completion_tokens_details"),
            num_cached: count("/num_cached_tokens"),
            cache_hit: count("/prompt_cache_hit_tokens"),
        }
    }

    /// The normalized usage, as [`Usage::to_normalized`] reads it, for a
    /// dialect with `quirks`. A counter inside a details object the usage
    /// has is zero when unreported; cached input falls back to Mistral's
    /// and then DeepSeek's spelling; output is the remainder of the total
    /// when unreported; and the reasoning count is left out where the
    /// dialect's cannot be trusted
    /// ([`Quirks::reliable_reasoning_count`](super::Quirks::reliable_reasoning_count)).
    pub(crate) fn normalized(&self, quirks: &super::Quirks) -> crate::completion::Usage {
        let audio = self.audio.unwrap_or(0);
        let input = self.prompt.map(|prompt| {
            let beside = prompt.saturating_add(audio);
            let accounted = beside.saturating_add(self.completion.unwrap_or(0));
            if audio != 0 && Some(accounted) == self.total {
                beside
            } else {
                prompt
            }
        });
        let in_details = |count: Option<u64>, present: bool| count.or(present.then_some(0));
        crate::completion::Usage {
            input_tokens: input,
            output_tokens: self.completion.or_else(|| {
                self.total
                    .map(|total| total.saturating_sub(input.unwrap_or(0)))
            }),
            total_tokens: self.total,
            cached_input_tokens: in_details(self.cached, self.prompt_details)
                .or(self.num_cached)
                .or(self.cache_hit),
            cache_creation_input_tokens: self.cache_write,
            reasoning_tokens: in_details(self.reasoning, self.completion_details)
                .filter(|_| quirks.reliable_reasoning_count),
            ..Default::default()
        }
    }
}

/// A field of a reply read leniently: a value of another type is absent,
/// so no one field fails a reply.
fn lenient<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    Ok(T::deserialize(serde_json::Value::deserialize(deserializer)?).unwrap_or_default())
}

/// One choice of a chat-completions frame, in either reply's shape. Every
/// field is read [`lenient`]ly.
#[derive(Deserialize, Debug)]
pub struct ChatChoice {
    /// The streamed shape's fragment, kept as the provider sent it. Defaulted
    /// because a choice on the wire is not guaranteed to carry one: Azure
    /// prepends a `prompt_filter_results` chunk (delta-less choice) to every
    /// stream when content filtering is enabled.
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) delta: serde_json::Map<String, serde_json::Value>,
    /// The unary shape's whole assistant message, as the provider sent it.
    /// Absent on a streamed frame; present exactly when this frame is the
    /// unary reply.
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) message: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) finish_reason: Option<FinishReason>,
    /// Upstream provider spelling forwarded by gateways such as OpenRouter.
    /// Direct providers omit it.
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) native_finish_reason: Option<String>,
    /// Which candidate this belongs to when the caller asked for `n > 1`.
    /// Optional because providers streaming a single candidate may omit it;
    /// absent is read as candidate 0.
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) index: Option<usize>,
    /// Per-token probabilities, kept as the provider sent them: compatible
    /// services extend the object independently.
    #[serde(default)]
    pub(crate) logprobs: Option<serde_json::Value>,
}

/// One frame of the chat-completions wire, read [`lenient`]ly but for its
/// `choices`.
#[derive(Deserialize, Debug)]
pub struct ChatFrame {
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) id: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub(crate) model: Option<String>,
    /// The one field a frame is built from, so a value of another type is
    /// a defect of the frame.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) choices: Vec<ChatChoice>,
    /// The usage object as the provider sent it, read by [`UsageCounts`].
    #[serde(default)]
    pub(crate) usage: Option<serde_json::Value>,
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
/// `U` is the accounting: the provider's usage object as it came on the wire
/// path, and [`ChatUsage`] for a caller reading it back typed.
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

#[cfg(test)]
mod tests;
