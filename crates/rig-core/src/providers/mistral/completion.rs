//! Mistral's model identifiers and its own view of a reply.
//!
//! Mistral is an OpenAI chat-completions dialect, so it has no client and no
//! completion model of its own:
//! [`openai::wire::MISTRAL`](crate::providers::openai::wire::MISTRAL) carries
//! the base URL, the `MISTRAL_API_KEY` variable, the `mistral-correlation-id`
//! request id, the `/v1`-prefixed paths, and the rewrite Mistral needs —
//! `any` for a forced tool choice, a forced choice relaxed beside a structured
//! response format, its assistant-message schema (`prefix`, no
//! `reasoning_content`), and its content chunks (text-only arrays flattened to
//! a string; images, audio and documents rebuilt as Mistral's own chunks).
//!
//! What remains here is data: the model identifiers, and
//! [`CompletionResponse`] — the typed read of Mistral's own reply document,
//! which a completion carries verbatim on
//! [`CompletionResponse::raw`](crate::completion::CompletionResponse::raw).
//! It is not a second mapping: the normalized response is produced by the
//! wire's decoder, and this type is how a caller reads the provider-native
//! fields that mapping does not name — [`Usage::service_tier`],
//! [`Usage::prompt_audio_seconds`], and the audio tokens Mistral reports
//! *beside* `prompt_tokens` rather than inside it.

use serde::{Deserialize, Deserializer, Serialize};

use crate::json_utils;
use crate::providers::openai;

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";

fn mistral_content_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => openai::completion::joined_text_parts(&parts),
        _ => String::new(),
    }
}

fn deserialize_mistral_content_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<serde_json::Value>::deserialize(deserializer)?
        .map(mistral_content_value_to_text)
        .unwrap_or_default())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// Mistral's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        #[serde(default, deserialize_with = "deserialize_mistral_content_string")]
        content: String,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        prefix: bool,
    },
    System {
        content: String,
    },
    Tool {
        /// The name of the tool that was called
        #[serde(skip_serializing_if = "String::is_empty")]
        name: String,
        /// The content of the tool call
        content: String,
        /// The id of the tool call
        tool_call_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    #[serde(
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// In-depth details on prompt tokens.
///
/// Mirrors Mistral's `PromptTokensDetails` schema. The Mistral API also exposes
/// the same shape under the singular field name `prompt_token_details`; the
/// `Usage` field accepts either form via `serde(alias = ...)`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    /// Number of tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
    /// Tokens the audio-input models charge for the prompt's audio. Reported
    /// *alongside* `prompt_tokens` rather than inside it — the two plus
    /// `completion_tokens` are what add up to `total_tokens`.
    #[serde(default)]
    pub audio_tokens: u64,
}

/// Token usage returned by Mistral's chat completions and embeddings endpoints.
///
/// See <https://docs.mistral.ai/api/> (`UsageInfo` schema). The three counts are
/// always present; the remaining fields are populated by Mistral on a best-effort
/// basis (e.g. cached-token information appears once a prompt is large enough to
/// be cached).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    /// Capacity tier that served the request, when Mistral reports it.
    ///
    /// Although the generated `UsageInfo` reference currently omits this
    /// field, the live chat-completions wire includes values such as
    /// `"standard"` in both blocking responses and terminal stream chunks.
    /// Keeping it here prevents the provider-native `raw_completion` and
    /// `raw_stream` surfaces from silently discarding that wire metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Duration in seconds of audio tokens in the prompt (audio-input models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_audio_seconds: Option<u64>,
    /// Total cached prompt tokens reported at the top level. Some Mistral
    /// responses populate this in addition to (or instead of)
    /// `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    /// In-depth breakdown of prompt token usage (currently only cached tokens).
    #[serde(
        default,
        alias = "prompt_token_details",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    /// Returns the number of cached prompt tokens, preferring the structured
    /// `prompt_tokens_details.cached_tokens` field and falling back to the
    /// top-level `num_cached_tokens`. `None` when neither is present.
    pub fn cached_tokens(&self) -> Option<u64> {
        self.prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .or(self.num_cached_tokens)
    }

    /// Tokens charged for audio in the prompt. 0 for every non-audio turn.
    pub fn audio_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .as_ref()
            .map_or(0, |details| details.audio_tokens)
    }

    /// Every token charged against the prompt.
    ///
    /// Mistral reports audio outside `prompt_tokens`: a Voxtral turn answering
    /// a 375-audio-token clip reports `prompt_tokens: 6`, `audio_tokens: 375`,
    /// `completion_tokens: 2` and `total_tokens: 383`. Counting only
    /// `prompt_tokens` as input leaves `input + output` short of `total` by the
    /// whole audio payload.
    pub fn input_tokens(&self) -> u64 {
        self.prompt_tokens as u64 + self.audio_tokens()
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens()),
            output_tokens: Some(usage.completion_tokens as u64),
            total_tokens: Some(usage.total_tokens as u64),
            cached_input_tokens: usage.cached_tokens(),
            ..Default::default()
        }
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(usage: Usage) -> Self {
        Self::from(&usage)
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests;
