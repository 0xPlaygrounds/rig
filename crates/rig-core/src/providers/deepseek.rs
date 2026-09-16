//! DeepSeek's model identifiers and its own view of a reply.
//!
//! DeepSeek is an OpenAI chat-completions dialect, so it has no client and
//! no completion model of its own:
//! [`openai::wire::DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK)
//! carries the base URL, the `DEEPSEEK_API_KEY` variable, the
//! `/user/balance` credential check, and the rewrite DeepSeek needs —
//! string-flattened message content, `content: ""` on tool-call-only
//! assistant turns, `index` on echoed tool calls, and forced tool choices
//! suppressed unless thinking is explicitly disabled.
//!
//! What remains here is data: the model identifiers, and
//! [`CompletionResponse`] — the typed read of DeepSeek's own reply document,
//! which a completion carries verbatim on
//! [`CompletionResponse::raw`](crate::completion::CompletionResponse::raw).
//! It is not a second mapping: the normalized response is produced by the
//! wire's decoder, and this type is how a caller reads the provider-native
//! fields that mapping does not name — `prompt_cache_hit_tokens` /
//! `prompt_cache_miss_tokens`, and `reasoning_content` under DeepSeek's own
//! spelling.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::deepseek;
//! use rig_core::providers::openai::wire::{DEEPSEEK, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let deepseek_chat = OpenAI::from_env_with(&DEEPSEEK)?.chat(deepseek::DEEPSEEK_V4_FLASH);
//! # let _ = deepseek_chat;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::json_utils;

// ================================================================
// DeepSeek Completion API
// ================================================================
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// A `max_tokens`-truncated tool call comes back with its `arguments`
    /// cut off partway through the JSON object. Dropping just that call at
    /// decode keeps the rest of the turn — the text, usage, id, model and
    /// finish reason — instead of failing the whole document (rig#2354).
    #[serde(
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub prompt_cache_hit_tokens: u32,
    pub prompt_cache_miss_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// DeepSeek's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// only exists on `deepseek-reasoner` model at time of addition
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
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
