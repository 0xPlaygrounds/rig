//! Moonshot's typed request options and reply extras. One entry serves its
//! OpenAI-format API (<https://platform.kimi.ai/docs/api/chat>) and its
//! Anthropic-format Messages API, which takes no Moonshot field of its own.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::moonshot::extension::{Moonshot, MoonshotChat, MoonshotOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = MoonshotOptions::new().chat(MoonshotChat::new().prompt_cache_key("session-7"));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<Moonshot>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::anthropic::extension::MessagesStop;
use crate::providers::anthropic::wire::MESSAGES_API;

/// Moonshot's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moonshot;

impl ProviderExtension for Moonshot {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = MoonshotOptions;
    type Extras = MoonshotExtras;
}

/// Moonshot's request options, by route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MoonshotOptions {
    /// Sent on the OpenAI-format API only.
    #[serde(rename = "openai.chat")]
    pub chat: MoonshotChat,
}

impl MoonshotOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `chat` on the OpenAI-format API.
    pub fn chat(mut self, chat: MoonshotChat) -> Self {
        self.chat = chat;
        self
    }
}

impl ExtensionOptions for MoonshotOptions {}

/// The fields Moonshot's OpenAI-format API takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MoonshotChat {
    /// Thinking fields sent beside the mapped thinking type.
    #[serde(skip_serializing_if = "MoonshotThinking::is_empty")]
    pub thinking: MoonshotThinking,
    /// Routes requests that share a prefix to the same cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

/// The `thinking` fields Moonshot takes beside the type the generation
/// options map.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MoonshotThinking {
    /// Which earlier turns' thinking the model keeps in context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep: Option<ThinkingKeep>,
}

impl MoonshotThinking {
    fn is_empty(&self) -> bool {
        self.keep.is_none()
    }
}

/// Which earlier turns' thinking the model keeps in context.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingKeep {
    /// Every turn's.
    All,
}

impl MoonshotChat {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep `keep` earlier turns' thinking in context.
    pub fn thinking_keep(mut self, keep: ThinkingKeep) -> Self {
        self.thinking.keep = Some(keep);
        self
    }

    /// Route the prompt cache by `key`.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }
}

/// Moonshot's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MoonshotExtras {
    /// The first choice's own usage. OpenAI-format API only.
    pub choice_usage: Option<Value>,
    /// Prompt token details. OpenAI-format API only.
    pub prompt_tokens_details: Option<Value>,
    /// `stop_reason`, verbatim. Messages route.
    pub stop_reason: Option<String>,
    /// `stop_sequence`: the stop sequence the turn ended on. Messages route.
    pub stop_sequence: Option<String>,
}

impl ReplyExtras for MoonshotExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() == MESSAGES_API {
            let MessagesStop {
                stop_reason,
                stop_sequence,
            } = MessagesStop::read("Moonshot", api, raw)?;
            return Ok(Self {
                stop_reason,
                stop_sequence,
                ..Self::default()
            });
        }
        if api.as_str() != "openai.chat" {
            return Ok(Self::default());
        }
        Ok(Self {
            choice_usage: reply_field(raw, "/choices/0/usage")?,
            prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests;
