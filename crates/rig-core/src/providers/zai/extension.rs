//! Z.AI's typed request options and reply extras on its OpenAI-format API
//! (<https://docs.z.ai/api-reference/llm/chat-completion>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::zai::extension::{Zai, ZaiChat, ZaiOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = ZaiOptions::new().chat(ZaiChat::new().user_id("user-1"));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<Zai>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Z.AI's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zai;

impl ProviderExtension for Zai {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = ZaiOptions;
    type Extras = ZaiExtras;
}

/// Z.AI's request options, by route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ZaiOptions {
    /// Sent on the OpenAI-format API only.
    #[serde(rename = "openai.chat")]
    pub chat: ZaiChat,
}

impl ZaiOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `chat` on the OpenAI-format API.
    pub fn chat(mut self, chat: ZaiChat) -> Self {
        self.chat = chat;
        self
    }
}

impl ExtensionOptions for ZaiOptions {}

/// The fields Z.AI's OpenAI-format API takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ZaiChat {
    /// Whether to sample; `false` decodes greedily.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    /// The caller's id for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// A stable id of the end user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Thinking fields sent beside the mapped thinking type.
    #[serde(skip_serializing_if = "ZaiThinking::is_empty")]
    pub thinking: ZaiThinking,
}

/// The `thinking` fields Z.AI takes beside the type the generation options
/// map.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ZaiThinking {
    /// Whether earlier turns' thinking is dropped from context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_thinking: Option<bool>,
}

impl ZaiThinking {
    fn is_empty(&self) -> bool {
        self.clear_thinking.is_none()
    }
}

impl ZaiChat {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether to sample.
    pub fn do_sample(mut self, sample: bool) -> Self {
        self.do_sample = Some(sample);
        self
    }

    /// Identify the request as `id`.
    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Name the end user.
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Whether earlier turns' thinking is dropped from context.
    pub fn clear_thinking(mut self, clear: bool) -> Self {
        self.thinking.clear_thinking = Some(clear);
        self
    }
}

/// Z.AI's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZaiExtras {
    /// The request's id. OpenAI-format API only.
    pub request_id: Option<String>,
}

impl ReplyExtras for ZaiExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != "openai.chat" {
            return Ok(Self::default());
        }
        Ok(Self {
            request_id: reply_field(raw, "/request_id")?,
        })
    }
}

#[cfg(test)]
mod tests;
