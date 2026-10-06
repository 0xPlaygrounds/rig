//! MiniMax's typed request options and reply extras on its OpenAI-format
//! API (<https://platform.minimax.io/docs/api-reference/text-openai-api>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::minimax::extension::{MiniMax, MiniMaxChat, MiniMaxOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = MiniMaxOptions::new().chat(MiniMaxChat::new().reasoning_split(true));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<MiniMax>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// MiniMax's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MiniMax;

impl ProviderExtension for MiniMax {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = MiniMaxOptions;
    type Extras = MiniMaxExtras;
}

/// MiniMax's request options, by route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MiniMaxOptions {
    /// Sent on the OpenAI-format API only.
    #[serde(rename = "openai.chat")]
    pub chat: MiniMaxChat,
}

impl MiniMaxOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `chat` on the OpenAI-format API.
    pub fn chat(mut self, chat: MiniMaxChat) -> Self {
        self.chat = chat;
        self
    }
}

impl ExtensionOptions for MiniMaxOptions {}

/// The fields MiniMax's OpenAI-format API takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MiniMaxChat {
    /// Whether the reasoning comes back in `reasoning_details` rather than
    /// in `<think>` tags in the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_split: Option<bool>,
}

impl MiniMaxChat {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the reasoning apart from the content when `split`.
    pub fn reasoning_split(mut self, split: bool) -> Self {
        self.reasoning_split = Some(split);
        self
    }
}

/// MiniMax's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MiniMaxExtras {
    /// The first choice's reasoning, split from the content. OpenAI-format
    /// API only.
    pub reasoning_details: Option<Vec<Value>>,
    /// MiniMax's status envelope. OpenAI-format API only.
    pub base_resp: Option<Value>,
}

impl ReplyExtras for MiniMaxExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() != "openai.chat" {
            return Ok(Self::default());
        }
        Ok(Self {
            reasoning_details: reply_field(raw, "/choices/0/message/reasoning_details")?,
            base_resp: reply_field(raw, "/base_resp")?,
        })
    }
}

#[cfg(test)]
mod tests;
