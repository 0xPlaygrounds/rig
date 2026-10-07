//! MiniMax's typed request options and reply extras. One entry serves its
//! OpenAI-format API
//! (<https://platform.minimax.io/docs/api-reference/text-openai-api>) and its
//! Anthropic-format Messages API; each section goes only to its route.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::minimax::extension::{MiniMaxExt, MiniMaxChat, MiniMaxOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = MiniMaxOptions::new().chat(MiniMaxChat::new().reasoning_split(true));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<MiniMaxExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::anthropic::extension::{MessagesStop, Nested};
use crate::providers::anthropic::wire::MESSAGES_API;

/// MiniMax's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MiniMaxExt;

impl ProviderExtension for MiniMaxExt {
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
    /// Sent on the Anthropic-format Messages API only.
    #[serde(rename = "anthropic.messages")]
    pub messages: MiniMaxMessages,
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

    /// Send `messages` on the Messages API.
    pub fn messages(mut self, messages: MiniMaxMessages) -> Self {
        self.messages = messages;
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

/// The fields MiniMax's Messages API takes. Each unset field is left out
/// of the body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MiniMaxMessages {
    /// `metadata.user_id`: an opaque id of the end user.
    pub metadata_user_id: Option<String>,
}

impl MiniMaxMessages {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `user_id` as `metadata.user_id`.
    pub fn metadata_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.metadata_user_id = Some(user_id.into());
        self
    }
}

impl Serialize for MiniMaxMessages {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self { metadata_user_id } = self;
        let mut map = serializer.serialize_map(None)?;
        if let Some(user_id) = metadata_user_id {
            map.serialize_entry("metadata", &Nested("user_id", user_id))?;
        }
        map.end()
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
    /// `stop_reason`, verbatim. Messages route.
    pub stop_reason: Option<String>,
    /// `stop_sequence`: the stop sequence the turn ended on. Messages route.
    pub stop_sequence: Option<String>,
}

impl ReplyExtras for MiniMaxExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() == MESSAGES_API {
            let MessagesStop {
                stop_reason,
                stop_sequence,
            } = MessagesStop::read("MiniMax", api, raw)?;
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
            reasoning_details: reply_field(raw, "/choices/0/message/reasoning_details")?,
            base_resp: reply_field(raw, "/base_resp")?,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests;
