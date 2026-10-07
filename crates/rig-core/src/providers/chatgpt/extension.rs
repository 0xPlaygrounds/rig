//! ChatGPT's typed request options and reply extras, keyed by
//! [`PROVIDER_NAME`](crate::providers::chatgpt::PROVIDER_NAME). The backend
//! speaks only the Responses format, so its options are one
//! `"openai.responses"` section. An `openai` entry is not read here.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::chatgpt::extension::{ChatGptOptions};
//!
//! let options = ChatGptOptions::default().prompt_cache_key("conversation-1");
//! let request = CompletionRequest::new("hi").provider_option(options);
//! # let _ = request;
//! ```

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::openai::extension::{AccessPrograms, Envelope, ItemPhase};

/// ChatGPT's provider extension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatGptExt;

impl ProviderExtension for ChatGptExt {
    const PROVIDER: &'static str = crate::providers::chatgpt::PROVIDER_NAME;
    type Options = ChatGptOptions;
    type Extras = ChatGptExtras;
}

/// ChatGPT's request options. Serialize-only; an unset field is not sent.
/// The backend stores nothing, so there is no `store`: the wire always
/// sends `false`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChatGptOptions {
    /// The Responses fields, the only route the backend takes.
    #[serde(rename = "openai.responses")]
    pub responses: ChatGptResponses,
}

impl ChatGptOptions {
    /// Send `prompt_cache_key`.
    #[must_use]
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.responses.prompt_cache_key = Some(key.into());
        self
    }

    /// Add `key: value` to `client_metadata`.
    #[must_use]
    pub fn client_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.responses
            .client_metadata
            .insert(key.into(), value.into());
        self
    }

    /// Send `access_programs`.
    #[must_use]
    pub fn access_programs(mut self, programs: AccessPrograms) -> Self {
        self.responses.access_programs = Some(programs);
        self
    }
}

impl ExtensionOptions for ChatGptOptions {
    type Ext = ChatGptExt;
}

/// The fields the ChatGPT backend takes at the top level of a Responses
/// body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChatGptResponses {
    /// `prompt_cache_key`: the key the backend routes cached prompts by. A
    /// stable one, such as a conversation id, keeps a conversation's cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// `client_metadata`: string pairs describing the client.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub client_metadata: BTreeMap<String, String>,
    /// `access_programs`: the access programs the request runs under.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_programs: Option<AccessPrograms>,
}

/// The typed view of a ChatGPT reply: the terminal response object of its
/// event stream. A field the reply does not carry is `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatGptExtras {
    /// `/service_tier`: the tier that served the request.
    pub service_tier: Option<String>,
    /// `/reasoning/effort`: the effort the model used.
    pub reasoning_effort: Option<String>,
    /// `/reasoning/summary`: `auto`, `concise` or `detailed`.
    pub reasoning_summary: Option<String>,
    /// `/reasoning/mode`: `standard` or `pro`.
    pub reasoning_mode: Option<String>,
    /// `/reasoning/context`: `auto`, `all_turns` or `current_turn`.
    pub reasoning_context: Option<String>,
    /// `/prompt_cache_retention`: the backend keeps prompts for `24h`.
    pub prompt_cache_retention: Option<String>,
    /// `/incomplete_details/reason`, such as `max_output_tokens`.
    pub incomplete_reason: Option<String>,
    /// The `phase` of each `message` item in `/output`, which the reply's
    /// `raw` rebuilds from the items the stream finished; `None` when the
    /// output holds no message.
    pub phases: Option<Vec<ItemPhase>>,
}

impl ReplyExtras for ChatGptExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        let envelope = Envelope::from_reply(raw)?;
        Ok(Self {
            service_tier: envelope.service_tier,
            reasoning_effort: envelope.reasoning_effort,
            reasoning_summary: envelope.reasoning_summary,
            reasoning_mode: envelope.reasoning_mode,
            reasoning_context: envelope.reasoning_context,
            prompt_cache_retention: envelope.prompt_cache_retention,
            incomplete_reason: envelope.incomplete_reason,
            phases: envelope.phases,
        })
    }
}

#[cfg(test)]
mod tests;
