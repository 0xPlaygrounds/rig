//! OpenAI's typed request options and reply extras. [`OpenAiExt`] keys one
//! entry that serves both routes: [`OpenAiShared`] (`"*"`) goes to whichever
//! route the request takes, [`ChatOptions`] (`"openai.chat"`) only to Chat
//! Completions and [`OpenAiResponsesOptions`] (`"openai.responses"`) only to
//! the Responses route.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::openai::extension::{
//!     OpenAiExt, OpenAiOptions, OpenAiResponsesOptions, OpenAiShared, ReasoningSummary,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = OpenAiOptions::default()
//!     .shared(OpenAiShared::default().store(false))
//!     .responses(OpenAiResponsesOptions::default().reasoning_summary(ReasoningSummary::Auto));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<OpenAiExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

mod chat;
mod responses;

pub use chat::{
    ApproximateLocation, AudioFormat, AudioOutput, ChatOptions, CompletionTokensDetails, Modality,
    Prediction, PromptTokensDetails, SearchContextSize, UserLocation, WebSearchOptions,
};
pub(crate) use responses::Envelope;
pub use responses::{
    AccessPrograms, ContextManagement, CyberAccess, Include, ItemPhase, OpenAiResponsesOptions,
    PromptCacheOptions, ReasoningContext, ReasoningMode, ReasoningOptions, ReasoningSummary,
    Truncation,
};

/// OpenAI's provider extension, keyed by
/// [`PROVIDER_NAME`](crate::providers::openai::PROVIDER_NAME).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenAiExt;

impl ProviderExtension for OpenAiExt {
    const PROVIDER: &'static str = crate::providers::openai::PROVIDER_NAME;
    type Options = OpenAiOptions;
    type Extras = OpenAiExtras;
}

/// OpenAI's request options, one section per route plus the shared one.
/// Serialize-only; an unset field is not sent.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenAiOptions {
    /// Fields both routes spell the same.
    #[serde(rename = "*")]
    pub shared: OpenAiShared,
    /// Fields only Chat Completions sends.
    #[serde(rename = "openai.chat")]
    pub chat: ChatOptions,
    /// Fields only the Responses route sends.
    #[serde(rename = "openai.responses")]
    pub responses: OpenAiResponsesOptions,
}

impl OpenAiOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// These options with `chat` as the Chat Completions section.
    #[must_use]
    pub fn chat(mut self, chat: ChatOptions) -> Self {
        self.chat = chat;
        self
    }

    /// These options with `shared` as the shared section.
    #[must_use]
    pub fn shared(mut self, shared: OpenAiShared) -> Self {
        self.shared = shared;
        self
    }

    /// These options with `responses` as the Responses section.
    #[must_use]
    pub fn responses(mut self, responses: OpenAiResponsesOptions) -> Self {
        self.responses = responses;
        self
    }
}

impl ExtensionOptions for OpenAiOptions {}

/// The fields both OpenAI routes take at the top level of the body.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenAiShared {
    /// `store`: whether the provider keeps the response. On Responses,
    /// `false` also sends prior reasoning with its ciphertext instead of by
    /// reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// `metadata`: string pairs the provider keeps with the response.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// `prompt_cache_key`: the key the provider routes cached prompts by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// `safety_identifier`: a stable, hashed id of the end user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

impl OpenAiShared {
    /// Send `store`.
    #[must_use]
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Add `key: value` to `metadata`.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Send `prompt_cache_key`.
    #[must_use]
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Send `safety_identifier`.
    #[must_use]
    pub fn safety_identifier(mut self, id: impl Into<String>) -> Self {
        self.safety_identifier = Some(id.into());
        self
    }
}

/// The typed view of an OpenAI reply. A field the reply does not carry,
/// including one only the other route returns, is `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAiExtras {
    /// `/service_tier`: the tier that served the request, on both routes.
    pub service_tier: Option<String>,
    /// `/reasoning/effort` on Responses: the effort the model used.
    pub reasoning_effort: Option<String>,
    /// `/reasoning/summary` on Responses: `auto`, `concise` or `detailed`.
    pub reasoning_summary: Option<String>,
    /// `/reasoning/mode` on Responses: `standard` or `pro`.
    pub reasoning_mode: Option<String>,
    /// `/reasoning/context` on Responses: `auto`, `all_turns` or
    /// `current_turn`.
    pub reasoning_context: Option<String>,
    /// `/prompt_cache_retention` on Responses: `in_memory` or `24h`.
    pub prompt_cache_retention: Option<String>,
    /// `/incomplete_details/reason` on Responses, such as
    /// `max_output_tokens`.
    pub incomplete_reason: Option<String>,
    /// The `phase` of each `message` item in `/output` on Responses, in
    /// order; `None` when the output holds no message.
    pub phases: Option<Vec<ItemPhase>>,
    /// `/billing/payer` on a unary Responses reply: who pays for it.
    pub billing_payer: Option<String>,
    /// `/system_fingerprint` on Chat: the backend configuration's
    /// fingerprint.
    pub system_fingerprint: Option<String>,
    /// `/usage/prompt_tokens_details` on Chat.
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// `/usage/completion_tokens_details` on Chat.
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// `/choices/0/message/annotations` on Chat, such as URL citations.
    pub annotations: Option<Vec<Value>>,
}

impl ReplyExtras for OpenAiExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() == "openai.chat" {
            return Ok(Self {
                service_tier: reply_field(raw, "/service_tier")?,
                system_fingerprint: reply_field(raw, "/system_fingerprint")?,
                prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
                completion_tokens_details: reply_field(raw, "/usage/completion_tokens_details")?,
                annotations: reply_field(raw, "/choices/0/message/annotations")?,
                ..Self::default()
            });
        }
        let envelope = responses::Envelope::from_reply(raw)?;
        Ok(Self {
            service_tier: envelope.service_tier,
            reasoning_effort: envelope.reasoning_effort,
            reasoning_summary: envelope.reasoning_summary,
            reasoning_mode: envelope.reasoning_mode,
            reasoning_context: envelope.reasoning_context,
            prompt_cache_retention: envelope.prompt_cache_retention,
            incomplete_reason: envelope.incomplete_reason,
            phases: envelope.phases,
            billing_payer: reply_field(raw, "/billing/payer")?,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests;
