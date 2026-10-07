//! Azure OpenAI's typed request options and reply extras, on Chat
//! Completions: OpenAI's Chat fields and Azure's own data sources.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::azure::extension::{AzureOptions};
//!
//! let options = AzureOptions::new().logprobs(true);
//! let request = CompletionRequest::new("hi").provider_option(options);
//! # let _ = request;
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::openai::extension::{
    ChatOptions, CompletionTokensDetails, PromptTokensDetails,
};

/// Azure OpenAI's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AzureExt;

impl ProviderExtension for AzureExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = AzureOptions;
    type Extras = AzureExtras;
}

/// Azure OpenAI's request options, sent on Chat Completions.
///
/// The Chat field setters here write the same field as
/// [`chat`](Self::chat), which replaces every such field set before it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AzureOptions {
    /// Sent on Chat Completions only.
    #[serde(rename = "openai.chat")]
    pub chat: AzureChat,
}

/// The Chat Completions fields Azure takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AzureChat {
    /// OpenAI's Chat fields.
    #[serde(flatten)]
    pub openai: ChatOptions,
    /// The data sources the model answers from
    /// (<https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/on-your-data>),
    /// each sent as given.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub data_sources: Vec<Value>,
}

impl AzureOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send OpenAI's Chat fields `chat`.
    pub fn chat(mut self, chat: ChatOptions) -> Self {
        self.chat.openai = chat;
        self
    }

    /// Answer from `source` too, such as an Azure AI Search index.
    pub fn data_source(mut self, source: Value) -> Self {
        self.chat.data_sources.push(source);
        self
    }

    /// Apply `set` to OpenAI's Chat fields.
    fn with_openai(mut self, set: impl FnOnce(ChatOptions) -> ChatOptions) -> Self {
        self.chat.openai = set(std::mem::take(&mut self.chat.openai));
        self
    }

    /// Whether the reply carries log probabilities, as
    /// [`ChatOptions::logprobs`].
    pub fn logprobs(self, logprobs: bool) -> Self {
        self.with_openai(|chat| chat.logprobs(logprobs))
    }

    /// Report the `count` most likely tokens per position, as
    /// [`ChatOptions::top_logprobs`].
    pub fn top_logprobs(self, count: u8) -> Self {
        self.with_openai(|chat| chat.top_logprobs(count))
    }

    /// Set the frequency penalty, as [`ChatOptions::frequency_penalty`].
    pub fn frequency_penalty(self, penalty: f64) -> Self {
        self.with_openai(|chat| chat.frequency_penalty(penalty))
    }

    /// Set the presence penalty, as [`ChatOptions::presence_penalty`].
    pub fn presence_penalty(self, penalty: f64) -> Self {
        self.with_openai(|chat| chat.presence_penalty(penalty))
    }

    /// Bias token id `token` by `bias`, as [`ChatOptions::logit_bias`].
    pub fn logit_bias(self, token: u32, bias: i32) -> Self {
        self.with_openai(|chat| chat.logit_bias(token, bias))
    }

    /// Predict the output as `content`, as [`ChatOptions::prediction`].
    pub fn prediction(self, content: impl Into<String>) -> Self {
        self.with_openai(|chat| chat.prediction(content))
    }
}

impl ExtensionOptions for AzureOptions {
    type Ext = AzureExt;
}

/// Azure OpenAI's Chat reply fields. Each is `None` when the reply lacks
/// it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AzureExtras {
    /// The service tier that served the request.
    pub service_tier: Option<String>,
    /// The backend configuration's fingerprint.
    pub system_fingerprint: Option<String>,
    /// Prompt token details.
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token details.
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// The content filter's verdicts on the prompt, one per prompt.
    pub prompt_filter_results: Option<Vec<Value>>,
    /// The content filter's verdict on the first choice.
    pub content_filter_results: Option<Value>,
}

impl ReplyExtras for AzureExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            service_tier: reply_field(raw, "/service_tier")?,
            system_fingerprint: reply_field(raw, "/system_fingerprint")?,
            prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
            completion_tokens_details: reply_field(raw, "/usage/completion_tokens_details")?,
            prompt_filter_results: reply_field(raw, "/prompt_filter_results")?,
            content_filter_results: reply_field(raw, "/choices/0/content_filter_results")?,
        })
    }
}

#[cfg(test)]
mod tests;
