//! Azure OpenAI's typed request options and reply extras, on Chat
//! Completions: OpenAI's Chat fields and Azure's own data sources.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::azure::extension::{AzureExt, AzureOptions};
//! use rig_core::providers::openai::extension::ChatOptions;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = AzureOptions::new().chat(ChatOptions::new().logprobs(true));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<AzureExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
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
}

impl ExtensionOptions for AzureOptions {}

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
