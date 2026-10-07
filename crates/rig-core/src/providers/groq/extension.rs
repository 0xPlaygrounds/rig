//! Groq's typed request options and reply extras
//! (<https://console.groq.com/docs/api-reference>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::groq::extension::{GroqExt, GroqOptions, ReasoningFormat};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = GroqOptions::new().reasoning_format(ReasoningFormat::Parsed);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<GroqExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Groq's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroqExt;

impl ProviderExtension for GroqExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = GroqOptions;
    type Extras = GroqExtras;
}

/// Groq's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GroqOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: GroqShared,
}

/// The fields Groq takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GroqShared {
    /// How the reply carries the reasoning. Excludes `include_reasoning`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// Whether the reply carries the reasoning. Excludes
    /// `reasoning_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    /// Which sites the built-in web search may read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_settings: Option<SearchSettings>,
    /// Whether the reply cites its documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_options: Option<CitationOptions>,
}

/// How a reply carries the reasoning.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    /// Not at all.
    Hidden,
    /// Inline in the content, in `<think>` tags.
    Raw,
    /// In the message's `reasoning` field.
    Parsed,
}

/// Whether a reply cites its documents.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CitationOptions {
    /// Cite them.
    Enabled,
    /// Do not.
    Disabled,
}

/// Which sites the built-in web search may read.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SearchSettings {
    /// Domains never searched.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exclude_domains: Vec<String>,
    /// The only domains searched.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub include_domains: Vec<String>,
    /// The country results are boosted for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
}

impl SearchSettings {
    /// No restriction.
    pub fn new() -> Self {
        Self::default()
    }

    /// Never search `domains`.
    pub fn exclude_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.exclude_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Search only `domains`.
    pub fn include_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.include_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Boost results for `country`.
    pub fn country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }
}

impl GroqOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Carry the reasoning as `format`, in place of `include_reasoning`.
    pub fn reasoning_format(mut self, format: ReasoningFormat) -> Self {
        self.shared.reasoning_format = Some(format);
        self.shared.include_reasoning = None;
        self
    }

    /// Whether the reply carries the reasoning, in place of
    /// `reasoning_format`.
    pub fn include_reasoning(mut self, include: bool) -> Self {
        self.shared.include_reasoning = Some(include);
        self.shared.reasoning_format = None;
        self
    }

    /// Restrict the built-in web search by `settings`.
    pub fn search_settings(mut self, settings: SearchSettings) -> Self {
        self.shared.search_settings = Some(settings);
        self
    }

    /// Whether the reply cites its documents.
    pub fn citation_options(mut self, citations: CitationOptions) -> Self {
        self.shared.citation_options = Some(citations);
        self
    }
}

impl ExtensionOptions for GroqOptions {}

/// Groq's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroqExtras {
    /// Groq's own envelope, such as the request id.
    pub x_groq: Option<Value>,
    /// Seconds the request queued.
    pub queue_time: Option<f64>,
    /// Seconds spent on the prompt.
    pub prompt_time: Option<f64>,
    /// Seconds spent on the completion.
    pub completion_time: Option<f64>,
    /// Seconds in all.
    pub total_time: Option<f64>,
    /// Usage per model, for compound systems.
    pub usage_breakdown: Option<Value>,
    /// The service tier that served the request.
    pub service_tier: Option<String>,
    /// The built-in tools the model ran.
    pub executed_tools: Option<Vec<Value>>,
    /// The backend configuration's fingerprint.
    pub system_fingerprint: Option<String>,
}

impl ReplyExtras for GroqExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            x_groq: reply_field(raw, "/x_groq")?,
            queue_time: reply_field(raw, "/usage/queue_time")?,
            prompt_time: reply_field(raw, "/usage/prompt_time")?,
            completion_time: reply_field(raw, "/usage/completion_time")?,
            total_time: reply_field(raw, "/usage/total_time")?,
            usage_breakdown: reply_field(raw, "/usage_breakdown")?,
            service_tier: reply_field(raw, "/service_tier")?,
            executed_tools: reply_field(raw, "/choices/0/message/executed_tools")?,
            system_fingerprint: reply_field(raw, "/system_fingerprint")?,
        })
    }
}

#[cfg(test)]
mod tests;
