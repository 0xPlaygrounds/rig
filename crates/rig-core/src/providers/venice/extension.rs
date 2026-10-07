//! Venice's typed request options and reply extras. Turning reasoning off
//! is [`Reasoning::Off`](crate::completion::Reasoning::Off), not a Venice
//! parameter.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::venice::extension::{
//!     VeniceExt, VeniceOptions, VeniceParameters, WebSearchMode,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = VeniceOptions::new().venice_parameters(
//!     VeniceParameters::new()
//!         .enable_web_search(WebSearchMode::On)
//!         .enable_web_citations(true),
//! );
//! let request = CompletionRequest::new("Summarize today's Rust news.")
//!     .provider_options(ProviderOptions::new().with::<VeniceExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Venice's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VeniceExt;

impl ProviderExtension for VeniceExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = VeniceOptions;
    type Extras = VeniceExtras;
}

/// Venice's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VeniceOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: VeniceShared,
}

/// The fields Venice takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VeniceShared {
    /// Venice's own parameter block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub venice_parameters: Option<VeniceParameters>,
    /// The prompt-cache routing key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

impl VeniceOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send Venice's parameter block `parameters`.
    pub fn venice_parameters(mut self, parameters: VeniceParameters) -> Self {
        self.shared.venice_parameters = Some(parameters);
        self
    }

    /// Route the prompt cache by `key`.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.shared.prompt_cache_key = Some(key.into());
        self
    }
}

impl ExtensionOptions for VeniceOptions {}

/// How Venice's web search behaves for a request.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchMode {
    /// Never search.
    Off,
    /// Always search.
    On,
    /// Let the model decide.
    Auto,
}

/// Venice's `venice_parameters` block. Unset fields keep Venice's defaults
/// (`include_venice_system_prompt` defaults to `true`).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct VeniceParameters {
    /// A public character to converse with, by slug.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character_slug: Option<String>,
    /// Strip `<think>` blocks from the reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strip_thinking_response: Option<bool>,
    /// The web-search mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_search: Option<WebSearchMode>,
    /// Scrape URLs found in the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_scraping: Option<bool>,
    /// Use xAI's own search on Grok models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_x_search: Option<bool>,
    /// Cite sources as `[REF]` markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_citations: Option<bool>,
    /// Stream the search results too.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_search_results_in_stream: Option<bool>,
    /// Return the search results as tool-call documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_search_results_as_documents: Option<bool>,
    /// Include Venice's default system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_venice_system_prompt: Option<bool>,
}

impl VeniceParameters {
    /// No parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Converse with the public character `slug`.
    pub fn character_slug(mut self, slug: impl Into<String>) -> Self {
        self.character_slug = Some(slug.into());
        self
    }

    /// Strip `<think>` blocks from the reply.
    pub fn strip_thinking_response(mut self, strip: bool) -> Self {
        self.strip_thinking_response = Some(strip);
        self
    }

    /// Set the web-search mode.
    pub fn enable_web_search(mut self, mode: WebSearchMode) -> Self {
        self.enable_web_search = Some(mode);
        self
    }

    /// Scrape URLs found in the prompt.
    pub fn enable_web_scraping(mut self, enable: bool) -> Self {
        self.enable_web_scraping = Some(enable);
        self
    }

    /// Use xAI's own search on Grok models.
    pub fn enable_x_search(mut self, enable: bool) -> Self {
        self.enable_x_search = Some(enable);
        self
    }

    /// Cite sources as `[REF]` markers.
    pub fn enable_web_citations(mut self, enable: bool) -> Self {
        self.enable_web_citations = Some(enable);
        self
    }

    /// Stream the search results too.
    pub fn include_search_results_in_stream(mut self, include: bool) -> Self {
        self.include_search_results_in_stream = Some(include);
        self
    }

    /// Return the search results as tool-call documents.
    pub fn return_search_results_as_documents(mut self, as_documents: bool) -> Self {
        self.return_search_results_as_documents = Some(as_documents);
        self
    }

    /// Include Venice's default system prompt.
    pub fn include_venice_system_prompt(mut self, include: bool) -> Self {
        self.include_venice_system_prompt = Some(include);
        self
    }
}

/// Venice's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VeniceExtras {
    /// The parameters Venice applied, echoed.
    pub venice_parameters: Option<VeniceParametersEcho>,
    /// What the request cost.
    pub cost: Option<VeniceCost>,
}

/// The `venice_parameters` block a reply echoes: the values Venice applied
/// and its reply-only fields.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct VeniceParametersEcho {
    /// The character conversed with.
    #[serde(default)]
    pub character_slug: Option<String>,
    /// Whether `<think>` blocks were stripped.
    #[serde(default)]
    pub strip_thinking_response: Option<bool>,
    /// Whether reasoning was off.
    #[serde(default)]
    pub disable_thinking: Option<bool>,
    /// The web-search mode.
    #[serde(default)]
    pub enable_web_search: Option<WebSearchMode>,
    /// Whether prompt URLs were scraped.
    #[serde(default)]
    pub enable_web_scraping: Option<bool>,
    /// Whether xAI's search was used.
    #[serde(default)]
    pub enable_x_search: Option<bool>,
    /// Whether sources were cited.
    #[serde(default)]
    pub enable_web_citations: Option<bool>,
    /// Whether search results were streamed.
    #[serde(default)]
    pub include_search_results_in_stream: Option<bool>,
    /// Whether search results came back as documents.
    #[serde(default)]
    pub return_search_results_as_documents: Option<bool>,
    /// Whether Venice's system prompt was included.
    #[serde(default)]
    pub include_venice_system_prompt: Option<bool>,
    /// Whether end-to-end encryption applied.
    #[serde(default)]
    pub enable_e2ee: Option<bool>,
    /// The sources web search consulted.
    #[serde(default)]
    pub web_search_citations: Option<Vec<WebSearchCitation>>,
}

/// A source Venice's web search consulted.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct WebSearchCitation {
    /// The page title.
    #[serde(default)]
    pub title: Option<String>,
    /// The source URL.
    #[serde(default)]
    pub url: Option<String>,
    /// The page content Venice extracted.
    #[serde(default)]
    pub content: Option<String>,
    /// The publication date, when Venice found one.
    #[serde(default)]
    pub date: Option<String>,
}

/// What Venice charged for a request.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
pub struct VeniceCost {
    /// In USD credits.
    #[serde(default)]
    pub usd: Option<f64>,
    /// In DIEM.
    #[serde(default)]
    pub diem: Option<f64>,
}

impl ReplyExtras for VeniceExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            venice_parameters: reply_field(raw, "/venice_parameters")?,
            cost: reply_field(raw, "/cost")?,
        })
    }
}

#[cfg(test)]
mod tests;
