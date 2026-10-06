//! Perplexity's typed reply extras. Perplexity has no typed request
//! option: its Sonar Chat API was retired on 2026-09-27, so its request
//! fields are left to `additional_params` and its replies are read from
//! recordings.
//!
//! ```
//! use rig_core::completion::CompletionResponse;
//! use rig_core::providers::perplexity::extension::Perplexity;
//!
//! fn sources(reply: &CompletionResponse) -> Vec<String> {
//!     reply
//!         .extras::<Perplexity>()
//!         .and_then(Result::ok)
//!         .and_then(|extras| extras.citations)
//!         .unwrap_or_default()
//! }
//! # let _ = sources;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Perplexity's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Perplexity;

impl ProviderExtension for Perplexity {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = PerplexityOptions;
    type Extras = PerplexityExtras;
}

/// Perplexity's request options: none.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PerplexityOptions {}

impl PerplexityOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExtensionOptions for PerplexityOptions {}

/// Perplexity's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerplexityExtras {
    /// The URLs the answer cites, in citation order.
    pub citations: Option<Vec<String>>,
    /// The search results the answer drew on.
    pub search_results: Option<Vec<SearchResult>>,
    /// The images the search returned.
    pub images: Option<Vec<Value>>,
    /// Follow-up questions Perplexity suggests.
    pub related_questions: Option<Vec<String>>,
    /// What the request cost.
    pub cost: Option<PerplexityCost>,
    /// How much search context went into the prompt.
    pub search_context_size: Option<String>,
}

/// One search result.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct SearchResult {
    /// The page title.
    #[serde(default)]
    pub title: Option<String>,
    /// The page URL.
    #[serde(default)]
    pub url: Option<String>,
    /// The publication date.
    #[serde(default)]
    pub date: Option<String>,
    /// When the page last changed.
    #[serde(default)]
    pub last_updated: Option<String>,
    /// The excerpt the answer drew on.
    #[serde(default)]
    pub snippet: Option<String>,
    /// Where the result came from, such as `web`.
    #[serde(default)]
    pub source: Option<String>,
}

/// What a Perplexity request cost, in USD.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
pub struct PerplexityCost {
    /// For the prompt tokens.
    #[serde(default)]
    pub input_tokens_cost: Option<f64>,
    /// For the completion tokens.
    #[serde(default)]
    pub output_tokens_cost: Option<f64>,
    /// Per request.
    #[serde(default)]
    pub request_cost: Option<f64>,
    /// In all.
    #[serde(default)]
    pub total_cost: Option<f64>,
}

impl ReplyExtras for PerplexityExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            citations: reply_field(raw, "/citations")?,
            search_results: reply_field(raw, "/search_results")?,
            images: reply_field(raw, "/images")?,
            related_questions: reply_field(raw, "/related_questions")?,
            cost: reply_field(raw, "/usage/cost")?,
            search_context_size: reply_field(raw, "/usage/search_context_size")?,
        })
    }
}

#[cfg(test)]
mod tests;
