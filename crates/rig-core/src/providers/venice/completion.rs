//! Venice chat model identifiers, request extensions, and response metadata.
//!
//! ```
//! use rig_core::providers::venice::{VeniceParameters, WebSearchMode};
//! let params = VeniceParameters::new()
//!     .enable_web_search(WebSearchMode::Auto)
//!     .into_additional_params();
//! assert!(params.get("venice_parameters").is_some());
//! ```

use serde::{Deserialize, Serialize};

use crate::providers::openai;

/// Identifier for `zai-org-glm-4.7`.
pub const GLM_4_7: &str = "zai-org-glm-4.7";
/// `zai-org-glm-5-2`
pub const GLM_5_2: &str = "zai-org-glm-5-2";
/// Identifier for `qwen3-5-9b`.
pub const QWEN3_5_9B: &str = "qwen3-5-9b";
/// `qwen3-5-397b-a17b`
pub const QWEN3_5_397B_A17B: &str = "qwen3-5-397b-a17b";
/// Identifier for `qwen3-235b-a22b-thinking-2507`.
pub const QWEN3_235B_A22B_THINKING: &str = "qwen3-235b-a22b-thinking-2507";
/// Identifier for `qwen3-vl-235b-a22b`.
pub const QWEN3_VL_235B_A22B: &str = "qwen3-vl-235b-a22b";
/// Identifier for `qwen3-coder-480b-a35b-instruct-turbo`.
pub const QWEN3_CODER_480B: &str = "qwen3-coder-480b-a35b-instruct-turbo";
/// Identifier for `venice-uncensored-1-2`.
pub const VENICE_UNCENSORED_1_2: &str = "venice-uncensored-1-2";
/// `gemini-3-6-flash`
pub const GEMINI_3_6_FLASH: &str = "gemini-3-6-flash";
/// `grok-4-6`
pub const GROK_4_6: &str = "grok-4-6";
/// `mistral-small-2603`
pub const MISTRAL_SMALL_2603: &str = "mistral-small-2603";
/// `mistral-small-3-2-24b-instruct`
pub const MISTRAL_SMALL_3_2_24B: &str = "mistral-small-3-2-24b-instruct";

/// How Venice's web search behaves for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchMode {
    /// Never search.
    Off,
    /// Always search.
    On,
    /// Let the model decide.
    Auto,
}

/// Venice's `venice_parameters` request block.
///
/// Merge into [`additional_params`](crate::completion::CompletionRequest::additional_params)
/// using [`Self::into_additional_params`]:
///
/// ```no_run
/// use rig_core::completion::CompletionRequestBuilder;
/// use rig_core::providers::venice::{VeniceParameters, WebSearchMode};
///
/// let request = CompletionRequestBuilder::unbound("Summarize today's Rust news.")
///     .additional_params(
///         VeniceParameters::new()
///             .enable_web_search(WebSearchMode::On)
///             .enable_web_citations(true)
///             .into_additional_params(),
///     )
///     .build();
/// ```
///
/// Every field is optional; omitted fields are left to Venice's own defaults
/// (notably `include_venice_system_prompt`, which Venice defaults to `true`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VeniceParameters {
    /// Public character slug to converse with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_slug: Option<String>,
    /// Strip `<think>` blocks from the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strip_thinking_response: Option<bool>,
    /// Disable reasoning on models that support it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_thinking: Option<bool>,
    /// Web-search mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_web_search: Option<WebSearchMode>,
    /// Scrape URLs found in the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_web_scraping: Option<bool>,
    /// Use xAI's native search on Grok models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_x_search: Option<bool>,
    /// Emit `[REF]`-style source citations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_web_citations: Option<bool>,
    /// Include search results in the stream (experimental).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_search_results_in_stream: Option<bool>,
    /// Return search results as tool-call documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_search_results_as_documents: Option<bool>,
    /// Include Venice's default system prompt (Venice defaults to `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_venice_system_prompt: Option<bool>,
    /// Prompt-cache routing hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

impl VeniceParameters {
    /// An empty parameter block; every field falls back to Venice's default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Converse with a public Venice character.
    pub fn character_slug(mut self, slug: impl Into<String>) -> Self {
        self.character_slug = Some(slug.into());
        self
    }

    /// Strip `<think>` blocks from the response.
    pub fn strip_thinking_response(mut self, strip: bool) -> Self {
        self.strip_thinking_response = Some(strip);
        self
    }

    /// Disable reasoning on models that support it.
    pub fn disable_thinking(mut self, disable: bool) -> Self {
        self.disable_thinking = Some(disable);
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

    /// Use xAI's native search on Grok models.
    pub fn enable_x_search(mut self, enable: bool) -> Self {
        self.enable_x_search = Some(enable);
        self
    }

    /// Emit `[REF]`-style source citations.
    pub fn enable_web_citations(mut self, enable: bool) -> Self {
        self.enable_web_citations = Some(enable);
        self
    }

    /// Include search results in the stream (experimental).
    pub fn include_search_results_in_stream(mut self, include: bool) -> Self {
        self.include_search_results_in_stream = Some(include);
        self
    }

    /// Return search results as tool-call documents.
    pub fn return_search_results_as_documents(mut self, as_documents: bool) -> Self {
        self.return_search_results_as_documents = Some(as_documents);
        self
    }

    /// Include Venice's default system prompt.
    pub fn include_venice_system_prompt(mut self, include: bool) -> Self {
        self.include_venice_system_prompt = Some(include);
        self
    }

    /// Set the prompt-cache routing hint.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Wrap this block in the `{"venice_parameters": …}` object Rig merges
    /// into the request body through `additional_params`.
    pub fn into_additional_params(self) -> serde_json::Value {
        serde_json::json!({ "venice_parameters": self })
    }
}

/// A web-search source Venice consulted for a completion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebSearchCitation {
    /// Page title.
    #[serde(default)]
    pub title: String,
    /// Source URL.
    #[serde(default)]
    pub url: String,
    /// Extracted page content, as Venice returned it.
    #[serde(default)]
    pub content: String,
    /// Publication date, empty when Venice could not determine one.
    #[serde(default)]
    pub date: String,
}

/// Venice's resolved `venice_parameters` block, echoed on every response.
///
/// The requested fields come back with the values Venice actually applied,
/// alongside the response-only fields below.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VeniceParametersEcho {
    /// The parameters Venice resolved for this request.
    #[serde(flatten)]
    pub parameters: VeniceParameters,
    /// Whether end-to-end encryption applied to this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_e2ee: Option<bool>,
    /// Sources consulted when web search ran; empty otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub web_search_citations: Vec<WebSearchCitation>,
}

/// What Venice charged for a request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Cost in USD credits.
    #[serde(default)]
    pub usd: f64,
    /// Cost in DIEM.
    #[serde(default)]
    pub diem: f64,
}

/// Typed chat payload including resolved parameters and request cost. Deserialize
/// from [`crate::completion::CompletionResponse::raw`] to inspect these fields.
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// The OpenAI-compatible portion of the payload.
    #[serde(flatten)]
    pub openai: openai::CompletionResponse,
    /// Venice's resolved parameter block, including web-search citations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub venice_parameters: Option<VeniceParametersEcho>,
    /// What Venice charged for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
}

impl CompletionResponse {
    /// The web-search sources Venice consulted, empty when search did not run.
    pub fn web_search_citations(&self) -> &[WebSearchCitation] {
        self.venice_parameters
            .as_ref()
            .map(|parameters| parameters.web_search_citations.as_slice())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests;
