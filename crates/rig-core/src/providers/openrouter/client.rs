use crate::client::{self, BearerAuth, DebugExt, Provider};
use http::HeaderValue;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ================================================================
// Main openrouter Client
// ================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExtBuilder;

type OpenRouterApiKey = BearerAuth;

pub type Client<H> = client::Client<OpenRouterExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenRouterExtBuilder, OpenRouterApiKey, H>;

impl Provider for OpenRouterExt {
    type Builder = OpenRouterExtBuilder;

    const VERIFY_PATH: &'static str = "/key";
}

client::impl_capabilities!(
    OpenRouterExt,
    completion = super::CompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
    transcription = super::transcription::TranscriptionModel<H>,
    model_listing = super::OpenRouterModelLister<H>,
    audio_generation = super::audio_generation::AudioGenerationModel<H>,
);

impl DebugExt for OpenRouterExt {}

client::impl_default_provider_builder!(
    OpenRouterExtBuilder => OpenRouterExt,
    api_key = OpenRouterApiKey,
    base_url = OPENROUTER_API_BASE_URL,
);

client::impl_provider_from_env!(
    OpenRouterExt,
    input = OpenRouterApiKey,
    api_key_env = "OPENROUTER_API_KEY",
);

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub cost: f64,
    /// OpenAI-compatible prompt-token details, returned by OpenRouter when a
    /// provider reports cache activity (Anthropic with cache_control, OpenAI
    /// with server-side automatic caching).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// OpenAI-compatible completion-token breakdown. OpenRouter includes full
    /// usage accounting on every response, so a reasoning-capable route
    /// reports here how much of `completion_tokens` went to hidden reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Prompt-token breakdown reported by OpenRouter for cached requests.
// `usize` matches the parent `Usage` struct in this module; the streaming counterpart
// in `streaming.rs` uses `u32` to match its own parent.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served from cache (cache hit).
    #[serde(default)]
    pub cached_tokens: usize,
    /// Tokens written to cache on this call (cache miss that populated the cache).
    #[serde(default)]
    pub cache_write_tokens: usize,
}

/// Completion-token breakdown reported by OpenRouter.
///
/// Only the reasoning share is modeled: it is the one entry rig's normalized
/// [`crate::completion::Usage`] has a slot for, and OpenRouter documents usage
/// accounting as always present (on the final SSE message when streaming).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens the upstream spent on hidden reasoning, counted inside
    /// `completion_tokens`.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        let (cached_input, cache_creation) =
            value.prompt_tokens_details.as_ref().map_or((0, 0), |d| {
                (d.cached_tokens as u64, d.cache_write_tokens as u64)
            });
        crate::completion::Usage {
            input_tokens: value.prompt_tokens as u64,
            // Reported completion tokens, falling back to saturating
            // total - prompt for gateways that omit the field (it
            // deserializes to 0).
            output_tokens: if value.completion_tokens > 0 {
                value.completion_tokens as u64
            } else {
                value.total_tokens.saturating_sub(value.prompt_tokens) as u64
            },
            total_tokens: value.total_tokens as u64,
            cached_input_tokens: cached_input,
            cache_creation_input_tokens: cache_creation,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: value
                .completion_tokens_details
                .as_ref()
                .map_or(0, |d| d.reasoning_tokens as u64),
        }
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        crate::completion::Usage::from(&value)
    }
}
impl<ApiKey, H> client::ClientBuilder<OpenRouterExtBuilder, ApiKey, H> {
    /// Attach OpenRouter app-identification headers (`X-OpenRouter-Title` and `HTTP-Referer`)
    /// to every request made by this client. `title` appears in the dashboard activity feed
    /// and rankings page; `url` is the primary app identifier required to create an app page
    /// on OpenRouter. Invalid (non-ASCII) values are silently skipped.
    pub fn with_app_identity(mut self, title: impl AsRef<str>, url: impl AsRef<str>) -> Self {
        if let Ok(val) = HeaderValue::from_str(title.as_ref()) {
            self.headers_mut().insert(
                http::header::HeaderName::from_static("x-openrouter-title"),
                val,
            );
        }
        if let Ok(val) = HeaderValue::from_str(url.as_ref()) {
            self.headers_mut()
                .insert(http::header::HeaderName::from_static("http-referer"), val);
        }
        self
    }

    /// Assign this app to up to two OpenRouter marketplace categories via the
    /// `X-OpenRouter-Categories` header. Categories must be lowercase and hyphen-separated
    /// (e.g. `"cli-agent"`, `"ide-extension"`). OpenRouter silently ignores unrecognized
    /// categories. Extra categories beyond the first two are not sent. Invalid (non-ASCII)
    /// values are silently skipped.
    pub fn with_app_categories<S>(mut self, categories: &[S]) -> Self
    where
        S: AsRef<str>,
    {
        let joined = categories
            .iter()
            .take(2)
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(",");
        if !joined.is_empty()
            && let Ok(val) = HeaderValue::from_str(&joined)
        {
            self.headers_mut().insert(
                http::header::HeaderName::from_static("x-openrouter-categories"),
                val,
            );
        }
        self
    }
}

#[cfg(test)]
mod tests;
