use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    completion::GetTokenUsage,
    http_client,
};
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

pub type Client<H = reqwest::Client> = client::Client<OpenRouterExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenRouterExtBuilder, OpenRouterApiKey, H>;

impl Provider for OpenRouterExt {
    type Builder = OpenRouterExtBuilder;

    const VERIFY_PATH: &'static str = "/key";
}

impl<H> Capabilities<H> for OpenRouterExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Capable<super::transcription::TranscriptionModel<H>>;
    type ModelListing = Capable<super::OpenRouterModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H>>;
    type Rerank = Nothing;
}

impl DebugExt for OpenRouterExt {}

impl ProviderBuilder for OpenRouterExtBuilder {
    type Extension<H>
        = OpenRouterExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = OpenRouterApiKey;

    const BASE_URL: &'static str = OPENROUTER_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(OpenRouterExt)
    }
}

impl ProviderClient for Client {
    type Input = OpenRouterApiKey;
    type Error = crate::client::ProviderClientError;

    /// Create a new openrouter client from the `OPENROUTER_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("OPENROUTER_API_KEY")?;

        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
}

/// Prompt-token breakdown reported by OpenRouter for cached requests.
// `usize` matches the parent `Usage` struct in this module; the streaming counterpart
// in `streaming.rs` uses `u32` to match its own parent.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served from cache (cache hit).
    #[serde(default)]
    pub cached_tokens: usize,
    /// Tokens written to cache on this call (cache miss that populated the cache).
    #[serde(default)]
    pub cache_write_tokens: usize,
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

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        let (cached_input, cache_creation) = self
            .prompt_tokens_details
            .as_ref()
            .map(|d| (d.cached_tokens as u64, d.cache_write_tokens as u64))
            .unwrap_or((0, 0));
        crate::completion::Usage {
            input_tokens: self.prompt_tokens as u64,
            output_tokens: self.completion_tokens as u64,
            total_tokens: self.total_tokens as u64,
            cached_input_tokens: cached_input,
            cache_creation_input_tokens: cache_creation,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
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
            .map(|c| c.as_ref())
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
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::openrouter::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_with_app_identity_sets_headers() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_identity("My App", "https://myapp.example.com")
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        let headers = client.headers();
        assert_eq!(
            headers
                .get("x-openrouter-title")
                .and_then(|v| v.to_str().ok()),
            Some("My App"),
        );
        assert_eq!(
            headers.get("http-referer").and_then(|v| v.to_str().ok()),
            Some("https://myapp.example.com"),
        );
    }

    #[test]
    fn test_without_app_identity_no_extra_headers() {
        let client = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        let headers = client.headers();
        assert!(headers.get("x-openrouter-title").is_none());
        assert!(headers.get("http-referer").is_none());
    }

    #[test]
    fn test_with_app_categories_sets_header() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&["cli-agent", "ide-extension"])
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert_eq!(
            client
                .headers()
                .get("x-openrouter-categories")
                .and_then(|v| v.to_str().ok()),
            Some("cli-agent,ide-extension"),
        );
    }

    #[test]
    fn test_with_app_categories_sends_at_most_two_categories() {
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&["cli-agent", "ide-extension", "chat"])
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert_eq!(
            client
                .headers()
                .get("x-openrouter-categories")
                .and_then(|v| v.to_str().ok()),
            Some("cli-agent,ide-extension"),
        );
    }

    #[test]
    fn test_with_app_categories_empty_list_no_header() {
        let empty: [&str; 0] = [];
        let client = crate::providers::openrouter::Client::builder()
            .with_app_categories(&empty)
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert!(client.headers().get("x-openrouter-categories").is_none());
    }

    #[test]
    fn test_without_app_categories_no_header() {
        let client = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");

        assert!(client.headers().get("x-openrouter-categories").is_none());
    }
}
