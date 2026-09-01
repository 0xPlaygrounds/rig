use crate::client::{self, ApiKey, DebugExt, Provider, ProviderBuilder, Transport};
use crate::http_client::{self};
use crate::providers::gemini::cached_content::CachedContentClient;
use crate::providers::gemini::model_listing::{GeminiInteractionsModelLister, GeminiModelLister};
use serde::Deserialize;
use std::fmt::Debug;

// ================================================================
// Google Gemini Client
// ================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Provider extension for the Gemini GenerateContent API.
#[derive(Debug, Default, Clone)]
pub struct GeminiExt {
    api_key: String,
}

/// Builder marker for the Gemini GenerateContent client.
#[derive(Debug, Default, Clone)]
pub struct GeminiBuilder;

/// Provider extension for the Gemini Interactions API.
#[derive(Debug, Default, Clone)]
pub struct GeminiInteractionsExt {
    api_key: String,
}

/// Builder marker for the Gemini Interactions client.
#[derive(Debug, Default, Clone)]
pub struct GeminiInteractionsBuilder;

/// Wrapper type for Gemini API keys.
pub struct GeminiApiKey(String);

impl<S> From<S> for GeminiApiKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

/// Gemini GenerateContent client.
pub type Client<H> = client::Client<GeminiExt, H>;
/// Builder for the Gemini GenerateContent client.
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<GeminiBuilder, GeminiApiKey, H>;
/// Gemini Interactions API client.
pub type InteractionsClient<H> = client::Client<GeminiInteractionsExt, H>;

impl ApiKey for GeminiApiKey {}

impl DebugExt for GeminiExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::once(("api_key", (&"******") as &dyn Debug))
    }
}

impl DebugExt for GeminiInteractionsExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::once(("api_key", (&"******") as &dyn Debug))
    }
}

impl Provider for GeminiExt {
    type Builder = GeminiBuilder;

    const VERIFY_PATH: &'static str = "/v1beta/models";

    fn build_uri(&self, base_url: &str, path: &str, transport: Transport) -> String {
        let trimmed = path.trim_start_matches('/');
        let separator = if trimmed.contains('?') { "&" } else { "?" };

        match transport {
            Transport::Sse => format!(
                "{base_url}/{trimmed}{separator}alt=sse&key={}",
                self.api_key
            ),
            _ => format!("{base_url}/{trimmed}{separator}key={}", self.api_key),
        }
    }
}

impl Provider for GeminiInteractionsExt {
    type Builder = GeminiInteractionsBuilder;

    const VERIFY_PATH: &'static str = "/v1beta/models";

    fn build_uri(&self, base_url: &str, path: &str, transport: Transport) -> String {
        let trimmed = path.trim_start_matches('/');
        match transport {
            Transport::Sse => {
                if trimmed.contains('?') {
                    format!("{base_url}/{trimmed}&alt=sse")
                } else {
                    format!("{base_url}/{trimmed}?alt=sse")
                }
            }
            _ => format!("{base_url}/{trimmed}"),
        }
    }

    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req.header("x-goog-api-key", self.api_key.clone()))
    }
}

client::impl_capabilities!(
    GeminiExt,
    completion = super::completion::CompletionModel<H>,
    embeddings = super::embedding::EmbeddingModel<H>,
    transcription = super::transcription::TranscriptionModel<H>,
    model_listing = GeminiModelLister<H>,
    image_generation = super::image_generation::ImageGenerationModel<H>,
);

client::impl_capabilities!(
    GeminiInteractionsExt,
    completion = super::interactions_api::InteractionsCompletionModel<H>,
    embeddings = super::embedding::EmbeddingModel<H>,
    transcription = super::transcription::TranscriptionModel<H>,
    model_listing = GeminiInteractionsModelLister<H>,
);

impl ProviderBuilder for GeminiBuilder {
    type Extension<H>
        = GeminiExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = GeminiApiKey;

    const BASE_URL: &'static str = GEMINI_API_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(GeminiExt {
            api_key: builder.get_api_key().0.clone(),
        })
    }
}

impl ProviderBuilder for GeminiInteractionsBuilder {
    type Extension<H>
        = GeminiInteractionsExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = GeminiApiKey;

    const BASE_URL: &'static str = GEMINI_API_BASE_URL;

    fn build<H>(
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(GeminiInteractionsExt {
            api_key: builder.get_api_key().0.clone(),
        })
    }
}

client::impl_provider_from_env!(
    GeminiExt,
    input = GeminiApiKey,
    api_key_env = "GEMINI_API_KEY",
);
client::impl_provider_from_env!(
    GeminiInteractionsExt,
    input = GeminiApiKey,
    api_key_env = "GEMINI_API_KEY",
);

impl<H> Client<H> {
    /// Client for Gemini's explicit context cache (`cachedContents`).
    ///
    /// Explicit caching is a different feature from the implicit prefix caching
    /// that happens automatically: it hits on the first request and across
    /// unrelated conversations, at the cost of billing storage per token-hour.
    /// See [`crate::providers::gemini::cached_content`] for when each pays.
    pub fn cached_contents(&self) -> CachedContentClient<H>
    where
        H: Clone,
    {
        CachedContentClient::new(self.clone())
    }

    /// Create an Interactions API client from this GenerateContent client.
    pub fn interactions_api(self) -> InteractionsClient<H> {
        let api_key = self.ext().api_key.clone();
        self.with_ext(GeminiInteractionsExt { api_key })
    }
}

impl<H> InteractionsClient<H> {
    /// Create a GenerateContent API client from this Interactions client.
    pub fn generate_content_api(self) -> Client<H> {
        let api_key = self.ext().api_key.clone();
        self.with_ext(GeminiExt { api_key })
    }
}

/// Error response payload returned by Gemini.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// Structured error details.
    pub error: ApiError,
}

/// Error details returned in a Gemini API error response.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// Human-readable description of the error.
    pub message: String,
}

/// Wrapper for successful or error Gemini API responses.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    // Untagged variants are tried in order, and some Gemini success response
    // types contain only defaulted or optional fields that accept error objects.
    Err(ApiErrorResponse),
    Ok(T),
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests;
