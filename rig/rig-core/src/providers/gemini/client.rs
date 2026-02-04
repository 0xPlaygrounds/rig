#[cfg(feature = "image")]
use crate::client::Nothing;
use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, Provider, ProviderBuilder, ProviderClient,
    Transport,
};
use crate::http_client;
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
pub type Client<H = reqwest::Client> = client::Client<GeminiExt, H>;
/// Builder for the Gemini GenerateContent client.
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<GeminiBuilder, GeminiApiKey, H>;
/// Gemini Interactions API client.
pub type InteractionsClient<H = reqwest::Client> = client::Client<GeminiInteractionsExt, H>;

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

    fn build<H>(
        builder: &client::ClientBuilder<Self::Builder, GeminiApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self {
            api_key: builder.get_api_key().0.clone(),
        })
    }

    fn build_uri(&self, base_url: &str, path: &str, transport: Transport) -> String {
        match transport {
            Transport::Sse => {
                format!(
                    "{}/{}?alt=sse&key={}",
                    base_url,
                    path.trim_start_matches('/'),
                    self.api_key
                )
            }
            _ => {
                format!(
                    "{}/{}?key={}",
                    base_url,
                    path.trim_start_matches('/'),
                    self.api_key
                )
            }
        }
    }
}

impl Provider for GeminiInteractionsExt {
    type Builder = GeminiInteractionsBuilder;

    const VERIFY_PATH: &'static str = "/v1beta/models";

    fn build<H>(
        builder: &client::ClientBuilder<Self::Builder, GeminiApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self {
            api_key: builder.get_api_key().0.clone(),
        })
    }

    fn build_uri(&self, base_url: &str, path: &str, transport: Transport) -> String {
        let trimmed = path.trim_start_matches('/');
        match transport {
            Transport::Sse => {
                if trimmed.contains('?') {
                    format!("{}/{}&alt=sse", base_url, trimmed)
                } else {
                    format!("{}/{}?alt=sse", base_url, trimmed)
                }
            }
            _ => format!("{}/{}", base_url, trimmed),
        }
    }

    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req.header("x-goog-api-key", self.api_key.clone()))
    }
}

impl<H> Capabilities<H> for GeminiExt {
    type Completion = Capable<super::completion::CompletionModel>;
    type Embeddings = Capable<super::embedding::EmbeddingModel>;
    type Transcription = Capable<super::transcription::TranscriptionModel>;

    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl<H> Capabilities<H> for GeminiInteractionsExt {
    type Completion = Capable<super::interactions_api::InteractionsCompletionModel<H>>;
    type Embeddings = Capable<super::embedding::EmbeddingModel>;
    type Transcription = Capable<super::transcription::TranscriptionModel>;

    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl ProviderBuilder for GeminiBuilder {
    type Output = GeminiExt;
    type ApiKey = GeminiApiKey;

    const BASE_URL: &'static str = GEMINI_API_BASE_URL;
}

impl ProviderBuilder for GeminiInteractionsBuilder {
    type Output = GeminiInteractionsExt;
    type ApiKey = GeminiApiKey;

    const BASE_URL: &'static str = GEMINI_API_BASE_URL;
}

impl ProviderClient for Client {
    type Input = GeminiApiKey;

    /// Create a new Google Gemini client from the `GEMINI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        Self::new(api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

impl ProviderClient for InteractionsClient {
    type Input = GeminiApiKey;

    /// Create a new Google Gemini interactions client from the `GEMINI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        Self::new(api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

impl<H> Client<H> {
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
    pub message: String,
}

/// Wrapper for successful or error Gemini API responses.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
