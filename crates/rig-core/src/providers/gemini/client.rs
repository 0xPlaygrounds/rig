#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::client::{
    self, ApiKey, HasCompletion, HasEmbeddings, HasModelListing, HasTranscription, ModelTransport,
    Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::gemini::cached_content::CachedContentClient;
use crate::providers::gemini::model_listing::{GeminiInteractionsModelLister, GeminiModelLister};
use serde::Deserialize;
use std::fmt::Debug;

// ================================================================
// Google Gemini Client
// ================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// The Gemini GenerateContent API provider. Authenticates through the
/// `key` query parameter, so the key lives here rather than in a header.
#[derive(Default, Clone)]
pub struct Gemini {
    api_key: String,
}

impl Debug for Gemini {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemini")
            .field("api_key", &"******")
            .finish()
    }
}

/// The Gemini Interactions API provider. Authenticates through the
/// per-request `x-goog-api-key` header.
#[derive(Default, Clone)]
pub struct GeminiInteractions {
    api_key: String,
}

impl Debug for GeminiInteractions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiInteractions")
            .field("api_key", &"******")
            .finish()
    }
}

/// Wrapper type for Gemini API keys.
#[derive(Clone)]
pub struct GeminiApiKey(String);

impl Debug for GeminiApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GeminiApiKey(<redacted>)")
    }
}

impl<S> From<S> for GeminiApiKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

/// Gemini GenerateContent client.
pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Gemini, H>;
/// Builder for the Gemini GenerateContent client.
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Gemini, H>;
/// Gemini Interactions API client.
pub type InteractionsClient<H = crate::http_client::BoxedHttpClient> =
    client::Client<GeminiInteractions, H>;

impl ApiKey for GeminiApiKey {}

impl Provider for Gemini {
    const NAME: &'static str = "gcp.gemini";
    const BASE_URL: &'static str = GEMINI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/v1beta/models";
    type ApiKey = GeminiApiKey;
    type Config = ();
    type EnvInput = GeminiApiKey;

    fn build(_: (), api_key: &GeminiApiKey) -> http_client::Result<Self> {
        Ok(Gemini {
            api_key: api_key.0.clone(),
        })
    }

    /// Appends the API key as the `key` query parameter. Streaming callers
    /// put `alt=sse` in the path themselves.
    fn build_uri(&self, base_url: &str, path: &str) -> String {
        let trimmed = path.trim_start_matches('/');
        let separator = if trimmed.contains('?') { "&" } else { "?" };

        format!("{base_url}/{trimmed}{separator}key={}", self.api_key)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("GEMINI_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(input: GeminiApiKey, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl Provider for GeminiInteractions {
    const NAME: &'static str = "gcp.gemini";
    const BASE_URL: &'static str = GEMINI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/v1beta/models";
    type ApiKey = GeminiApiKey;
    type Config = ();
    type EnvInput = GeminiApiKey;

    fn build(_: (), api_key: &GeminiApiKey) -> http_client::Result<Self> {
        Ok(GeminiInteractions {
            api_key: api_key.0.clone(),
        })
    }

    fn build_uri(&self, base_url: &str, path: &str) -> String {
        format!("{base_url}/{}", path.trim_start_matches('/'))
    }

    fn prepare(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req.header("x-goog-api-key", self.api_key.clone()))
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<InteractionsClient<H>> {
        InteractionsClient::from_env_api_key("GEMINI_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(
        input: GeminiApiKey,
        http: H,
    ) -> ProviderClientResult<InteractionsClient<H>> {
        InteractionsClient::new_with(input, http)
    }
}

impl HasCompletion for Gemini {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Gemini {
    type Model<H>
        = super::embedding::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::embedding::EmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for Gemini {
    type Model<H>
        = super::transcription::TranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::transcription::TranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for Gemini {
    type Lister<H>
        = GeminiModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        GeminiModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for Gemini {
    type Model<H>
        = super::image_generation::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::image_generation::ImageGenerationModel::new(client.clone(), model)
    }
}

impl HasCompletion for GeminiInteractions {
    type Model<H>
        = super::interactions_api::InteractionsCompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(
        client: &InteractionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::interactions_api::InteractionsCompletionModel::new(client.clone(), model)
    }
}

impl HasModelListing for GeminiInteractions {
    type Lister<H>
        = GeminiInteractionsModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &InteractionsClient<H>) -> Self::Lister<H> {
        GeminiInteractionsModelLister::new(client.clone())
    }
}

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
        let api_key = self.provider().api_key.clone();
        self.with_provider(GeminiInteractions { api_key })
    }
}

impl<H> InteractionsClient<H> {
    /// Create a GenerateContent API client from this Interactions client.
    pub fn generate_content_api(self) -> Client<H> {
        let api_key = self.provider().api_key.clone();
        self.with_provider(Gemini { api_key })
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
