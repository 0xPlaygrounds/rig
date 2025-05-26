use serde::Deserialize;

use super::{
    embedding::{EmbeddingModel, MISTRAL_EMBED},
    CompletionModel,
};
use crate::client::{CompletionClient, EmbeddingsClient, ProviderClient};
use crate::impl_conversion_traits;

const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

#[derive(Debug, Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, MISTRAL_API_BASE_URL)
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {api_key}")
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("Mistral reqwest client should build"),
        }
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }
}

impl ProviderClient for Client {
    /// Create a new Mistral client from the `MISTRAL_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self
    where
        Self: Sized,
    {
        let api_key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::mistral::{Client, self};
    ///
    /// // Initialize the Mistral client
    /// let mistral = Client::new("your-mistral-api-key");
    ///
    /// let codestral = mistral.completion_model(mistral::CODESTRAL);
    /// ```
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    ///
    /// # Example
    /// ```
    /// use rig::providers::mistral::{Client, self};
    ///
    /// // Initialize mistral client
    /// let mistral = Client::new("your-mistral-api-key");
    ///
    /// let embedding_model = mistral.embedding_model(mistral::MISTRAL_EMBED);
    /// ```
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        let ndims = match model {
            MISTRAL_EMBED => 1024,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl_conversion_traits!(AsTranscription, AsAudioGeneration, AsImageGeneration for Client);

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
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

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
