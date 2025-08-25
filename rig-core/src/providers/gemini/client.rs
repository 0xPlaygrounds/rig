use super::{
    completion::CompletionModel, embedding::EmbeddingModel, transcription::TranscriptionModel,
};
use crate::client::{
    ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient,
    VerifyClient, VerifyError, impl_conversion_traits,
};
use crate::{
    Embed,
    embeddings::{self},
};
use serde::Deserialize;

// ================================================================
// Google Gemini Client
// ================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: GEMINI_API_BASE_URL,
            http_client: None,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            reqwest::Client::builder().build()?
        };

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client,
        })
    }
}
#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    default_headers: reqwest::header::HeaderMap,
    http_client: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("default_headers", &self.default_headers)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client {
    /// Create a new Google Gemini client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::gemini::{ClientBuilder, self};
    ///
    /// // Initialize the Google Gemini client
    /// let gemini_client = Client::builder("your-google-gemini-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Google Gemini client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Gemini client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        // API key gets inserted as query param - no need to add bearer auth or headers
        let url = format!("{}/{}?key={}", self.base_url, path, self.api_key).replace("//", "/");

        tracing::debug!("POST {}/{}?key={}", self.base_url, path, "****");
        self.http_client
            .post(url)
            .headers(self.default_headers.clone())
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        // API key gets inserted as query param - no need to add bearer auth or headers
        let url = format!("{}/{}?key={}", self.base_url, path, self.api_key).replace("//", "/");

        tracing::debug!("GET {}/{}?key={}", self.base_url, path, "****");
        self.http_client
            .get(url)
            .headers(self.default_headers.clone())
    }

    pub(crate) fn post_sse(&self, path: &str) -> reqwest::RequestBuilder {
        let url =
            format!("{}/{}?alt=sse&key={}", self.base_url, path, self.api_key).replace("//", "/");

        tracing::debug!("POST {}/{}?alt=sse&key={}", self.base_url, path, "****");
        self.http_client
            .post(url)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client {
    /// Create a new Google Gemini client from the `GEMINI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        Self::new(&api_key)
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
    /// Gemini-specific parameters can be set using the [GenerationConfig](crate::providers::gemini::completion::gemini_api_types::GenerationConfig) struct.
    /// [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig::providers::gemini::{Client, self};
    ///
    /// // Initialize the Google Gemini client
    /// let gemini = Client::new("your-google-gemini-api-key");
    ///
    /// let embedding_model = gemini.embedding_model(gemini::embedding::EMBEDDING_GECKO_001);
    /// ```
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::gemini::{Client, self};
    ///
    /// // Initialize the Google Gemini client
    /// let gemini = Client::new("your-google-gemini-api-key");
    ///
    /// let embedding_model = gemini.embedding_model_with_ndims("model-unknown-to-rig", 1024);
    /// ```
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::gemini::{Client, self};
    ///
    /// // Initialize the Google Gemini client
    /// let gemini = Client::new("your-google-gemini-api-key");
    ///
    /// let embeddings = gemini.embeddings(gemini::embedding::EMBEDDING_GECKO_001)
    ///     .simple_document("doc0", "Hello, world!")
    ///     .simple_document("doc1", "Goodbye, world!")
    ///     .build()
    ///     .await
    ///     .expect("Failed to embed documents");
    /// ```
    fn embeddings<D: Embed>(
        &self,
        model: &str,
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

impl TranscriptionClient for Client {
    type TranscriptionModel = TranscriptionModel;

    /// Create a transcription model with the given name.
    /// Gemini-specific parameters can be set using the [GenerationConfig](crate::providers::gemini::completion::gemini_api_types::GenerationConfig) struct.
    /// [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let response = self.get("/v1beta/models").send().await?;
        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::FORBIDDEN => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            _ => {
                response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsImageGeneration,
    AsAudioGeneration for Client
);

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
