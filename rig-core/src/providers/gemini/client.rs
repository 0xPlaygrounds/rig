use super::{
    completion::CompletionModel, embedding::EmbeddingModel, transcription::TranscriptionModel,
};
use crate::client::{
    ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient, TranscriptionClient,
    VerifyClient, VerifyError, impl_conversion_traits,
};
use crate::http_client::{self, HttpClientExt};
use crate::wasm_compat::*;
use crate::{
    Embed,
    embeddings::{self},
};
use bytes::Bytes;
use serde::Deserialize;
use std::fmt::Debug;

// ================================================================
// Google Gemini Client
// ================================================================
const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: HttpClientExt,
{
    pub fn new(api_key: &'a str) -> ClientBuilder<'a, reqwest::Client> {
        ClientBuilder {
            api_key,
            base_url: GEMINI_API_BASE_URL,
            http_client: Default::default(),
        }
    }

    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url: GEMINI_API_BASE_URL,
            http_client,
        }
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U>
    where
        U: HttpClientExt,
    {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn build(self) -> Result<Client<T>, ClientBuilderError> {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client: self.http_client,
        })
    }
}
#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    default_headers: reqwest::header::HeaderMap,
    http_client: T,
}

impl<T> Debug for Client<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("default_headers", &self.default_headers)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt + Default,
{
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
    pub fn builder(api_key: &str) -> ClientBuilder<'_, T> {
        ClientBuilder::new_with_client(api_key, Default::default())
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
}

impl Client<reqwest::Client> {
    pub(crate) fn post_sse(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/{}?alt=sse&key={}",
            self.base_url,
            path.trim_start_matches('/'),
            self.api_key
        );

        tracing::debug!("POST {}/{}?alt=sse&key={}", self.base_url, path, "****");

        self.http_client
            .post(url)
            .headers(self.default_headers.clone())
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Builder {
        // API key gets inserted as query param - no need to add bearer auth or headers
        let url = format!(
            "{}/{}?key={}",
            self.base_url,
            path.trim_start_matches('/'),
            self.api_key
        );

        tracing::debug!("POST {}/{}?key={}", self.base_url, path, "****");
        let mut req = http_client::Request::post(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        req
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Builder {
        // API key gets inserted as query param - no need to add bearer auth or headers
        let url = format!(
            "{}/{}?key={}",
            self.base_url,
            path.trim_start_matches('/'),
            self.api_key
        );

        tracing::debug!("GET {}/{}?key={}", self.base_url, path, "****");

        let mut req = http_client::Request::get(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        req
    }

    pub(crate) async fn send<U, R>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<R>>>
    where
        U: Into<Bytes> + Send,
        R: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }
}

// NOTE: (@FayCarsons) This cannot be implemented for all T because `AsCompletion`/`CompletionModel` requires SSE
// which we are not able to implement for any `T: HttpClientExt` right now
impl ProviderClient for Client<reqwest::Client> {
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

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;

    /// Create a completion model with the given name.
    /// Gemini-specific parameters can be set using the [GenerationConfig](crate::providers::gemini::completion::gemini_api_types::GenerationConfig) struct.
    /// [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> EmbeddingsClient for Client<T>
where
    T: HttpClientExt + Clone + Debug + Default + 'static,
    Client<T>: CompletionClient,
{
    type EmbeddingModel = EmbeddingModel<T>;

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
    fn embedding_model(&self, model: &str) -> EmbeddingModel<T> {
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
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel<T> {
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
    ) -> embeddings::EmbeddingsBuilder<EmbeddingModel<T>, D> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

impl<T> TranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + Debug + Default + 'static,
    Client<T>: CompletionClient,
{
    type TranscriptionModel = TranscriptionModel<T>;

    /// Create a transcription model with the given name.
    /// Gemini-specific parameters can be set using the [GenerationConfig](crate::providers::gemini::completion::gemini_api_types::GenerationConfig) struct.
    /// [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    fn transcription_model(&self, model: &str) -> TranscriptionModel<T> {
        TranscriptionModel::new(self.clone(), model)
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + Debug + Default + WasmCompatSend + WasmCompatSync + 'static,
    Client<T>: CompletionClient,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/v1beta/models")
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;
        let response = self.http_client.send::<_, Vec<u8>>(req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::FORBIDDEN => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                // TODO: Find/write some alternative for this that uses `http::StatusCode` vs
                // reqwest::StatusCode
                //
                // response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsImageGeneration,
    AsAudioGeneration for Client<T>
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
