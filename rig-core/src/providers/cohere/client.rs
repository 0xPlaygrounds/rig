use crate::{
    Embed,
    client::{VerifyClient, VerifyError},
    embeddings::EmbeddingsBuilder,
    http_client::{self, HttpClientExt},
    wasm_compat::*,
};

use super::{CompletionModel, EmbeddingModel};
use crate::client::{CompletionClient, EmbeddingsClient, ProviderClient, impl_conversion_traits};
use bytes::Bytes;
use serde::Deserialize;

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

// ================================================================
// Main Cohere Client
// ================================================================
const COHERE_API_BASE_URL: &str = "https://api.cohere.ai";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a> ClientBuilder<'a, reqwest::Client> {
    pub fn new(api_key: &'a str) -> ClientBuilder<'a, reqwest::Client> {
        ClientBuilder {
            api_key,
            base_url: COHERE_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        ClientBuilder {
            api_key,
            base_url: COHERE_API_BASE_URL,
            http_client,
        }
    }

    pub fn with_client<U>(api_key: &str, http_client: U) -> ClientBuilder<'_, U> {
        ClientBuilder {
            api_key,
            base_url: COHERE_API_BASE_URL,
            http_client,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> ClientBuilder<'a, T> {
        self.base_url = base_url;
        self
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: HttpClientExt + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client<reqwest::Client> {
    pub fn builder(api_key: &str) -> ClientBuilder<'_, reqwest::Client> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Cohere client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        ClientBuilder::new(api_key).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) fn req(
        &self,
        method: http_client::Method,
        path: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(
            http_client::Builder::new().method(method).uri(url),
            &self.api_key,
        )
    }

    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(http_client::Method::POST, path)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(http_client::Method::GET, path)
    }

    pub fn http_client(&self) -> T {
        self.http_client.clone()
    }

    pub(crate) async fn send<U, V>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<V>>>
    where
        U: Into<Bytes> + Send,
        V: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }

    pub fn embeddings<D: Embed>(
        &self,
        model: &str,
        input_type: &str,
    ) -> EmbeddingsBuilder<EmbeddingModel<T>, D> {
        EmbeddingsBuilder::new(self.embedding_model(model, input_type))
    }

    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    pub fn embedding_model(&self, model: &str, input_type: &str) -> EmbeddingModel<T> {
        let ndims = match model {
            super::EMBED_ENGLISH_V3
            | super::EMBED_MULTILINGUAL_V3
            | super::EMBED_ENGLISH_LIGHT_V2 => 1024,
            super::EMBED_ENGLISH_LIGHT_V3 | super::EMBED_MULTILINGUAL_LIGHT_V3 => 384,
            super::EMBED_ENGLISH_V2 => 4096,
            super::EMBED_MULTILINGUAL_V2 => 768,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, input_type, ndims)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    pub fn embedding_model_with_ndims(
        &self,
        model: &str,
        input_type: &str,
        ndims: usize,
    ) -> EmbeddingModel<T> {
        EmbeddingModel::new(self.clone(), model, input_type, ndims)
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    /// Create a new Cohere client from the `COHERE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
        ClientBuilder::new_with_client(&api_key, T::default()).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::new_with_client(&api_key, T::default()).build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    type CompletionModel = CompletionModel<T>;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> EmbeddingsClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    type EmbeddingModel = EmbeddingModel<T>;

    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        self.embedding_model(model, "search_document")
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        self.embedding_model_with_ndims(model, "search_document", ndims)
    }

    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        self.embeddings(model, "search_document")
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let response = self.http_client.send::<_, Vec<u8>>(req).await?;
        let status = response.status();
        let body = http_client::text(response).await?;

        match status {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => Err(VerifyError::ProviderError(body)),
            _ => Err(VerifyError::ProviderError(body)),
        }
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);
