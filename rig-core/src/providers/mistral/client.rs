use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::{
    CompletionModel,
    embedding::{EmbeddingModel, MISTRAL_EMBED},
};
use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient, VerifyClient, VerifyError},
    http_client::HttpClientExt,
};
use crate::{http_client, impl_conversion_traits};
use std::fmt::Debug;

const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: MISTRAL_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url: MISTRAL_API_BASE_URL,
            http_client,
        }
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
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
    T: Debug,
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
    /// Create a new Mistral client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
    }

    /// Create a new Mistral client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::mistral::{ClientBuilder, self};
    ///
    /// // Initialize the Mistral client
    /// let mistral = Client::builder("your-mistral-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, reqwest::Client> {
        ClientBuilder::new(api_key)
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(http_client::Request::post(url), &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        http_client::with_bearer_auth(http_client::Request::get(url), &self.api_key)
    }

    pub(crate) async fn send<Body, R>(
        &self,
        req: http_client::Request<Body>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<R>>>
    where
        Body: Into<Bytes> + Send,
        R: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Create a new Mistral client from the `MISTRAL_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self
    where
        Self: Sized,
    {
        let api_key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
        ClientBuilder::<T>::new(&api_key).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::<T>::new(&api_key).build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type CompletionModel = CompletionModel<T>;

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

impl<T> EmbeddingsClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type EmbeddingModel = EmbeddingModel<T>;

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
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
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

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                // TODO: implement equivalent with `http` crate `StatusCode` type
                //response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(AsTranscription, AsAudioGeneration, AsImageGeneration for Client<T>);

#[derive(Clone, Debug, Deserialize, Serialize)]
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
