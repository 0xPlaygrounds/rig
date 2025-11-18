use super::{M2_BERT_80M_8K_RETRIEVAL, completion::CompletionModel, embedding::EmbeddingModel};
use crate::{
    client::{
        EmbeddingsClient, ProviderClient, StandardClientBuilder, VerifyClient, VerifyError,
        impl_conversion_traits,
    },
    http_client::{self, HttpClientExt},
};
use bytes::Bytes;
use rig::client::CompletionClient;

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    default_headers: reqwest::header::HeaderMap,
    api_key: String,
    pub http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
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
    T: HttpClientExt,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        tracing::debug!("POST {}", url);

        let mut req = http_client::Request::post(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        tracing::debug!("GET {}", url);

        let mut req = http_client::Request::get(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) async fn send<U, R>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http::Response<http_client::LazyBody<R>>>
    where
        U: Into<Bytes> + Send,
        R: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }
}

impl Client<reqwest::Client> {
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Together AI client should build")
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }

    /// Create a new Together AI client builder
    pub fn builder(api_key: &str) -> crate::client::Builder<'_, Self, reqwest::Client> {
        <Self as StandardClientBuilder<reqwest::Client>>::builder(api_key)
    }
}

impl<T> StandardClientBuilder<T> for Client<T>
where
    T: HttpClientExt,
{
    fn build_from_builder<Ext>(
        builder: crate::client::Builder<'_, Self, T, Ext>,
    ) -> Result<Self, crate::client::ClientBuilderError>
    where
        Ext: Default,
        T: Default + Clone,
    {
        let api_key = builder.get_api_key();
        let base_url = builder.get_base_url(TOGETHER_AI_BASE_URL);
        let http_client = builder.get_http_client();
        let default_headers = builder.get_headers(true);

        Ok(Client {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            default_headers,
            http_client,
        })
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Create a new Together AI client from the `TOGETHER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY not set");
        Self::builder(&api_key)
            .build()
            .expect("Together AI client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::builder(&api_key)
            .build()
            .expect("Together AI client should build")
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type CompletionModel = CompletionModel<T>;

    /// Create a completion model with the given name.
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
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig::providers::together_ai::{Client, self};
    ///
    /// // Initialize the Together AI client
    /// let together_ai = Client::new("your-together-ai-api-key");
    ///
    /// let embedding_model = together_ai.embedding_model(together_ai::embedding::EMBEDDING_V1);
    /// ```
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        let ndims = match model {
            M2_BERT_80M_8K_RETRIEVAL => 8192,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding
    /// generated by the model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::together_ai::{Client, self};
    ///
    /// // Initialize the Together AI client
    /// let together_ai = Client::new("your-together-ai-api-key");
    ///
    /// let embedding_model = together_ai.embedding_model_with_ndims("model-unknown-to-rig", 1024);
    /// ```
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
            reqwest::StatusCode::INTERNAL_SERVER_ERROR | reqwest::StatusCode::GATEWAY_TIMEOUT => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                //response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(AsTranscription, AsImageGeneration, AsAudioGeneration for Client<T>);

pub mod together_ai_api_types {
    use serde::Deserialize;

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            format!("Code `{}`: {}", self.code, self.error)
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: String,
        pub code: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse),
    }
}
