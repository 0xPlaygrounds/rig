use super::{M2_BERT_80M_8K_RETRIEVAL, completion::CompletionModel, embedding::EmbeddingModel};
use crate::{
    client::{EmbeddingsClient, ProviderClient, VerifyClient, VerifyError, impl_conversion_traits},
    http_client::{self, HttpClientExt},
};
use bytes::Bytes;
use rig::client::CompletionClient;

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

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
            base_url: TOGETHER_AI_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client: self.http_client,
        }
    }
}
#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    default_headers: reqwest::header::HeaderMap,
    api_key: String,
    http_client: T,
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
    T: Default,
{
    /// Create a new Together AI client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::together_ai::{ClientBuilder, self};
    ///
    /// // Initialize the Together AI client
    /// let together_ai = Client::builder("your-together-ai-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, T> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Together AI client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
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
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

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
    pub(crate) fn reqwest_post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        tracing::debug!("POST {}", url);

        self.http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client<reqwest::Client> {
    /// Create a new Together AI client from the `TOGETHER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY not set");
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
    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client<reqwest::Client> {
    type EmbeddingModel = EmbeddingModel<reqwest::Client>;

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
    fn embedding_model(&self, model: &str) -> EmbeddingModel<reqwest::Client> {
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
    fn embedding_model_with_ndims(
        &self,
        model: &str,
        ndims: usize,
    ) -> EmbeddingModel<reqwest::Client> {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl VerifyClient for Client<reqwest::Client> {
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
