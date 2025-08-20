use super::{M2_BERT_80M_8K_RETRIEVAL, completion::CompletionModel, embedding::EmbeddingModel};
use crate::client::{
    ClientBuilderError, EmbeddingsClient, ProviderClient, VerifyClient, VerifyError,
    impl_conversion_traits,
};
use rig::client::CompletionClient;

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: TOGETHER_AI_BASE_URL,
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
    default_headers: reqwest::header::HeaderMap,
    api_key: String,
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
    pub fn builder(api_key: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Together AI client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Together AI client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        tracing::debug!("POST {}", url);
        self.http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .headers(self.default_headers.clone())
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        tracing::debug!("GET {}", url);
        self.http_client
            .get(url)
            .bearer_auth(&self.api_key)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client {
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

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
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
    /// use rig::providers::together_ai::{Client, self};
    ///
    /// // Initialize the Together AI client
    /// let together_ai = Client::new("your-together-ai-api-key");
    ///
    /// let embedding_model = together_ai.embedding_model(together_ai::embedding::EMBEDDING_V1);
    /// ```
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
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
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let response = self.get("/models").send().await?;
        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR | reqwest::StatusCode::GATEWAY_TIMEOUT => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            _ => {
                response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(AsTranscription, AsImageGeneration, AsAudioGeneration for Client);

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
