use serde::{Deserialize, Serialize};

use super::{
    CompletionModel,
    embedding::{EmbeddingModel, MISTRAL_EMBED},
};
use crate::client::{
    ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient, VerifyClient,
    VerifyError,
};
use crate::impl_conversion_traits;

const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: MISTRAL_API_BASE_URL,
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
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            reqwest::Client::builder().build()?
        };

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            http_client,
        })
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client {
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
    pub fn builder(api_key: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Mistral client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Mistral client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url).bearer_auth(&self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.get(url).bearer_auth(&self.api_key)
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

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let response = self.get("/models").send().await?;
        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            _ => {
                response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(AsTranscription, AsAudioGeneration, AsImageGeneration for Client);

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
