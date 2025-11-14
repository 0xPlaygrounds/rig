use bytes::Bytes;
use serde::Deserialize;

use crate::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient},
    completion::GetTokenUsage,
    http_client::{self, HttpClientExt},
    impl_conversion_traits,
    providers::azure_ai_foundry::{completion::CompletionModel, embedding::EmbeddingModel},
};

pub const DEFAULT_API_VERSION: &str = "2024-10-21";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    api_version: Option<&'a str>,
    azure_endpoint: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new(api_key: &'a str, endpoint: &'a str) -> Self {
        Self {
            api_key,
            api_version: None,
            azure_endpoint: endpoint,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, azure_endpoint: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            api_version: None,
            azure_endpoint,
            http_client,
        }
    }

    /// API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
    pub fn api_version(mut self, api_version: &'a str) -> Self {
        self.api_version = Some(api_version);
        self
    }

    /// Azure OpenAI endpoint URL, for example: https://{your-resource-name}.services.ai.azure.com
    /// SAFETY: Don't add a forward slash on the end of the URL
    pub fn azure_endpoint(mut self, azure_endpoint: &'a str) -> Self {
        self.azure_endpoint = azure_endpoint;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            api_version: self.api_version,
            azure_endpoint: self.azure_endpoint,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        let api_version = self.api_version.unwrap_or(DEFAULT_API_VERSION);

        Client {
            api_version: api_version.to_string(),
            azure_endpoint: self.azure_endpoint.to_string(),
            api_key: self.api_key.to_string(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    api_version: String,
    azure_endpoint: String,
    api_key: String,
    pub http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("azure_endpoint", &self.azure_endpoint)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .field("api_version", &self.api_version)
            .finish()
    }
}

impl Client<reqwest::Client> {
    /// Create a new Azure AI Foundry client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure_ai_foundry::{ClientBuilder, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::builder("your-azure-api-key", "https://{your-resource-name}.services.ai.azure.com")
    ///    .build()
    /// ```
    pub fn builder<'a>(api_key: &'a str, endpoint: &'a str) -> ClientBuilder<'a, reqwest::Client> {
        ClientBuilder::new(api_key, endpoint)
    }

    /// Creates a new Azure OpenAI client. For more control, use the `builder` method.
    pub fn new(api_key: &str, endpoint: &str) -> Self {
        Self::builder(api_key, endpoint).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub fn post(&self, url: String) -> http_client::Builder {
        http_client::Request::post(url).header("api-key", &self.api_key)
    }

    pub fn post_chat_completion(&self) -> http_client::Builder {
        let url = format!(
            "{}/models/completions?api-version={}",
            self.azure_endpoint, self.api_version
        );

        self.post(url)
    }

    pub fn post_embedding(&self) -> http_client::Builder {
        let url = format!(
            "{}/models/embeddings?api-version={}",
            self.azure_endpoint, self.api_version
        );

        self.post(url)
    }

    pub async fn send<U, R>(
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

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    /// Create a new Azure OpenAI client from the `AZURE_API_KEY` or `AZURE_TOKEN`, `AZURE_API_VERSION`, and `AZURE_ENDPOINT` environment variables.
    fn from_env() -> Self {
        let Ok(api_key) = std::env::var("AZURE_API_KEY") else {
            panic!("Neither AZURE_API_KEY nor AZURE_TOKEN is set");
        };

        let api_version = std::env::var("AZURE_API_VERSION").expect("AZURE_API_VERSION not set");
        let azure_endpoint = std::env::var("AZURE_ENDPOINT").expect("AZURE_ENDPOINT not set");

        ClientBuilder::<T>::new(&api_key, &azure_endpoint)
            .api_version(&api_version)
            .build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::ApiKeyWithVersionAndHeader(api_key, version, header) =
            input
        else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::<T>::new(&api_key, &header)
            .api_version(&version)
            .build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    type CompletionModel = super::completion::CompletionModel<T>;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let gpt4 = azure.completion_model(azure::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> EmbeddingsClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    type EmbeddingModel = super::embedding::EmbeddingModel<T>;

    /// Create an embedding model with the given name.
    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let embedding_model = azure.embedding_model(azure::TEXT_EMBEDDING_3_LARGE);
    /// ```
    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        let ndims = 0;
        EmbeddingModel::new(self.clone(), model, ndims)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::azure::{Client, self};
    ///
    /// // Initialize the Azure OpenAI client
    /// let azure = Client::new("YOUR_API_KEY", "YOUR_API_VERSION", "YOUR_ENDPOINT");
    ///
    /// let embedding_model = azure.embedding_model("model-unknown-to-rig", 3072);
    /// ```
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

impl_conversion_traits!(
    AsTranscription,
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

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage.output_tokens = usage.total_tokens - usage.input_tokens;

        Some(usage)
    }
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
