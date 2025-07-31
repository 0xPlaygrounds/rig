use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Embed, OneOrMany,
    client::{ClientBuilderError, CompletionClient, EmbeddingsClient, ProviderClient},
    completion::{self, CompletionError, CompletionRequest, Usage},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    impl_conversion_traits,
};

const FOUNDRY_API_BASE_URL: &str = "http://localhost:8080";

pub struct ClientBuilder<'a> {
    base_url: &'a str,
    http_client: Option<reqwest::Client>,
}

impl<'a> ClientBuilder<'a> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            base_url: FOUNDRY_API_BASE_URL,
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
            http_client,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a new Ollama client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::ollama::{ClientBuilder, self};
    ///
    /// // Initialize the Ollama client
    /// let client = Client::builder()
    ///    .build()
    /// ```
    pub fn builder() -> ClientBuilder<'static> {
        ClientBuilder::new()
    }

    /// Create a new Ollama client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new() -> Self {
        Self::builder().build().expect("Ollama client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path);
        self.http_client.post(url)
    }
}

impl ProviderClient for Client {
    fn from_env() -> Self
    where
        Self: Sized,
    {
        let api_base = std::env::var("OLLAMA_API_BASE_URL").expect("OLLAMA_API_BASE_URL not set");
        Self::builder().base_url(&api_base).build().unwrap()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(_) = input else {
            panic!("Incorrect provider value type")
        };

        Self::new()
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;
    fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }
    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client
);

// ---------- API Error and Response Structures ----------

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ---------- Embedding API ----------

/// TODO: mention the commpletion models here

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingData {
    object: String,
    embedding: Vec<f64>,
    index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    total_tokens: u64,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

// ----------- Embedding Model --------------

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_owned(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;
    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();
        let payload = json!({
            "model": self.model,
            "input":docs,
        });
        let response = self
            .client
            .post("v1/embeddings")
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::ResponseError(e.to_string()))?;
        if response.status().is_success() {
            let api_resp: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

            if api_resp.data.len() != docs.len() {
                return Err(EmbeddingError::ResponseError(
                    "Number of returned embeddings does not match input".into(),
                ));
            }
            Ok(api_resp
                .data
                .into_iter()
                .zip(docs.into_iter())
                .map(|(vec, document)| embeddings::Embedding { document, vec })
                .collect())
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

// ----------- Completions API -------------

// TODO: add models here

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Choice {
    pub index: u64,
    pub message: CompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompletionMessage {
    pub role: String,
    pub content: String,
}

pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: CompletionsUsage,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let mut assitant_contents = Vec::new();

        // foundry only responds with an array of choices which have
        // role and content (role is always "assistant" for responses)
        for choice in resp.choices.clone() {
            assitant_contents.push(completion::AssistantContent::text(&choice.message.content));
        }

        let choice = OneOrMany::many(assitant_contents)
            .map_err(|_| CompletionError::ResponseError("No content provided".to_owned()))?;

        Ok(completion::CompletionResponse {
            choice,
            usage: rig::completion::request::Usage {
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                total_tokens: resp.usage.total_tokens,
            },
            raw_response: resp,
        })
    }
}

// ----------- Completion Model ----------

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
    }
}
