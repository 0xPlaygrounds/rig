//! OpenAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::openai;
//!
//! let client = openai::Client::new("YOUR_API_KEY");
//!
//! let gpt4o = client.completion_model(openai::GPT_4O);
//! ```
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    extractor::ExtractorBuilder,
    json_utils,
    model::ModelBuilder,
    rag::RagAgentBuilder,
    vector_store::{NoIndex, VectorStoreIndex},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, OPENAI_API_BASE_URL)
    }

    /// Create a new OpenAI client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("OpenAI reqwest client should build"),
        }
    }

    /// Create a new OpenAI client from the `OPENAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        Self::new(&api_key)
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create an embedding model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_3_LARGE);
    /// ```
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model)
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let embeddings = openai.embeddings(openai::TEXT_EMBEDDING_3_LARGE)
    ///     .simple_document("doc0", "Hello, world!")
    ///     .simple_document("doc1", "Goodbye, world!")
    ///     .build()
    ///     .await
    ///     .expect("Failed to embed documents");
    /// ```
    pub fn embeddings(&self, model: &str) -> embeddings::EmbeddingsBuilder<EmbeddingModel> {
        embeddings::EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.completion_model(openai::GPT_4);
    /// ```
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create a model builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let completion_model = openai.model(openai::GPT_4)
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    pub fn model(&self, model: &str) -> ModelBuilder<CompletionModel> {
        ModelBuilder::new(self.completion_model(model))
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let agent = openai.agent(openai::GPT_4)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn rag_agent<C: VectorStoreIndex, T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn tool_rag_agent<T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, NoIndex, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn context_rag_agent<C: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, NoIndex> {
        RagAgentBuilder::new(self.completion_model(model))
    }
}

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

// ================================================================
// OpenAI Embedding API
// ================================================================
pub const OPENAI_EMBEDDINGS_ENDPOINT: &str = "/v1/embeddings";

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
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

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let response = self
            .client
            .post(OPENAI_EMBEDDINGS_ENDPOINT)
            .json(&json!({
                "model": self.model,
                "input": documents,
            }))
            .send()
            .await?
            .json::<ApiResponse<EmbeddingResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => {
                if response.data.len() != documents.len() {
                    return Err(EmbeddingError::ResponseError(
                        "Response data length does not match input length".into(),
                    ));
                }

                Ok(response
                    .data
                    .into_iter()
                    .zip(documents.into_iter())
                    .map(|(embedding, document)| embeddings::Embedding {
                        document,
                        vec: embedding.embedding,
                    })
                    .collect())
            }
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

// ================================================================
// OpenAI Completion API
// ================================================================
pub const OPENAI_COMPLETION_ENDPOINT: &str = "/v1/chat/completions";

/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

#[derive(Debug, Deserialize, PartialEq)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(value: CompletionResponse) -> std::prelude::v1::Result<Self, Self::Error> {
        match value.choices.as_slice() {
            [Choice {
                message:
                    Message {
                        content: Some(content),
                        ..
                    },
                ..
            }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(content.to_string()),
                raw_response: value,
            }),
            [Choice {
                message:
                    Message {
                        tool_calls: Some(calls),
                        ..
                    },
                ..
            }, ..] => {
                let call = calls.first().ok_or(CompletionError::ResponseError(
                    "Tool selection is empty".into(),
                ))?;

                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::ToolCall(
                        call.function.name.clone(),
                        serde_json::from_str(&call.function.arguments)?,
                    ),
                    raw_response: value,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a message or tool call".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Add preamble to chat history (if available)
        let mut full_history = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        // Add context documents to chat history
        full_history.append(
            completion_request
                .documents
                .into_iter()
                .map(|doc| completion::Message {
                    role: "system".into(),
                    content: serde_json::to_string(&doc).expect("Document should serialize"),
                })
                .collect::<Vec<_>>()
                .as_mut(),
        );

        // Add context documents to chat history
        full_history.append(&mut completion_request.chat_history);

        // Add context documents to chat history
        full_history.push(completion::Message {
            role: "user".into(),
            content: completion_request.prompt,
        });

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let response = self
            .client
            .post(OPENAI_COMPLETION_ENDPOINT)
            .json(
                &if let Some(params) = completion_request.additional_params {
                    json_utils::merge(request, params)
                } else {
                    request
                },
            )
            .send()
            .await?
            .json::<ApiResponse<CompletionResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{completion::{CompletionModel, CompletionRequestBuilder}, embeddings::EmbeddingModel};
    use httpmock::{Method::POST, MockServer};
    
    #[tokio::test]
    async fn test_embedding() {
        use super::*;

        let server = MockServer::start();

        let mock_embedding_endpoint = server.mock(|when, then| {
            when.method(POST)
                .path(OPENAI_EMBEDDINGS_ENDPOINT)
                .json_body(json!({
                    "model": TEXT_EMBEDDING_3_LARGE,
                    "input": ["Hello, world!"]
                }));

            then.status(200)
                .json_body(json!({
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": vec![0.0, 1.0, 2.0],
                            "index": 0
                        }
                    ],
                    "model": TEXT_EMBEDDING_3_LARGE,
                    "usage": {
                        "prompt_tokens": 1,
                        "total_tokens": 3
                    }
                }));
        });

        let client = Client::from_url("", &server.base_url());
        let model = client.embedding_model(TEXT_EMBEDDING_3_LARGE);

        let embeddings = model
            .embed_documents(vec!["Hello, world!".to_string()])
            .await
            .expect("Failed to embed documents");

        mock_embedding_endpoint.assert();

        assert_eq!(
            embeddings,
            serde_json::from_value::<Vec<_>>(json!([
                {
                    "document": "Hello, world!",
                    "vec": vec![0.0, 1.0, 2.0]
                }
            ])).unwrap()
        )
    }

    #[tokio::test]
    async fn test_completion() {
        use super::*;

        let server = MockServer::start();

        let mock_completion_endpoint = server.mock(|when, then| {
            when.method(POST)
                .path(OPENAI_COMPLETION_ENDPOINT)
                .json_body(json!({
                    "model": GPT_4O,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, world!"
                        }
                    ],
                    "temperature": Option::<f64>::None
                }));

            then.status(200)
                .json_body(json!({
                    "id": "chatcmpl-9m2pR3BqoB0n4FtHjDSFRl4oOZB01",
                    "object": "chat.completion",
                    "created": 1721237645,
                    "model": "gpt-4o-2024-05-13",
                    "choices": [
                      {
                        "index": 0,
                        "message": {
                          "role": "assistant",
                          "content": "Hi there! How can I assist you today?"
                        },
                        "logprobs": null,
                        "finish_reason": "stop"
                      }
                    ],
                    "usage": {
                      "prompt_tokens": 19,
                      "completion_tokens": 10,
                      "total_tokens": 29
                    },
                    "system_fingerprint": "fp_c4e5b6fa31"
                }));
        });

        let client = Client::from_url("", &server.base_url());
        let model = client.completion_model(GPT_4O);

        let completion = model
            .completion(CompletionRequestBuilder::new(
                model.clone(),
                "Hello, world!".to_string()).build())
            .await
            .expect("Failed to complete prompt");

        mock_completion_endpoint.assert();

        assert_eq!(
            completion,
            completion::CompletionResponse {
                choice: completion::ModelChoice::Message("Hi there! How can I assist you today?".to_string()),
                raw_response: CompletionResponse {
                    id: "chatcmpl-9m2pR3BqoB0n4FtHjDSFRl4oOZB01".to_string(),
                    object: "chat.completion".to_string(),
                    created: 1721237645,
                    model: "gpt-4o-2024-05-13".to_string(),
                    system_fingerprint: Some("fp_c4e5b6fa31".to_string()),
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: "assistant".to_string(),
                            content: Some("Hi there! How can I assist you today?".to_string()),
                            tool_calls: None
                        },
                        logprobs: None,
                        finish_reason: "stop".to_string()
                    }],
                    usage: Some(Usage {
                        prompt_tokens: 19,
                        completion_tokens: 10,
                        total_tokens: 29,
                    })
                }
            }
        )
    }
}