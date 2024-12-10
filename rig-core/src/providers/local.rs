//! Local API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::local;
//!
//! let client = local::Client::new();
//!
//! let model = client.completion_model("llama3.1:8b-instruct-q8_0");
//! ```

use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    extractor::ExtractorBuilder,
    json_utils, Embed,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// Main Local Client
// ================================================================
const DEFAULT_API_BASE_URL: &str = "http://localhost:11434"; // Ollama and LM Studio endpoint

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new Local client with an optional API key, using the default endpoint (ollama).
    pub fn new() -> Self {
        Self::from_url("", DEFAULT_API_BASE_URL)
    }

    /// Create a new Local client with the given API key and base API URL.
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
                .expect("Local reqwest client should build"),
        }
    }

    /// Create a new Local client from the `LOCAL_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        Self::new()
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create an embedding model with the given name.
    pub fn embedding_model(&self, model: &str) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 1536) // Default to 1536 dimensions
    }

    /// Create an embedding builder with the given embedding model.
    pub fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
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
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
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
// Local Embedding API
// ================================================================

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

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
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

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let response = self
            .client
            .post("/v1/embeddings")
            .json(&json!({
                "model": self.model,
                "input": documents,
            }))
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<EmbeddingResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Local embedding token usage: {}",
                        response.usage
                    );

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
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

// ================================================================
// Local Completion API
// ================================================================

#[derive(Debug, Deserialize)]
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
        tracing::debug!(target: "rig", ?value, "Processing completion response");
        match value.choices.as_slice() {
            // First check for tool calls
            [Choice {
                message:
                    Message {
                        tool_calls: Some(calls),
                        ..
                    },
                ..
            }, ..] => {
                tracing::debug!(target: "rig", ?calls, "Received tool calls");
                let call = calls.first().ok_or_else(|| {
                    CompletionError::ResponseError("Tool selection is empty".into())
                })?;

                tracing::info!(target: "rig",
                    tool_name = ?call.function.name,
                    args = ?call.function.arguments,
                    "Processing tool call"
                );

                // Validate the arguments are valid JSON
                let parsed_args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .map_err(|e| {
                        CompletionError::ResponseError(format!(
                            "Invalid tool arguments JSON: {}",
                            e
                        ))
                    })?;

                // Validate it's a JSON object
                if !parsed_args.is_object() {
                    return Err(CompletionError::ResponseError(
                        "Tool arguments must be a JSON object".into(),
                    ));
                }

                tracing::debug!(target: "rig", ?parsed_args, "Parsed and validated tool arguments");

                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::ToolCall(
                        call.function.name.clone(),
                        parsed_args,
                    ),
                    raw_response: value,
                })
            }
            // Then check for content
            [Choice {
                message:
                    Message {
                        content: Some(content),
                        tool_calls: None,
                        ..
                    },
                ..
            }, ..] => {
                tracing::debug!(target: "rig", content_length = content.len(), "Received text response");
                Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::Message(content.to_string()),
                    raw_response: value,
                })
            }
            _ => {
                tracing::error!(
                    target: "rig",
                    choices = ?value.choices,
                    "Response contained neither valid message nor tool calls"
                );
                Err(CompletionError::ResponseError(
                    "Response did not contain a valid message or tool call".into(),
                ))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
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
            tracing::debug!(target: "rig", system_prompt = ?preamble, "System prompt");
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };

        // Extend existing chat history
        full_history.append(&mut completion_request.chat_history);

        // Add context documents to chat history
        let prompt_with_context = completion_request.prompt_with_context();
        tracing::debug!(target: "rig", prompt = ?prompt_with_context, "User prompt with context");

        // Add context documents to chat history
        full_history.push(completion::Message {
            role: "user".into(),
            content: prompt_with_context,
        });

        let request = if completion_request.tools.is_empty() {
            tracing::debug!(target: "rig", "No tools provided in request");
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            let tools: Vec<ToolDefinition> = completion_request
                .tools
                .into_iter()
                .map(ToolDefinition::from)
                .collect();
            tracing::info!(target: "rig", tool_count = tools.len(), ?tools, "Sending tools to model");
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": tools,
                "tool_choice": "auto",
            })
        };

        tracing::debug!(target: "rig", ?request, "Request to local model");

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(
                &if let Some(params) = completion_request.additional_params {
                    json_utils::merge(request, params)
                } else {
                    request
                },
            )
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Local completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    tracing::debug!(target: "rig", ?response, "Raw response from local model");
                    response.try_into()
                }
                ApiResponse::Err(err) => {
                    tracing::error!(target: "rig", error = ?err.message, "Local model error");
                    Err(CompletionError::ProviderError(err.message))
                }
            }
        } else {
            let error_text = response.text().await?;
            tracing::error!(target: "rig", error = ?error_text, "Local model HTTP error");
            Err(CompletionError::ProviderError(error_text))
        }
    }
}
