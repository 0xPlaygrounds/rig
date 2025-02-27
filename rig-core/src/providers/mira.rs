//! Mira API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use crate::{
    agent::AgentBuilder,
    completion::{self, CompletionError, CompletionRequest},
    extractor::ExtractorBuilder,
    message::{self, Message},
    OneOrMany,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::string::FromUtf8Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MiraError {
    #[error("Invalid API key")]
    InvalidApiKey,
    #[error("API error: {0}")]
    ApiError(u16),
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] FromUtf8Error),
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err,
}

#[derive(Debug, Serialize)]
pub struct AiRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

#[derive(Clone)]
/// Client for interacting with the Mira API
pub struct Client {
    base_url: String,
    client: reqwest::Client,
    headers: HeaderMap,
}

impl Client {
    /// Create a new Mira client with the given API key
    pub fn new(api_key: impl AsRef<str>) -> Result<Self, MiraError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key.as_ref()))
                .map_err(|_| MiraError::InvalidApiKey)?,
        );

        Ok(Self {
            base_url: "https://apis.mira.network".to_string(),
            client: reqwest::Client::new(),
            headers,
        })
    }

    /// Create a new Mira client with a custom base URL and API key
    pub fn new_with_base_url(
        api_key: impl AsRef<str>,
        base_url: impl Into<String>,
    ) -> Result<Self, MiraError> {
        let mut client = Self::new(api_key)?;
        client.base_url = base_url.into();
        Ok(client)
    }

    /// Generate a chat completion
    pub async fn generate(&self, request: AiRequest) -> Result<CompletionResponse, MiraError> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.headers.clone())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let response = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        let models: ModelsResponse = response.json().await?;
        Ok(models.data.into_iter().map(|model| model.id).collect())
    }

    /// Get user credits information
    pub async fn get_user_credits(&self) -> Result<Value, MiraError> {
        let response = self
            .client
            .get(format!("{}/user-credits", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }

    /// Get credits history
    pub async fn get_credits_history(&self) -> Result<Vec<Value>, MiraError> {
        let response = self
            .client
            .get(format!("{}/user-credits-history", self.base_url))
            .headers(self.headers.clone())
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(MiraError::ApiError(response.status().as_u16()));
        }

        Ok(response.json().await?)
    }

    /// Create a completion model with the given name.
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.to_owned(), model)
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

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Convert messages to Mira format
        let mut messages = Vec::new();

        // Add preamble as system message if available
        if let Some(preamble) = &completion_request.preamble {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: preamble.to_string(),
            });
        }

        // Add prompt first
        let prompt = completion_request.prompt_with_context();
        let prompt_str = match prompt {
            Message::User { content } => content
                .into_iter()
                .filter_map(|c| match c {
                    message::UserContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        };

        if !prompt_str.is_empty() {
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: prompt_str,
            });
        }

        // Add chat history
        for message in completion_request.chat_history {
            match message {
                Message::User { content } => {
                    // Convert user content to string
                    let content_str = content
                        .into_iter()
                        .filter_map(|c| match c {
                            message::UserContent::Text(text) => Some(text.text),
                            _ => None, // Skip other content types
                        })
                        .collect::<Vec<_>>()
                        .join(" ");

                    if !content_str.is_empty() {
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: content_str,
                        });
                    }
                }
                Message::Assistant { content } => {
                    // Convert assistant content to string
                    let content_str = content
                        .into_iter()
                        .filter_map(|c| match c {
                            message::AssistantContent::Text(text) => Some(text.text),
                            _ => None, // Skip tool calls
                        })
                        .collect::<Vec<_>>()
                        .join(" ");

                    if !content_str.is_empty() {
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: content_str,
                        });
                    }
                }
            }
        }

        let request = AiRequest {
            model: self.model.clone(),
            messages,
            temperature: Some(completion_request.temperature.unwrap_or(0.7) as f32),
            max_tokens: None,
            stream: None,
        };

        let response = self
            .client
            .generate(request)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        response.try_into()
    }
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = vec![completion::AssistantContent::text(&choice.message.content)];

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Text, UserContent};

    #[tokio::test]
    async fn test_generate() {
        let client =
            Client::new("mira-api-key").unwrap();

        let request = AiRequest {
            model: "deepseek-r1".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello, What can you do?".to_string(),
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: None,
        };

        let _response = client.generate(request).await.unwrap();
    }

    #[tokio::test]
    async fn test_completion_model() {
        let client =
            Client::new("mira-api-key").unwrap();
        let model = client.completion_model("deepseek-r1");

        let request = CompletionRequest {
            prompt: Message::User {
                content: OneOrMany::one(UserContent::Text(Text {
                    text: "Hello, what can you do?".to_string(),
                })),
            },
            temperature: Some(0.7),
            preamble: None,
            chat_history: Vec::new(),
            additional_params: None,
            documents: Vec::new(),
            tools: Vec::new(),
            max_tokens: None,
        };

        let _response = completion::CompletionModel::completion(&model, request)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_list_models() {
        let client = Client::new("mira-api-key").unwrap();
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[tokio::test]
    async fn test_get_user_credits() {
        let client =
            Client::new("mira-api-key").unwrap();
        let _credits = client.get_user_credits().await.unwrap();
    }

    #[tokio::test]
    async fn test_get_credits_history() {
        let client =
            Client::new("mira-api-key").unwrap();
        let _history = client.get_credits_history().await.unwrap();
    }
}
