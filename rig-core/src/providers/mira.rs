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
    message::{self, AssistantContent, Message, UserContent},
    OneOrMany,
};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::string::FromUtf8Error;
use thiserror::Error;
use tracing;

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
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
struct RawMessage {
    role: String,
    content: String,
}

const MIRA_API_BASE_URL: &str = "https://api.mira.network";

impl TryFrom<RawMessage> for message::Message {
    type Error = CompletionError;

    fn try_from(raw: RawMessage) -> Result<Self, Self::Error> {
        match raw.role.as_str() {
            "user" => Ok(message::Message::User {
                content: OneOrMany::one(UserContent::Text(message::Text { text: raw.content })),
            }),
            "assistant" => Ok(message::Message::Assistant {
                content: OneOrMany::one(AssistantContent::Text(message::Text {
                    text: raw.content,
                })),
            }),
            _ => Err(CompletionError::ResponseError(format!(
                "Unsupported message role: {}",
                raw.role
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CompletionResponse {
    Structured {
        id: String,
        object: String,
        created: u64,
        model: String,
        choices: Vec<ChatChoice>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    Simple(String),
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    #[serde(deserialize_with = "deserialize_message")]
    pub message: message::Message,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
}

fn deserialize_message<'de, D>(deserializer: D) -> Result<message::Message, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = RawMessage::deserialize(deserializer)?;
    message::Message::try_from(raw).map_err(serde::de::Error::custom)
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
    pub fn new(api_key: &str) -> Result<Self, MiraError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|_| MiraError::InvalidApiKey)?,
        );
        headers.insert(
            reqwest::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static("rig-client/1.0"),
        );

        Ok(Self {
            base_url: MIRA_API_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
            headers,
        })
    }

    /// Create a new Mira client from the `MIRA_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Result<Self, MiraError> {
        let api_key = std::env::var("MIRA_API_KEY").expect("MIRA_API_KEY not set");
        Self::new(&api_key)
    }

    /// Create a new Mira client with a custom base URL and API key
    pub fn new_with_base_url(
        api_key: &str,
        base_url: impl Into<String>,
    ) -> Result<Self, MiraError> {
        let mut client = Self::new(api_key)?;
        client.base_url = base_url.into();
        Ok(client)
    }

    /// Generate a chat completion
    pub async fn generate(
        &self,
        model: &str,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, MiraError> {
        let mut messages = Vec::new();

        // Add prompt first
        let prompt_text = match &request.prompt {
            Message::User { content } => content
                .iter()
                .map(|c| match c {
                    UserContent::Text(text) => &text.text,
                    _ => "",
                })
                .collect::<Vec<_>>()
                .join(" "),
            _ => return Err(MiraError::ApiError(422)),
        };

        messages.push(serde_json::json!({
            "role": "user",
            "content": prompt_text
        }));

        // Then add chat history
        for msg in request.chat_history {
            let (role, content) = match msg {
                Message::User { content } => {
                    let text = content
                        .iter()
                        .map(|c| match c {
                            UserContent::Text(text) => &text.text,
                            _ => "",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    ("user", text)
                }
                Message::Assistant { content } => {
                    let text = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text(text) => &text.text,
                            _ => "",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    ("assistant", text)
                }
            };
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }

        let mira_request = serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": request.temperature.map(|t| t as f32),
            "max_tokens": request.max_tokens.map(|t| t as u32),
            "stream": false
        });

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.headers.clone())
            .json(&mira_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(MiraError::ApiError(status.as_u16()));
        }

        // Parse the response
        let response_text = response.text().await?;
        let parsed_response: CompletionResponse = serde_json::from_str(&response_text)?;
        Ok(parsed_response)
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let url = format!("{}/v1/models", self.base_url);
        tracing::debug!("Requesting models from: {}", url);
        tracing::debug!("Headers: {:?}", self.headers);

        let response = self
            .client
            .get(&url)
            .headers(self.headers.clone())
            .send()
            .await?;

        let status = response.status();

        if !status.is_success() {
            // Log the error text but don't store it in an unused variable
            let _error_text = response.text().await.unwrap_or_default();
            tracing::error!("Error response: {}", _error_text);
            return Err(MiraError::ApiError(status.as_u16()));
        }

        let response_text = response.text().await?;

        let models: ModelsResponse = serde_json::from_str(&response_text).map_err(|e| {
            tracing::error!("Failed to parse response: {}", e);
            MiraError::JsonError(e)
        })?;

        Ok(models.data.into_iter().map(|model| model.id).collect())
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
        if !completion_request.tools.is_empty() {
            tracing::warn!(target: "rig",
                "Tool calls are not supported by the Mira provider. {} tools will be ignored.",
                completion_request.tools.len()
            );
        }

        let mut messages = Vec::new();

        // Add preamble as user message if available
        if let Some(preamble) = &completion_request.preamble {
            messages.push(serde_json::json!({
                "role": "user",
                "content": preamble.to_string()
            }));
        }

        // Add prompt
        messages.push(match &completion_request.prompt {
            Message::User { content } => {
                let text = content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text(text) => &text.text,
                        _ => "",
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                serde_json::json!({
                    "role": "user",
                    "content": text
                })
            }
            _ => unreachable!(),
        });

        // Add chat history
        for msg in completion_request.chat_history {
            let (role, content) = match msg {
                Message::User { content } => {
                    let text = content
                        .iter()
                        .map(|c| match c {
                            UserContent::Text(text) => &text.text,
                            _ => "",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    ("user", text)
                }
                Message::Assistant { content } => {
                    let text = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text(text) => &text.text,
                            _ => "",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    ("assistant", text)
                }
            };
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }

        let mira_request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": completion_request.temperature.map(|t| t as f32).unwrap_or(0.7),
            "max_tokens": completion_request.max_tokens.map(|t| t as u32).unwrap_or(100),
            "stream": false
        });

        let response = self
            .client
            .client
            .post(format!("{}/v1/chat/completions", self.client.base_url))
            .headers(self.client.headers.clone())
            .json(&mira_request)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(CompletionError::ProviderError(format!(
                "API error: {} - {}",
                status, error_text
            )));
        }

        let response: CompletionResponse = response
            .json()
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
        let content = match &response {
            CompletionResponse::Structured { choices, .. } => {
                let choice = choices.first().ok_or_else(|| {
                    CompletionError::ResponseError("Response contained no choices".to_owned())
                })?;

                match &choice.message {
                    Message::Assistant { content } => {
                        if content.is_empty() {
                            return Err(CompletionError::ResponseError(
                                "Response contained empty content".to_owned(),
                            ));
                        }

                        // Log warning for unsupported content types
                        for c in content.iter() {
                            if !matches!(c, AssistantContent::Text(_)) {
                                tracing::warn!(target: "rig",
                                    "Unsupported content type encountered: {:?}. The Mira provider currently only supports text content", c
                                );
                            }
                        }

                        content.iter().map(|c| {
                            match c {
                                AssistantContent::Text(text) => Ok(completion::AssistantContent::text(&text.text)),
                                other => Err(CompletionError::ResponseError(
                                    format!("Unsupported content type: {:?}. The Mira provider currently only supports text content", other)
                                ))
                            }
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    Message::User { .. } => {
                        tracing::warn!(target: "rig", "Received user message in response where assistant message was expected");
                        return Err(CompletionError::ResponseError(
                            "Received user message in response where assistant message was expected".to_owned()
                        ));
                    }
                }
            }
            CompletionResponse::Simple(text) => {
                vec![completion::AssistantContent::text(text)]
            }
        };

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

impl From<Message> for serde_json::Value {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User { content } => {
                let text = content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text(text) => &text.text,
                        _ => "",
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                serde_json::json!({
                    "role": "user",
                    "content": text
                })
            }
            Message::Assistant { content } => {
                let text = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text(text) => &text.text,
                        _ => "",
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                serde_json::json!({
                    "role": "assistant",
                    "content": text
                })
            }
        }
    }
}

impl TryFrom<serde_json::Value> for Message {
    type Error = CompletionError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        let role = value["role"].as_str().ok_or_else(|| {
            CompletionError::ResponseError("Message missing role field".to_owned())
        })?;

        // Handle both string and array content formats
        let content = match value.get("content") {
            Some(content) => match content {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|c| {
                        c.get("text")
                            .and_then(|t| t.as_str())
                            .map(|text| text.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Message content must be string or array".to_owned(),
                    ))
                }
            },
            None => {
                return Err(CompletionError::ResponseError(
                    "Message missing content field".to_owned(),
                ))
            }
        };

        match role {
            "user" => Ok(Message::User {
                content: OneOrMany::one(UserContent::Text(message::Text { text: content })),
            }),
            "assistant" => Ok(Message::Assistant {
                content: OneOrMany::one(AssistantContent::Text(message::Text { text: content })),
            }),
            _ => Err(CompletionError::ResponseError(format!(
                "Unsupported message role: {}",
                role
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::UserContent;
    use serde_json::json;

    #[test]
    fn test_deserialize_message() {
        // Test string content format
        let assistant_message_json = json!({
            "role": "assistant",
            "content": "Hello there, how may I assist you today?"
        });

        let user_message_json = json!({
            "role": "user",
            "content": "What can you help me with?"
        });

        // Test array content format
        let assistant_message_array_json = json!({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello there, how may I assist you today?"
            }]
        });

        let assistant_message = Message::try_from(assistant_message_json).unwrap();
        let user_message = Message::try_from(user_message_json).unwrap();
        let assistant_message_array = Message::try_from(assistant_message_array_json).unwrap();

        // Test string content format
        match assistant_message {
            Message::Assistant { content } => {
                assert_eq!(
                    content.first(),
                    AssistantContent::Text(message::Text {
                        text: "Hello there, how may I assist you today?".to_string()
                    })
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match user_message {
            Message::User { content } => {
                assert_eq!(
                    content.first(),
                    UserContent::Text(message::Text {
                        text: "What can you help me with?".to_string()
                    })
                );
            }
            _ => panic!("Expected user message"),
        }

        // Test array content format
        match assistant_message_array {
            Message::Assistant { content } => {
                assert_eq!(
                    content.first(),
                    AssistantContent::Text(message::Text {
                        text: "Hello there, how may I assist you today?".to_string()
                    })
                );
            }
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_message_conversion() {
        // Test converting from our Message type to Mira's format and back
        let original_message = message::Message::User {
            content: OneOrMany::one(message::UserContent::text("Hello")),
        };

        // Convert to Mira format
        let mira_value: serde_json::Value = original_message.clone().try_into().unwrap();

        // Convert back to our Message type
        let converted_message: Message = mira_value.try_into().unwrap();

        // Convert back to original format
        let final_message: message::Message = converted_message.try_into().unwrap();

        assert_eq!(original_message, final_message);
    }

    #[test]
    fn test_completion_response_deserialization() {
        // Test structured response
        let structured_json = json!({
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-r1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I can help you with various tasks."
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 20
            }
        });

        // Test simple response
        let simple_json = json!("Simple response text");

        // Try both formats
        let structured: CompletionResponse = serde_json::from_value(structured_json).unwrap();
        let simple: CompletionResponse = serde_json::from_value(simple_json).unwrap();

        match structured {
            CompletionResponse::Structured {
                id,
                object,
                created,
                model,
                choices,
                usage,
            } => {
                assert_eq!(id, "resp_123");
                assert_eq!(object, "chat.completion");
                assert_eq!(created, 1234567890);
                assert_eq!(model, "deepseek-r1");
                assert!(!choices.is_empty());
                assert!(usage.is_some());

                let choice = &choices[0];
                match &choice.message {
                    Message::Assistant { content } => {
                        assert_eq!(
                            content.first(),
                            AssistantContent::Text(message::Text {
                                text: "I can help you with various tasks.".to_string()
                            })
                        );
                    }
                    _ => panic!("Expected assistant message"),
                }

                assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
                assert_eq!(choice.index, Some(0));
            }
            CompletionResponse::Simple(_) => panic!("Expected structured response"),
        }

        match simple {
            CompletionResponse::Simple(text) => {
                assert_eq!(text, "Simple response text");
            }
            CompletionResponse::Structured { .. } => panic!("Expected simple response"),
        }
    }

    #[test]
    fn test_completion_response_conversion() {
        let mira_response = CompletionResponse::Structured {
            id: "resp_123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "deepseek-r1".to_string(),
            choices: vec![ChatChoice {
                message: Message::Assistant {
                    content: OneOrMany::one(AssistantContent::Text(message::Text {
                        text: "Test response".to_string(),
                    })),
                },
                finish_reason: Some("stop".to_string()),
                index: Some(0),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                total_tokens: 20,
            }),
        };

        let completion_response: completion::CompletionResponse<CompletionResponse> =
            mira_response.try_into().unwrap();

        assert_eq!(
            completion_response.choice.first(),
            completion::AssistantContent::text("Test response")
        );
    }
}
