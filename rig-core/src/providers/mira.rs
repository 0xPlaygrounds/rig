//! Mira API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge;
use crate::message::{Document, DocumentSourceKind};
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    message::{self, AssistantContent, Message, UserContent},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::string::FromUtf8Error;
use thiserror::Error;
use tracing::{self, Instrument, info_span};

#[derive(Debug, Default, Clone, Copy)]
pub struct MiraExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MiraBuilder;

type MiraApiKey = BearerAuth;

impl Provider for MiraExt {
    type Builder = MiraBuilder;

    const VERIFY_PATH: &'static str = "/user-credits";

    fn build<H>(
        _: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as crate::client::ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for MiraExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;

    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for MiraExt {}

impl ProviderBuilder for MiraBuilder {
    type Output = MiraExt;
    type ApiKey = MiraApiKey;

    const BASE_URL: &'static str = MIRA_API_BASE_URL;
}

pub type Client<H = reqwest::Client> = client::Client<MiraExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<MiraBuilder, MiraApiKey, H>;

#[derive(Debug, Error)]
pub enum MiraError {
    #[error("Invalid API key")]
    InvalidApiKey,
    #[error("API error: {0}")]
    ApiError(u16),
    #[error("Request error: {0}")]
    RequestError(#[from] http_client::Error),
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] FromUtf8Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct RawMessage {
    pub role: String,
    pub content: String,
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
                id: None,
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

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatChoice {
    pub message: RawMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    id: String,
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let req = self.get("/v1/models").and_then(|req| {
            req.body(http_client::NoBody)
                .map_err(http_client::Error::Protocol)
        })?;

        let response = self.http_client().send(req).await?;

        let status = response.status();

        if !status.is_success() {
            // Log the error text but don't store it in an unused variable
            let error_text = http_client::text(response).await.unwrap_or_default();
            tracing::error!("Error response: {}", error_text);
            return Err(MiraError::ApiError(status.as_u16()));
        }

        let response_text = http_client::text(response).await?;

        let models: ModelsResponse = serde_json::from_str(&response_text).map_err(|e| {
            tracing::error!("Failed to parse response: {}", e);
            MiraError::JsonError(e)
        })?;

        Ok(models.data.into_iter().map(|model| model.id).collect())
    }
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Mira client from the `MIRA_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("MIRA_API_KEY").expect("MIRA_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        if completion_request.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported on Mira AI");
        }

        let mut messages = Vec::new();

        // Add preamble as user message if available
        if let Some(preamble) = &completion_request.preamble {
            messages.push(serde_json::json!({
                "role": "user",
                "content": preamble.to_string()
            }));
        }

        // Add docs
        if let Some(Message::User { content }) = completion_request.normalized_documents() {
            let text = content
                .into_iter()
                .filter_map(|doc| match doc {
                    UserContent::Document(Document {
                        data: DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data),
                        ..
                    }) => Some(data),
                    UserContent::Text(text) => Some(text.text),

                    // This should always be `Document`
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            messages.push(serde_json::json!({
                "role": "user",
                "content": text
            }));
        }

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
                        .join("\n");
                    ("user", text)
                }
                Message::Assistant { content, .. } => {
                    let text = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text(text) => &text.text,
                            _ => "",
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    ("assistant", text)
                }
            };
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }

        let request = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": completion_request.temperature.map(|t| t as f32).unwrap_or(0.7),
            "max_tokens": completion_request.max_tokens.map(|t| t as u32).unwrap_or(100),
            "stream": false
        });

        Ok(request)
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        if !completion_request.tools.is_empty() {
            tracing::warn!(target: "rig::completions",
                "Tool calls are not supported by the Mira provider. {len} tools will be ignored.",
                len = completion_request.tools.len()
            );
        }

        let preamble = completion_request.preamble.clone();

        let request = self.create_completion_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "mira",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self
                .client
                .http_client()
                .send::<_, bytes::Bytes>(req)
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                let status = status.as_u16();
                let error_text = String::from_utf8_lossy(&response_body).to_string();
                return Err(CompletionError::ProviderError(format!(
                    "API error: {status} - {error_text}"
                )));
            }

            let response: CompletionResponse = serde_json::from_slice(&response_body)?;

            if let CompletionResponse::Structured {
                id,
                model,
                choices,
                usage,
                ..
            } = &response
            {
                let span = tracing::Span::current();
                span.record("gen_ai.response.model_name", model);
                span.record("gen_ai.response.id", id);
                span.record(
                    "gen_ai.output.messages",
                    serde_json::to_string(choices).unwrap(),
                );
                if let Some(usage) = usage {
                    span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                    span.record(
                        "gen_ai.usage.output_tokens",
                        usage.total_tokens - usage.prompt_tokens,
                    );
                }
            }

            response.try_into()
        };

        async_block.instrument(span).await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let mut request = self.create_completion_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "mira",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        request = merge(request, json!({"stream": true}));
        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        send_compatible_streaming_request(self.client.http_client().clone(), req)
            .instrument(span)
            .await
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
        let (content, usage) = match &response {
            CompletionResponse::Structured { choices, usage, .. } => {
                let choice = choices.first().ok_or_else(|| {
                    CompletionError::ResponseError("Response contained no choices".to_owned())
                })?;

                let usage = usage
                    .as_ref()
                    .map(|usage| completion::Usage {
                        input_tokens: usage.prompt_tokens as u64,
                        output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                        total_tokens: usage.total_tokens as u64,
                    })
                    .unwrap_or_default();

                // Convert RawMessage to message::Message
                let message = message::Message::try_from(choice.message.clone())?;

                let content = match message {
                    Message::Assistant { content, .. } => {
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
                                    format!("Unsupported content type: {other:?}. The Mira provider currently only supports text content")
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
                };

                (content, usage)
            }
            CompletionResponse::Simple(text) => (
                vec![completion::AssistantContent::text(text)],
                completion::Usage::new(),
            ),
        };

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
                    .join("\n");
                serde_json::json!({
                    "role": "user",
                    "content": text
                })
            }
            Message::Assistant { content, .. } => {
                let text = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text(text) => &text.text,
                        _ => "",
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
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
                    .join("\n"),
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Message content must be string or array".to_owned(),
                    ));
                }
            },
            None => {
                return Err(CompletionError::ResponseError(
                    "Message missing content field".to_owned(),
                ));
            }
        };

        match role {
            "user" => Ok(Message::User {
                content: OneOrMany::one(UserContent::Text(message::Text { text: content })),
            }),
            "assistant" => Ok(Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::Text(message::Text { text: content })),
            }),
            _ => Err(CompletionError::ResponseError(format!(
                "Unsupported message role: {role}"
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
            Message::Assistant { content, .. } => {
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
            Message::Assistant { content, .. } => {
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
        let mira_value: serde_json::Value = original_message.clone().into();

        // Convert back to our Message type
        let converted_message: Message = mira_value.try_into().unwrap();

        assert_eq!(original_message, converted_message);
    }

    #[test]
    fn test_completion_response_conversion() {
        let mira_response = CompletionResponse::Structured {
            id: "resp_123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "deepseek-r1".to_string(),
            choices: vec![ChatChoice {
                message: RawMessage {
                    role: "assistant".to_string(),
                    content: "Test response".to_string(),
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
