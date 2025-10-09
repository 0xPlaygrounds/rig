//! Mira API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use crate::client::{CompletionClient, ProviderClient, VerifyClient, VerifyError};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge;
use crate::message::{Document, DocumentSourceKind};
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    impl_conversion_traits,
    message::{self, AssistantContent, Message, UserContent},
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::string::FromUtf8Error;
use thiserror::Error;
use tracing::{self, Instrument, info_span};

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
            base_url: MIRA_API_BASE_URL,
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
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            reqwest::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static("rig-client/1.0"),
        );

        Client {
            base_url: self.base_url.to_string(),
            http_client: self.http_client,
            api_key: self.api_key.to_string(),
            headers,
        }
    }
}

#[derive(Clone)]
/// Client for interacting with the Mira API
pub struct Client<T = reqwest::Client> {
    base_url: String,
    http_client: T,
    api_key: String,
    headers: HeaderMap,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .field("headers", &self.headers)
            .finish()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    /// Create a new Mira client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::mira::{ClientBuilder, self};
    ///
    /// // Initialize the Mira client
    /// let mira = Client::builder("your-mira-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, T> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Mira client. For more control, use the `builder` method.
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
    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let req = self.get("/v1/models").and_then(|req| {
            req.body(http_client::NoBody)
                .map_err(http_client::Error::Protocol)
        })?;

        let response = self.http_client.send(req).await?;

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

    fn req(
        &self,
        method: http_client::Method,
        path: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        let mut req = http_client::Builder::new().method(method).uri(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.headers.clone();
        }

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(http_client::Method::POST, path)
    }
}

impl Client<reqwest::Client> {
    fn reqwest_post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        self.http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .headers(self.headers.clone())
    }
}

impl ProviderClient for Client<reqwest::Client> {
    /// Create a new Mira client from the `MIRA_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("MIRA_API_KEY").expect("MIRA_API_KEY not set");
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
        CompletionModel::new(self.to_owned(), model)
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/user-credits")?
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE
            | reqwest::StatusCode::BAD_GATEWAY => {
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

impl_conversion_traits!(
    AsEmbeddings,
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

#[derive(Clone)]
pub struct CompletionModel<T> {
    client: Client<T>,
    /// Name of the model
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
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

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

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

        let async_block = async move {
            let response = self
                .client
                .reqwest_post("/v1/chat/completions")
                .json(&request)
                .send()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_default();
                return Err(CompletionError::ProviderError(format!(
                    "API error: {status} - {error_text}"
                )));
            }

            let response: CompletionResponse = response
                .json()
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

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

        let builder = self
            .client
            .reqwest_post("/v1/chat/completions")
            .json(&request);

        send_compatible_streaming_request(builder)
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
