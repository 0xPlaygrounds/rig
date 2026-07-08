//! Mira API client and Rig integration
//!
//! # Example
//! ```
//! use rig_core::providers::mira;
//!
//! let client = mira::Client::new("YOUR_API_KEY");
//!
//! ```
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    message::{self, AssistantContent, Message, UserContent},
};
use serde::{Deserialize, Serialize};
use std::string::FromUtf8Error;
use thiserror::Error;
use tracing::{self};

#[derive(Debug, Default, Clone, Copy)]
pub struct MiraExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MiraBuilder;

type MiraApiKey = BearerAuth;

impl Provider for MiraExt {
    type Builder = MiraBuilder;

    const VERIFY_PATH: &'static str = "/user-credits";
}

impl<H> Capabilities<H> for MiraExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;

    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for MiraExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for MiraExt {
    const PROVIDER_NAME: &'static str = "mira";

    // Mira's gateway does not accept OpenAI structured-output parameters.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = CompletionResponse;

    // The client base URL is the bare host; `list_models` builds its own v1 path.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }

    fn prepare_request(
        &self,
        request: &mut crate::providers::openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Mira's gateway rejects tool and pass-through parameters.
        if !request.tools.is_empty() {
            tracing::warn!("Tool use is not supported by Mira; tools will be ignored");
            request.tools.clear();
        }
        if request.tool_choice.take().is_some() {
            tracing::warn!("Tool choice is not supported by Mira and will be ignored");
        }
        if request.additional_params.take().is_some() {
            tracing::warn!("Additional parameters are not supported by Mira and will be ignored");
        }

        Ok(())
    }

    fn finalize_request_body(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };

        // Mira only understands plain `{role, content}` string messages.
        if let Some(messages) = map
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            messages.retain(|message| {
                message.get("role").and_then(serde_json::Value::as_str) != Some("tool")
            });
            for message in messages {
                let Some(message) = message.as_object_mut() else {
                    continue;
                };
                if let Some(content) = message.get_mut("content")
                    && let Some(parts) = content.as_array()
                {
                    let flattened = parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n");
                    *content = serde_json::Value::String(flattened);
                }
                message.remove("tool_calls");
                message.remove("reasoning_content");
                message.remove("name");
                if !message.contains_key("content") {
                    message.insert(
                        "content".to_string(),
                        serde_json::Value::String(String::new()),
                    );
                }
            }
        }

        Ok(())
    }
}

impl ProviderBuilder for MiraBuilder {
    type Extension<H>
        = MiraExt
    where
        H: HttpClientExt;
    type ApiKey = MiraApiKey;

    const BASE_URL: &'static str = MIRA_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MiraExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<MiraExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MiraBuilder, MiraApiKey, H>;

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
            "system" => Ok(message::Message::System {
                content: raw.content,
            }),
            "user" => Ok(message::Message::User {
                content: OneOrMany::one(UserContent::Text(message::Text::new(raw.content))),
            }),
            "assistant" => Ok(message::Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::Text(message::Text::new(raw.content))),
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
    T: HttpClientExt + 'static,
{
    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, MiraError> {
        let req = self.get("/v1/models").and_then(|req| {
            req.body(http_client::NoBody)
                .map_err(http_client::Error::Protocol)
        })?;

        let response = self.send(req).await?;

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
    type Error = crate::client::ProviderClientError;

    /// Create a new Mira client from the `MIRA_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("MIRA_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

/// Mira completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    crate::providers::openai::completion::GenericCompletionModel<MiraExt, H>;

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type OutputMessage = ChatChoice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        match self {
            Self::Structured { id, .. } => Some(id.clone()),
            Self::Simple(_) => None,
        }
    }

    fn get_response_model_name(&self) -> Option<String> {
        match self {
            Self::Structured { model, .. } => Some(model.clone()),
            Self::Simple(_) => None,
        }
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        match self {
            Self::Structured { choices, .. } => choices
                .iter()
                .map(|choice| ChatChoice {
                    message: choice.message.clone(),
                    finish_reason: choice.finish_reason.clone(),
                    index: choice.index,
                })
                .collect(),
            Self::Simple(_) => Vec::new(),
        }
    }

    fn get_text_response(&self) -> Option<String> {
        match self {
            Self::Structured { choices, .. } => choices
                .iter()
                .find(|choice| choice.message.role == "assistant")
                .map(|choice| choice.message.content.clone()),
            Self::Simple(text) => Some(text.clone()),
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        match self {
            Self::Structured { usage, .. } => usage.clone(),
            Self::Simple(_) => None,
        }
    }
}

impl crate::completion::GetTokenUsage for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = (self.total_tokens - self.prompt_tokens) as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage
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
                        cached_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        tool_use_prompt_tokens: 0,
                        reasoning_tokens: 0,
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
                    Message::System { .. } => {
                        tracing::warn!(target: "rig", "Received system message in response where assistant message was expected");
                        return Err(CompletionError::ResponseError(
                            "Received system message in response where assistant message was expected".to_owned(),
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
            message_id: None,
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

#[cfg(test)]
mod tests {
    use super::*;

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
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::mira::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::mira::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    // Proves a non-success HTTP response from `/v1/chat/completions` preserves
    // the provider's status + body through the `provider_response_*` helpers
    // (issue #1931).
    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("deepseek-r1");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
