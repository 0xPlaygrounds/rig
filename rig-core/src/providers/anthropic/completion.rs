//! Anthropic completion api implementation

use crate::{
    completion::{self, CompletionError},
    json_utils,
    message::{self, MessageError},
};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;

// ================================================================
// Anthropic Completion API
// ================================================================
/// `claude-3-5-sonnet-20240620` completion model
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-20240620";

/// `claude-3-5-haiku-20240620` completion model
pub const CLAUDE_3_OPUS: &str = "claude-3-opus-20240229";

/// `claude-3-sonnet-20240229` completion model
pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet-20240229";

/// `claude-3-haiku-20240307` completion model
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub output_tokens: u64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nCache read input tokens: {}\nCache creation input tokens: {}\nOutput tokens: {}",
            self.input_tokens,
            match self.cache_read_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            match self.cache_creation_input_tokens {
                Some(token) => token.to_string(),
                None => "n/a".to_string(),
            },
            self.output_tokens
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> std::prelude::v1::Result<Self, Self::Error> {
        match response.content.as_slice() {
            [Content::Text { text, .. }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(text.to_string()),
                raw_response: response,
            }),
            [Content::ToolUse { name, input, .. }, ..] => Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::ToolCall(name.clone(), input.clone()),
                raw_response: response,
            }),
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a message or tool call".into(),
            )),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        is_error: bool,
    },
    Document {
        source: DocumentSource,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image(ImageSource),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ImageSource {
    pub data: String,
    pub format: ImageFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentSource {
    pub data: String,
    pub format: DocumentFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    #[serde(rename = "image/jpeg")]
    JPEG,
    #[serde(rename = "image/png")]
    PNG,
    #[serde(rename = "image/gif")]
    GIF,
    #[serde(rename = "image/webp")]
    WEBP,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    #[serde(rename = "application/pdf")]
    PDF,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    BASE64,
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text { text }
    }
}

impl From<String> for ToolResultContent {
    fn from(text: String) -> Self {
        ToolResultContent::Text { text }
    }
}

impl TryFrom<message::ContentFormat> for SourceType {
    type Error = MessageError;

    fn try_from(format: message::ContentFormat) -> Result<Self, Self::Error> {
        match format {
            message::ContentFormat::Base64 => Ok(SourceType::BASE64),
            message::ContentFormat::String => Err(MessageError::ConversionError(
                "Image urls are not supported in Anthropic".to_owned(),
            )),
        }
    }
}

impl From<message::ImageMediaType> for ImageFormat {
    fn from(media_type: message::ImageMediaType) -> Self {
        match media_type {
            message::ImageMediaType::JPEG => ImageFormat::JPEG,
            message::ImageMediaType::PNG => ImageFormat::PNG,
            message::ImageMediaType::GIF => ImageFormat::GIF,
            message::ImageMediaType::WEBP => ImageFormat::WEBP,
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => content
                .into_iter()
                .map(|content| match content {
                    message::UserContent::Text { text } => Ok(Content::Text { text }),
                    message::UserContent::Image {
                        data,
                        format,
                        media_type,
                        ..
                    } => {
                        let source = ImageSource {
                            data,
                            format: media_type.into(),
                            r#type: format.try_into()?,
                        };
                        Ok(Content::Image { source })
                    }
                    message::UserContent::Document { data, format, .. } => {
                        let source = DocumentSource {
                            data,
                            format: DocumentFormat::PDF,
                            r#type: format.try_into()?,
                        };
                        Ok(Content::Document { source })
                    }
                    message::UserContent::Audio { .. } => Err(MessageError::ConversionError(
                        "Audio is not supported in Anthropic".to_owned(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|content| Message {
                    role: "user".to_owned(),
                    content,
                })
                .collect::<Vec<_>>(),

            message::Message::Assistant {
                content,
                tool_calls,
            } => content
                .into_iter()
                .map(|content| Message {
                    role: "assistant".to_owned(),
                    content: content.into(),
                })
                .chain(tool_calls.into_iter().map(|tool_call| Message {
                    role: "assistant".to_owned(),
                    content: Content::ToolUse {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        input: tool_call.function.arguments,
                    },
                }))
                .collect::<Vec<_>>(),

            message::Message::Tool { id, content } => vec![Message {
                role: "assistant".to_owned(),
                content: Content::ToolResult {
                    tool_use_id: id,
                    content: content.into(),
                    is_error: false,
                },
            }],
        })
    }
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

#[derive(Debug, Deserialize, Serialize)]
struct Metadata {
    user_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Note: Ideally we'd introduce provider-specific Request models to handle the
        // specific requirements of each provider. For now, we just manually check while
        // building the request as a raw JSON document.

        // Check if max_tokens is set, required for Anthropic
        if completion_request.max_tokens.is_none() {
            return Err(CompletionError::RequestError(
                "max_tokens must be set for Anthropic".into(),
            ));
        }

        let prompt_message: Vec<Message> = completion_request
            .prompt_with_context()
            .try_into()
            .map_err(|e: MessageError| CompletionError::RequestError(e.to_string().into()))?;

        let mut messages = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                message
                    .try_into()
                    .map_err(|e: MessageError| CompletionError::RequestError(e.to_string().into()))
            })
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        messages.extend(prompt_message);

        let mut request = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": completion_request.max_tokens,
            "system": completion_request.preamble.unwrap_or("".to_string()),
        });

        if let Some(temperature) = completion_request.temperature {
            json_utils::merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        if !completion_request.tools.is_empty() {
            json_utils::merge_inplace(
                &mut request,
                json!({
                    "tools": completion_request
                        .tools
                        .into_iter()
                        .map(|tool| ToolDefinition {
                            name: tool.name,
                            description: Some(tool.description),
                            input_schema: tool.parameters,
                        })
                        .collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }

        if let Some(ref params) = completion_request.additional_params {
            json_utils::merge_inplace(&mut request, params.clone())
        }

        let response = self
            .client
            .post("/v1/messages")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Message(completion) => {
                    tracing::info!(target: "rig",
                        "Anthropic completion token usage: {}",
                        completion.usage
                    );
                    completion.try_into()
                }
                ApiResponse::Error(error) => Err(CompletionError::ProviderError(error.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}
