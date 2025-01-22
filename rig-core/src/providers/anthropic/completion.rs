//! Anthropic completion api implementation

use crate::{
    completion::{self, CompletionError},
    json_utils,
    message::{self, MessageError},
    OneOrMany,
};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::client::Client;

// ================================================================
// Anthropic Completion API
// ================================================================
/// `claude-3-5-sonnet-latest` completion model
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";

/// `claude-3-5-haiku-latest` completion model
pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-latest";

/// `claude-3-5-haiku-latest` completion model
pub const CLAUDE_3_OPUS: &str = "claude-3-opus-latest";

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
        if let Some(tool_use) = response.content.iter().find_map(|content| match content {
            Content::ToolUse {
                name, input, id, ..
            } => Some((name.clone(), id.clone(), input.clone())),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::ToolCall(tool_use.0, tool_use.1, tool_use.2),
                raw_response: response,
            });
        }

        if let Some(text_content) = response.content.iter().find_map(|content| match content {
            Content::Text { text, .. } => Some(text.clone()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: completion::ModelChoice::Message(text_content),
                raw_response: response,
            });
        }

        Err(CompletionError::ResponseError(
            "Response did not contain a message or tool call".into(),
        ))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: OneOrMany<Content>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image(ImageSource),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageSource {
    pub data: String,
    pub format: ImageFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DocumentSource {
    pub data: String,
    pub format: DocumentFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    #[serde(rename = "application/pdf")]
    PDF,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

impl TryFrom<message::ImageMediaType> for ImageFormat {
    type Error = MessageError;

    fn try_from(media_type: message::ImageMediaType) -> Result<Self, Self::Error> {
        Ok(match media_type {
            message::ImageMediaType::JPEG => ImageFormat::JPEG,
            message::ImageMediaType::PNG => ImageFormat::PNG,
            message::ImageMediaType::GIF => ImageFormat::GIF,
            message::ImageMediaType::WEBP => ImageFormat::WEBP,
            _ => {
                return Err(MessageError::ConversionError(
                    format!("Unsupported image media type: {:?}", media_type).to_owned(),
                ))
            }
        })
    }
}

impl From<ImageFormat> for message::ImageMediaType {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::JPEG => message::ImageMediaType::JPEG,
            ImageFormat::PNG => message::ImageMediaType::PNG,
            ImageFormat::GIF => message::ImageMediaType::GIF,
            ImageFormat::WEBP => message::ImageMediaType::WEBP,
        }
    }
}

impl From<message::AssistantContent> for Content {
    fn from(text: message::AssistantContent) -> Self {
        match text {
            message::AssistantContent::Text { text } => Content::Text { text },
            message::AssistantContent::ToolCall { tool_call } => Content::ToolUse {
                id: tool_call.id,
                name: tool_call.function.name,
                input: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<message::Message> for Message {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message {
                role: Role::User,
                content: content.try_map(|content| match content {
                    message::UserContent::Text { text } => Ok(Content::Text { text }),
                    message::UserContent::Image {
                        data,
                        format,
                        media_type,
                        ..
                    } => {
                        let source = ImageSource {
                            data,
                            format: match media_type {
                                Some(media_type) => media_type.try_into()?,
                                None => ImageFormat::JPEG,
                            },
                            r#type: match format {
                                Some(format) => format.try_into()?,
                                None => SourceType::BASE64,
                            },
                        };
                        Ok(Content::Image { source })
                    }
                    message::UserContent::Document { data, format, .. } => {
                        let source = DocumentSource {
                            data,
                            format: DocumentFormat::PDF,
                            r#type: match format {
                                Some(format) => format.try_into()?,
                                None => SourceType::BASE64,
                            },
                        };
                        Ok(Content::Document { source })
                    }
                    message::UserContent::Audio { .. } => {
                        return Err(MessageError::ConversionError(
                            "Audio is not supported in Anthropic".to_owned(),
                        ))
                    }
                })?,
            },

            message::Message::Assistant { content } => Message {
                content: content.map(|content| content.into()),
                role: Role::Assistant,
            },

            message::Message::ToolResult { id, content } => Message {
                role: Role::Assistant,
                content: OneOrMany::one(Content::ToolResult {
                    tool_use_id: id,
                    content: content.into(),
                    is_error: false,
                }),
            },
        })
    }
}

impl TryFrom<Content> for message::AssistantContent {
    type Error = MessageError;

    fn try_from(content: Content) -> Result<Self, Self::Error> {
        Ok(match content {
            Content::Text { text } => message::AssistantContent::Text { text },
            Content::ToolUse { id, name, input } => message::AssistantContent::ToolCall {
                tool_call: message::ToolCall {
                    id,
                    function: message::ToolFunction {
                        name,
                        arguments: input,
                    },
                },
            },
            _ => {
                return Err(MessageError::ConversionError(
                    format!("Unsupported content type for Assistant role: {:?}", content)
                        .to_owned(),
                ))
            }
        })
    }
}

impl TryFrom<Message> for message::Message {
    type Error = MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message.role {
            Role::User => message::Message::User {
                content: message.content.try_map(|content| {
                    Ok(match content {
                        Content::Text { text } => message::UserContent::Text { text },
                        Content::Image { source } => message::UserContent::Image {
                            data: source.data,
                            format: Some(message::ContentFormat::Base64),
                            media_type: Some(source.format.into()),
                            detail: None,
                        },
                        Content::Document { source } => message::UserContent::Document {
                            data: source.data,
                            format: Some(message::ContentFormat::Base64),
                            media_type: Some(message::DocumentMediaType::PDF),
                        },
                        _ => {
                            return Err(MessageError::ConversionError(
                                "Unsupported content type for User role".to_owned(),
                            ))
                        }
                    })
                })?,
            },
            Role::Assistant => match message.content.first() {
                Content::Text { .. } | Content::ToolUse { .. } => message::Message::Assistant {
                    content: message.content.try_map(|content| content.try_into())?,
                },
                Content::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => message::Message::ToolResult {
                    id: tool_use_id,
                    content: match content {
                        ToolResultContent::Text { text } => text,
                        _ => {
                            return Err(MessageError::ConversionError(
                                format!(
                                    "Unsupported tool result content type for Assistant role: {:?}",
                                    message.content.first()
                                )
                                .to_owned(),
                            ))
                        }
                    },
                },
                _ => {
                    return Err(MessageError::ConversionError(
                        "Unsupported content type for Assistant role".to_owned(),
                    ))
                }
            },
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
    default_max_tokens: Option<u64>,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
            default_max_tokens: calculate_max_tokens(model),
        }
    }
}

/// Anthropic requires a `max_tokens` parameter to be set, which is dependant on the model. If not
/// set or if set too high, the request will fail. The following values are based on the models
/// available at the time of writing.
///
/// Dev Note: This is really bad design, I'm not sure why they did it like this..
fn calculate_max_tokens(model: &str) -> Option<u64> {
    if model.starts_with("claude-3-5-sonnet") || model.starts_with("claude-3-5-haiku") {
        Some(8192)
    } else if model.starts_with("claude-3-opus")
        || model.starts_with("claude-3-sonnet")
        || model.starts_with("claude-3-haiku")
    {
        Some(4096)
    } else {
        None
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Note: Ideally we'd introduce provider-specific Request models to handle the
        // specific requirements of each provider. For now, we just manually check while
        // building the request as a raw JSON document.

        // Check if max_tokens is set, required for Anthropic
        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let prompt_message: Message = completion_request
            .prompt_with_context()
            .try_into()
            .map_err(|e: MessageError| CompletionError::RequestError(e.into()))?;

        let mut messages = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                message
                    .try_into()
                    .map_err(|e: MessageError| CompletionError::RequestError(e.into()))
            })
            .collect::<Result<Vec<Message>, _>>()?;

        messages.push(prompt_message);

        let mut request = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
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
