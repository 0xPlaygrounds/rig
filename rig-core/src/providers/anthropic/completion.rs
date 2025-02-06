//! Anthropic completion api implementation

use std::{convert::Infallible, str::FromStr};

use crate::{
    completion::{self, CompletionError},
    json_utils,
    message::{self, MessageError},
    one_or_many::string_or_one_or_many,
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

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let content = response
            .content
            .iter()
            .map(|content| {
                Ok(match content {
                    Content::Text { text } => completion::AssistantContent::text(text),
                    Content::ToolUse { id, name, input } => {
                        completion::AssistantContent::tool_call(id, name, input.clone())
                    }
                    _ => {
                        return Err(CompletionError::ResponseError(
                            "Response did not contain a message or tool call".into(),
                        ))
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

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

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(deserialize_with = "string_or_one_or_many")]
    pub content: OneOrMany<Content>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Document {
        source: DocumentSource,
    },
}

impl FromStr for Content {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Content::Text { text: s.to_owned() })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image(ImageSource),
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ToolResultContent::Text { text: s.to_owned() })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ImageSource {
    pub data: String,
    pub media_type: ImageFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct DocumentSource {
    pub data: String,
    pub media_type: DocumentFormat,
    pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    #[serde(rename = "application/pdf")]
    PDF,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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

impl From<SourceType> for message::ContentFormat {
    fn from(source_type: SourceType) -> Self {
        match source_type {
            SourceType::BASE64 => message::ContentFormat::Base64,
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
            message::AssistantContent::Text(message::Text { text }) => Content::Text { text },
            message::AssistantContent::ToolCall(message::ToolCall { id, function }) => {
                Content::ToolUse {
                    id,
                    name: function.name,
                    input: function.arguments,
                }
            }
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
                    message::UserContent::Text(message::Text { text }) => {
                        Ok(Content::Text { text })
                    }
                    message::UserContent::ToolResult(message::ToolResult { id, content }) => {
                        Ok(Content::ToolResult {
                            tool_use_id: id,
                            content: content.try_map(|content| match content {
                                message::ToolResultContent::Text(message::Text { text }) => {
                                    Ok(ToolResultContent::Text { text })
                                }
                                message::ToolResultContent::Image(image) => {
                                    let media_type =
                                        image.media_type.ok_or(MessageError::ConversionError(
                                            "Image media type is required".to_owned(),
                                        ))?;
                                    let format =
                                        image.format.ok_or(MessageError::ConversionError(
                                            "Image format is required".to_owned(),
                                        ))?;
                                    Ok(ToolResultContent::Image(ImageSource {
                                        data: image.data,
                                        media_type: media_type.try_into()?,
                                        r#type: format.try_into()?,
                                    }))
                                }
                            })?,
                            is_error: None,
                        })
                    }
                    message::UserContent::Image(message::Image {
                        data,
                        format,
                        media_type,
                        ..
                    }) => {
                        let source = ImageSource {
                            data,
                            media_type: match media_type {
                                Some(media_type) => media_type.try_into()?,
                                None => {
                                    return Err(MessageError::ConversionError(
                                        "Image media type is required".to_owned(),
                                    ))
                                }
                            },
                            r#type: match format {
                                Some(format) => format.try_into()?,
                                None => SourceType::BASE64,
                            },
                        };
                        Ok(Content::Image { source })
                    }
                    message::UserContent::Document(message::Document { data, format, .. }) => {
                        let source = DocumentSource {
                            data,
                            media_type: DocumentFormat::PDF,
                            r#type: match format {
                                Some(format) => format.try_into()?,
                                None => SourceType::BASE64,
                            },
                        };
                        Ok(Content::Document { source })
                    }
                    message::UserContent::Audio { .. } => Err(MessageError::ConversionError(
                        "Audio is not supported in Anthropic".to_owned(),
                    )),
                })?,
            },

            message::Message::Assistant { content } => Message {
                content: content.map(|content| content.into()),
                role: Role::Assistant,
            },
        })
    }
}

impl TryFrom<Content> for message::AssistantContent {
    type Error = MessageError;

    fn try_from(content: Content) -> Result<Self, Self::Error> {
        Ok(match content {
            Content::Text { text } => message::AssistantContent::text(text),
            Content::ToolUse { id, name, input } => {
                message::AssistantContent::tool_call(id, name, input)
            }
            _ => {
                return Err(MessageError::ConversionError(
                    format!("Unsupported content type for Assistant role: {:?}", content)
                        .to_owned(),
                ))
            }
        })
    }
}

impl From<ToolResultContent> for message::ToolResultContent {
    fn from(content: ToolResultContent) -> Self {
        match content {
            ToolResultContent::Text { text } => message::ToolResultContent::text(text),
            ToolResultContent::Image(ImageSource {
                data,
                media_type: format,
                r#type,
            }) => message::ToolResultContent::image(
                data,
                Some(r#type.into()),
                Some(format.into()),
                None,
            ),
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message.role {
            Role::User => message::Message::User {
                content: message.content.try_map(|content| {
                    Ok(match content {
                        Content::Text { text } => message::UserContent::text(text),
                        Content::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => message::UserContent::tool_result(
                            tool_use_id,
                            content.map(|content| content.into()),
                        ),
                        Content::Image { source } => message::UserContent::Image(message::Image {
                            data: source.data,
                            format: Some(message::ContentFormat::Base64),
                            media_type: Some(source.media_type.into()),
                            detail: None,
                        }),
                        Content::Document { source } => message::UserContent::document(
                            source.data,
                            Some(message::ContentFormat::Base64),
                            Some(message::DocumentMediaType::PDF),
                        ),
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

                _ => {
                    return Err(MessageError::ConversionError(
                        format!("Unsupported message for Assistant role: {:?}", message).to_owned(),
                    ))
                }
            },
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
    pub default_max_tokens: Option<u64>,
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

/// Anthropic requires a `max_tokens` parameter to be set, which is dependent on the model. If not
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

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    Any,
    Tool {
        name: String,
    },
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

        tracing::debug!("Anthropic completion request: {request}");

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_path_to_error::deserialize;

    #[test]
    fn test_deserialize_message() {
        let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

        let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                }
            ]
        }
        "#;

        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "15 degrees"
                }
            ]
        }
        "#;

        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        let user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };

        match assistant_message {
            Message { role, content } => {
                assert_eq!(role, Role::Assistant);
                assert_eq!(
                    content.first(),
                    Content::Text {
                        text: "\n\nHello there, how may I assist you today?".to_owned()
                    }
                );
            }
        }

        match assistant_message2 {
            Message { role, content } => {
                assert_eq!(role, Role::Assistant);
                assert_eq!(content.len(), 2);

                let mut iter = content.into_iter();

                match iter.next().unwrap() {
                    Content::Text { text } => {
                        assert_eq!(text, "\n\nHello there, how may I assist you today?");
                    }
                    _ => panic!("Expected text content"),
                }

                match iter.next().unwrap() {
                    Content::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                        assert_eq!(name, "get_weather");
                        assert_eq!(input, json!({"location": "San Francisco, CA"}));
                    }
                    _ => panic!("Expected tool use content"),
                }

                assert_eq!(iter.next(), None);
            }
        }

        match user_message {
            Message { role, content } => {
                assert_eq!(role, Role::User);
                assert_eq!(content.len(), 3);

                let mut iter = content.into_iter();

                match iter.next().unwrap() {
                    Content::Image { source } => {
                        assert_eq!(
                            source,
                            ImageSource {
                                data: "/9j/4AAQSkZJRg...".to_owned(),
                                media_type: ImageFormat::JPEG,
                                r#type: SourceType::BASE64,
                            }
                        );
                    }
                    _ => panic!("Expected image content"),
                }

                match iter.next().unwrap() {
                    Content::Text { text } => {
                        assert_eq!(text, "What is in this image?");
                    }
                    _ => panic!("Expected text content"),
                }

                match iter.next().unwrap() {
                    Content::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        assert_eq!(tool_use_id, "toolu_01A09q90qw90lq917835lq9");
                        assert_eq!(
                            content.first(),
                            ToolResultContent::Text {
                                text: "15 degrees".to_owned()
                            }
                        );
                        assert_eq!(is_error, None);
                    }
                    _ => panic!("Expected tool result content"),
                }

                assert_eq!(iter.next(), None);
            }
        }
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message: Message = serde_json::from_str(
            r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "base64_encoded_pdf_data",
                        "media_type": "application/pdf"
                    }
                }
            ]
        }
        "#,
        )
        .unwrap();

        let assistant_message = Message {
            role: Role::Assistant,
            content: OneOrMany::one(Content::ToolUse {
                id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "San Francisco, CA"}),
            }),
        };

        let tool_message = Message {
            role: Role::User,
            content: OneOrMany::one(Content::ToolResult {
                tool_use_id: "toolu_01A09q90qw90lq917835lq9".to_string(),
                content: OneOrMany::one(ToolResultContent::Text {
                    text: "15 degrees".to_string(),
                }),
                is_error: None,
            }),
        };

        let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: message::Message =
            assistant_message.clone().try_into().unwrap();
        let converted_tool_message: message::Message = tool_message.clone().try_into().unwrap();

        match converted_user_message.clone() {
            message::Message::User { content } => {
                assert_eq!(content.len(), 3);

                let mut iter = content.into_iter();

                match iter.next().unwrap() {
                    message::UserContent::Image(message::Image {
                        data,
                        format,
                        media_type,
                        ..
                    }) => {
                        assert_eq!(data, "/9j/4AAQSkZJRg...");
                        assert_eq!(format.unwrap(), message::ContentFormat::Base64);
                        assert_eq!(media_type, Some(message::ImageMediaType::JPEG));
                    }
                    _ => panic!("Expected image content"),
                }

                match iter.next().unwrap() {
                    message::UserContent::Text(message::Text { text }) => {
                        assert_eq!(text, "What is in this image?");
                    }
                    _ => panic!("Expected text content"),
                }

                match iter.next().unwrap() {
                    message::UserContent::Document(message::Document {
                        data,
                        format,
                        media_type,
                    }) => {
                        assert_eq!(data, "base64_encoded_pdf_data");
                        assert_eq!(format.unwrap(), message::ContentFormat::Base64);
                        assert_eq!(media_type, Some(message::DocumentMediaType::PDF));
                    }
                    _ => panic!("Expected document content"),
                }

                assert_eq!(iter.next(), None);
            }
            _ => panic!("Expected user message"),
        }

        match converted_tool_message.clone() {
            message::Message::User { content } => {
                let message::ToolResult { id, content, .. } = match content.first() {
                    message::UserContent::ToolResult(tool_result) => tool_result,
                    _ => panic!("Expected tool result content"),
                };
                assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                match content.first() {
                    message::ToolResultContent::Text(message::Text { text }) => {
                        assert_eq!(text, "15 degrees");
                    }
                    _ => panic!("Expected text content"),
                }
            }
            _ => panic!("Expected tool result content"),
        }

        match converted_assistant_message.clone() {
            message::Message::Assistant { content } => {
                assert_eq!(content.len(), 1);

                match content.first() {
                    message::AssistantContent::ToolCall(message::ToolCall { id, function }) => {
                        assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                        assert_eq!(function.name, "get_weather");
                        assert_eq!(function.arguments, json!({"location": "San Francisco, CA"}));
                    }
                    _ => panic!("Expected tool call content"),
                }
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: Message = converted_user_message.try_into().unwrap();
        let original_assistant_message: Message = converted_assistant_message.try_into().unwrap();
        let original_tool_message: Message = converted_tool_message.try_into().unwrap();

        assert_eq!(user_message, original_user_message);
        assert_eq!(assistant_message, original_assistant_message);
        assert_eq!(tool_message, original_tool_message);
    }
}
