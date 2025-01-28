use std::{convert::Infallible, str::FromStr};

use crate::OneOrMany;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompletionError;

// ================================================================
// Request models
// ================================================================

/// A message represents a run of input (user) and output (assistant).
/// Each message type (based on it's `role`) can contain a atleast one bit of content such as text,
///  images, audio, documents, or tool related information. While each message type can contain
///  multiple content, most often, you'll only see one content type per message
///  (an image w/ a description, etc).
///
/// Each provider is responsible with converting the generic message into it's provider specific
///  type using `From` or `TryFrom` traits. Since not every provider supports every feature, the
///  conversion can be lossy (providing an image might be discarded for a non-image supporting
///  provider) though the message being converted back and forth should always be the same.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// User message containing one or more content types defined by `UserContent`.
    User { content: OneOrMany<UserContent> },

    /// Assistant message containing one or more content types defined by `AssistantContent`.
    Assistant {
        content: OneOrMany<AssistantContent>,
    },
}

/// Describes the content of a message, which can be text, a tool result, an image, audio, or
///  a document. Dependent on provider supporting the content type. Multimedia content is generally
///  base64 (defined by it's format) encoded but additionally supports urls (for some providers).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text(Text),
    ToolResult(ToolResult),
    Image(Image),
    Audio(Audio),
    Document(Document),
}

/// Describes responses from a provider which is either text or a tool call.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContent {
    Text(Text),
    ToolCall(ToolCall),
}

/// Basic text content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    pub text: String,
}

/// Tool result content containing information about a tool call and it's resulting content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolResult {
    pub id: String,
    pub content: OneOrMany<ToolResultContent>,
}

/// Describes the content of a tool result, which can be text or an image.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum ToolResultContent {
    Text(Text),
    Image(Image),
}

/// Image content containing image data and metadata about it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Image {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ContentFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<ImageMediaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Audio content containing audio data and metadata about it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Audio {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ContentFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<AudioMediaType>,
}

/// Document content containing document data and metadata about it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Document {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ContentFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
}

/// Describes a tool call with an id and function to call, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolFunction,
}

/// Describes a tool function to call with a name and arguments, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Describes the format of the content, which can be base64 or string.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    Base64,
    String,
}

impl Default for ContentFormat {
    fn default() -> Self {
        ContentFormat::Base64
    }
}

/// Helper enum that tracks the media type of the content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum MediaType {
    Image(ImageMediaType),
    Audio(AudioMediaType),
    Document(DocumentMediaType),
}

impl MimeType for MediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        ImageMediaType::from_mime_type(mime_type)
            .map(MediaType::Image)
            .or_else(|| {
                DocumentMediaType::from_mime_type(mime_type)
                    .map(MediaType::Document)
                    .or_else(|| AudioMediaType::from_mime_type(mime_type).map(MediaType::Audio))
            })
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            MediaType::Image(media_type) => media_type.to_mime_type(),
            MediaType::Audio(media_type) => media_type.to_mime_type(),
            MediaType::Document(media_type) => media_type.to_mime_type(),
        }
    }
}

/// Describes the image media type of the content. Not every provider supports every media type.
/// Convertable to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    PNG,
    GIF,
    WEBP,
    HEIC,
    HEIF,
}

/// Describes the document media type of the content. Not every provider supports every media type.
/// Includes also programming languages as document types for providers who support code running.
/// Convertable to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
    TXT,
    RTF,
    HTML,
    CSS,
    MARKDOWN,
    CSV,
    XML,
    Javascript,
    Python,
}

/// Describes the audio media type of the content. Not every provider supports every media type.
/// Convertable to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    WAV,
    MP3,
    AIFF,
    AAC,
    OGG,
    FLAC,
}

/// Trait for converting between MIME types and media types.
pub trait MimeType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized;
    fn to_mime_type(&self) -> &'static str;
}

impl MimeType for ImageMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "image/jpeg" => Some(ImageMediaType::JPEG),
            "image/png" => Some(ImageMediaType::PNG),
            "image/gif" => Some(ImageMediaType::GIF),
            "image/webp" => Some(ImageMediaType::WEBP),
            "image/heic" => Some(ImageMediaType::HEIC),
            "image/heif" => Some(ImageMediaType::HEIF),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            ImageMediaType::JPEG => "image/jpeg",
            ImageMediaType::PNG => "image/png",
            ImageMediaType::GIF => "image/gif",
            ImageMediaType::WEBP => "image/webp",
            ImageMediaType::HEIC => "image/heic",
            ImageMediaType::HEIF => "image/heif",
        }
    }
}

impl MimeType for DocumentMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "application/pdf" => Some(DocumentMediaType::PDF),
            "text/plain" => Some(DocumentMediaType::TXT),
            "text/rtf" => Some(DocumentMediaType::RTF),
            "text/html" => Some(DocumentMediaType::HTML),
            "text/css" => Some(DocumentMediaType::CSS),
            "text/md" | "text/markdown" => Some(DocumentMediaType::MARKDOWN),
            "text/csv" => Some(DocumentMediaType::CSV),
            "text/xml" => Some(DocumentMediaType::XML),
            "application/x-javascript" | "text/x-javascript" => Some(DocumentMediaType::Javascript),
            "application/x-python" | "text/x-python" => Some(DocumentMediaType::Python),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            DocumentMediaType::PDF => "application/pdf",
            DocumentMediaType::TXT => "text/plain",
            DocumentMediaType::RTF => "text/rtf",
            DocumentMediaType::HTML => "text/html",
            DocumentMediaType::CSS => "text/css",
            DocumentMediaType::MARKDOWN => "text/markdown",
            DocumentMediaType::CSV => "text/csv",
            DocumentMediaType::XML => "text/xml",
            DocumentMediaType::Javascript => "application/x-javascript",
            DocumentMediaType::Python => "application/x-python",
        }
    }
}

impl MimeType for AudioMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "audio/wav" => Some(AudioMediaType::WAV),
            "audio/mp3" => Some(AudioMediaType::MP3),
            "audio/aiff" => Some(AudioMediaType::AIFF),
            "audio/aac" => Some(AudioMediaType::AAC),
            "audio/ogg" => Some(AudioMediaType::OGG),
            "audio/flac" => Some(AudioMediaType::FLAC),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            AudioMediaType::WAV => "audio/wav",
            AudioMediaType::MP3 => "audio/mp3",
            AudioMediaType::AIFF => "audio/aiff",
            AudioMediaType::AAC => "audio/aac",
            AudioMediaType::OGG => "audio/ogg",
            AudioMediaType::FLAC => "audio/flac",
        }
    }
}

/// Describes the detail of the image content, which can be low, high, or auto (open-ai specific).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

impl Default for ImageDetail {
    fn default() -> Self {
        ImageDetail::Auto
    }
}

impl std::str::FromStr for ImageDetail {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(ImageDetail::Low),
            "high" => Ok(ImageDetail::High),
            "auto" => Ok(ImageDetail::Auto),
            _ => Err(()),
        }
    }
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl FromStr for Text {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text.into())),
        }
    }
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::Text(text.into())),
        }
    }
}

impl Message {
    pub(crate) fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                if let UserContent::Text(Text { text }) = content.first() {
                    Some(text.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: OneOrMany::one(UserContent::text(text)),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            content: OneOrMany::one(AssistantContent::text(text)),
        }
    }
}

impl UserContent {
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into().into())
    }

    pub fn image(
        data: impl Into<String>,
        format: Option<ContentFormat>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: data.into(),
            format,
            media_type,
            detail,
        })
    }

    pub fn audio(
        data: impl Into<String>,
        format: Option<ContentFormat>,
        media_type: Option<AudioMediaType>,
    ) -> Self {
        UserContent::Audio(Audio {
            data: data.into(),
            format,
            media_type,
        })
    }

    pub fn document(
        data: impl Into<String>,
        format: Option<ContentFormat>,
        media_type: Option<DocumentMediaType>,
    ) -> Self {
        UserContent::Document(Document {
            data: data.into(),
            format,
            media_type,
        })
    }

    pub fn tool_result(id: impl Into<String>, content: OneOrMany<ToolResultContent>) -> Self {
        UserContent::ToolResult(ToolResult {
            id: id.into(),
            content,
        })
    }
}

impl AssistantContent {
    pub fn text(text: impl Into<String>) -> Self {
        AssistantContent::Text(text.into().into())
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall {
            id: id.into(),
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        })
    }
}

impl ToolResultContent {
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text(text.into().into())
    }

    pub fn image(
        data: impl Into<String>,
        format: Option<ContentFormat>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: data.into(),
            format,
            media_type,
            detail,
        })
    }
}

impl From<String> for ToolResultContent {
    fn from(text: String) -> Self {
        ToolResultContent::text(text)
    }
}

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}

impl From<MessageError> for CompletionError {
    fn from(error: MessageError) -> Self {
        CompletionError::RequestError(error.into())
    }
}
