use crate::OneOrMany;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ================================================================
// Request models
// ================================================================
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User { content: OneOrMany<UserContent> },
    Assistant(AssistantContent),
    ToolResult { id: String, content: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContent {
    Content { content: OneOrMany<String> },
    ToolCalls { tool_calls: OneOrMany<ToolCall> },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    Image {
        data: String,
        format: Option<ContentFormat>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    },
    Audio {
        data: String,
        format: Option<ContentFormat>,
        media_type: Option<AudioMediaType>,
    },
    Document {
        data: String,
        format: Option<ContentFormat>,
        media_type: Option<DocumentMediaType>,
    },
}

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

impl ImageMediaType {
    pub fn from_mime_type(mime_type: &str) -> Option<Self> {
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

    pub fn to_mime_type(&self) -> &'static str {
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

impl DocumentMediaType {
    pub fn from_mime_type(mime_type: &str) -> Option<Self> {
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

    pub fn to_mime_type(&self) -> &'static str {
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

impl AudioMediaType {
    pub fn from_mime_type(mime_type: &str) -> Option<Self> {
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

    pub fn to_mime_type(&self) -> &'static str {
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

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message::User {
            content: OneOrMany::<UserContent>::one(UserContent::Text { text }),
        }
    }
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message::User {
            content: OneOrMany::<UserContent>::one(UserContent::Text {
                text: text.to_owned(),
            }),
        }
    }
}

impl Message {
    pub fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                if let UserContent::Text { text } = content.first() {
                    Some(text.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn create_user(text: impl Into<String>) -> Self {
        Message::User {
            content: OneOrMany::<UserContent>::one(UserContent::Text { text: text.into() }),
        }
    }

    pub fn create_assistant(text: impl Into<String>) -> Self {
        Message::Assistant(AssistantContent::Content {
            content: OneOrMany::<String>::one(text.into()),
        })
    }
}

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}
