use crate::OneOrMany;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ================================================================
// Request models
// ================================================================
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: OneOrMany<UserContent>,
    },
    Assistant {
        content: Vec<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        id: String,
        content: String,
    },
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
        format: ContentFormat,
        detail: ImageDetail,
        media_type: ImageMediaType,
    },
    Document {
        data: String,
        format: ContentFormat,
        media_type: DocumentMediaType,
    },
    Audio {
        data: String,
        format: ContentFormat,
        media_type: AudioMediaType,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    Base64,
    String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    PNG,
    GIF,
    WEBP,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    WAV,
    MP4,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
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
}

#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}
