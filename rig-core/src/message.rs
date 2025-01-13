use crate::OneOrMany;
use serde::{Deserialize, Serialize};

// ================================================================
// Request models
// ================================================================
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: OneOrMany<String>,
    },
    User {
        content: OneOrMany<UserContent>,
    },
    Assistant {
        content: OneOrMany<String>,
        tool_calls: OneOrMany<ToolCall>,
    },
    Tool {
        id: String,
        content: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    Image {
        data: String,
        format: ContentFormat,
        detail: ImageDetail,
        r#media_type: MediaType,
    },
    Document {
        data: String,
        format: ContentFormat,
        r#media_type: String,
    },
    Audio {
        data: String,
        format: ContentFormat,
        r#media_type: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    Base64,
    String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaType {
    #[serde(rename = "application/pdf")]
    ApplicationPdf,
    #[serde(rename = "image/jpeg")]
    ImageJpeg,
    #[serde(rename = "image/png")]
    ImagePng,
    #[serde(rename = "image/gif")]
    ImageGif,
    #[serde(rename = "image/webp")]
    ImageWebp,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

impl Message {
    pub fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                if let UserContent::Text { text }= content.first() {
                    Some(text.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
