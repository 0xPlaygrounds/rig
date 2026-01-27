//! xAI Responses API types
//!
//! Types for the xAI Responses API: <https://docs.x.ai/docs/guides/chat>
//!
//! This module reuses OpenAI's Responses API types where compatible,
//! since xAI's API format is designed to be compatible with OpenAI.

use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError};
use crate::message::{Message as RigMessage, MimeType};

// ================================================================
// Request Types
// ================================================================

/// Input item for xAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub enum Message {
    /// A message
    Message { role: Role, content: Content },
    /// A function call from the assistant
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    /// A function call output/result
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Array(Vec<ContentItem>),
}

/// Content item types for multimodal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    Image {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    #[serde(rename = "input_file")]
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::Message {
            role: Role::System,
            content: Content::Text(content.into()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::Message {
            role: Role::User,
            content: Content::Text(content.into()),
        }
    }

    pub fn user_with_content(content: Vec<ContentItem>) -> Self {
        Self::Message {
            role: Role::User,
            content: Content::Array(content),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Message {
            role: Role::Assistant,
            content: Content::Text(content.into()),
        }
    }

    pub fn function_call(call_id: String, name: String, arguments: String) -> Self {
        Self::FunctionCall {
            call_id,
            name,
            arguments,
        }
    }

    pub fn function_call_output(call_id: String, output: String) -> Self {
        Self::FunctionCallOutput { call_id, output }
    }
}

impl TryFrom<RigMessage> for Vec<Message> {
    type Error = CompletionError;

    fn try_from(msg: RigMessage) -> Result<Self, Self::Error> {
        use crate::message::{
            AssistantContent, Document, DocumentSourceKind, Image as RigImage, Text,
            ToolResultContent, UserContent,
        };

        fn image_item(img: RigImage) -> Result<ContentItem, CompletionError> {
            let url = match img.data {
                DocumentSourceKind::Url(u) => u,
                DocumentSourceKind::Base64(data) => {
                    let mime = img
                        .media_type
                        .map(|m| m.to_mime_type())
                        .unwrap_or("image/png");
                    format!("data:{mime};base64,{data}")
                }
                _ => {
                    return Err(CompletionError::RequestError(
                        "xAI does not support raw image data; use base64 or URL".into(),
                    ));
                }
            };
            Ok(ContentItem::Image {
                image_url: url,
                detail: img.detail.map(|d| format!("{d:?}").to_lowercase()),
            })
        }

        fn document_item(doc: Document) -> Result<ContentItem, CompletionError> {
            let (file_data, file_url) = match doc.data {
                DocumentSourceKind::Url(url) => (None, Some(url)),
                DocumentSourceKind::Base64(data) => {
                    let mime = doc
                        .media_type
                        .map(|m| m.to_mime_type())
                        .unwrap_or("application/pdf");
                    (Some(format!("data:{mime};base64,{data}")), None)
                }
                DocumentSourceKind::String(text) => {
                    // Plain text document - just return as text
                    return Ok(ContentItem::Text { text });
                }
                _ => {
                    return Err(CompletionError::RequestError(
                        "xAI does not support raw document data; use base64 or URL".into(),
                    ));
                }
            };
            Ok(ContentItem::File {
                file_url,
                file_data,
            })
        }

        match msg {
            RigMessage::User { content } => {
                let mut items = Vec::new();
                let mut text_parts = Vec::new();
                let mut content_items = Vec::new();
                let mut has_images = false;

                for c in content {
                    match c {
                        UserContent::Text(Text { text }) => text_parts.push(text),
                        UserContent::Image(img) => {
                            has_images = true;
                            content_items.push(image_item(img)?);
                        }
                        UserContent::ToolResult(tr) => {
                            // Flush accumulated text/images as a message first
                            if has_images {
                                let mut msg_items: Vec<_> = text_parts
                                    .drain(..)
                                    .map(|t| ContentItem::Text { text: t })
                                    .collect();
                                msg_items.append(&mut content_items);
                                if !msg_items.is_empty() {
                                    items.push(Message::user_with_content(msg_items));
                                }
                            } else if !text_parts.is_empty() {
                                items.push(Message::user(text_parts.join("\n")));
                            }
                            has_images = false;

                            // Tool result becomes FunctionCallOutput
                            let output = tr
                                .content
                                .into_iter()
                                .map(|tc| match tc {
                                    ToolResultContent::Text(t) => Ok(t.text),
                                    ToolResultContent::Image(_) => {
                                        Err(CompletionError::RequestError(
                                            "xAI does not support images in tool results".into(),
                                        ))
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .join("\n");
                            items.push(Message::function_call_output(
                                tr.call_id.unwrap_or_default(),
                                output,
                            ));
                        }
                        UserContent::Document(doc) => {
                            has_images = true; // Force array format for files
                            content_items.push(document_item(doc)?);
                        }
                        UserContent::Audio(_) => {
                            return Err(CompletionError::RequestError(
                                "xAI does not support audio".into(),
                            ));
                        }
                        UserContent::Video(_) => {
                            return Err(CompletionError::RequestError(
                                "xAI does not support video".into(),
                            ));
                        }
                    }
                }

                // Flush remaining text/images
                if has_images {
                    let mut msg_items: Vec<_> = text_parts
                        .into_iter()
                        .map(|t| ContentItem::Text { text: t })
                        .collect();
                    msg_items.append(&mut content_items);
                    if !msg_items.is_empty() {
                        items.push(Message::user_with_content(msg_items));
                    }
                } else if !text_parts.is_empty() {
                    items.push(Message::user(text_parts.join("\n")));
                }

                Ok(items)
            }
            RigMessage::Assistant { content, .. } => {
                let mut items = Vec::new();
                let mut text_parts = Vec::new();

                for c in content {
                    match c {
                        AssistantContent::Text(t) => text_parts.push(t.text),
                        AssistantContent::ToolCall(tc) => {
                            // Flush accumulated text as a message first
                            if !text_parts.is_empty() {
                                items.push(Message::assistant(text_parts.join("\n")));
                            }
                            // Tool call becomes FunctionCall
                            items.push(Message::function_call(
                                tc.call_id.unwrap_or_default(),
                                tc.function.name,
                                tc.function.arguments.to_string(),
                            ));
                        }
                        AssistantContent::Reasoning(r) => text_parts.extend(r.reasoning),
                        AssistantContent::Image(_) => {
                            return Err(CompletionError::RequestError(
                                "xAI does not support images in assistant content".into(),
                            ));
                        }
                    }
                }

                // Flush remaining text
                if !text_parts.is_empty() {
                    items.push(Message::assistant(text_parts.join("\n")));
                }

                Ok(items)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    #[serde(flatten)]
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            function: tool,
        }
    }
}

// ================================================================
// Error Types
// ================================================================

/// API error response
#[derive(Debug, Deserialize)]
pub struct ApiError {
    pub error: String,
    pub code: String,
}

impl ApiError {
    pub fn message(&self) -> String {
        format!("Code `{}`: {}", self.code, self.error)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Error(ApiError),
}
