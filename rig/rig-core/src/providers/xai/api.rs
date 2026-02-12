//! xAI Responses API types
//!
//! Types for the xAI Responses API: <https://docs.x.ai/docs/guides/chat>
//!
//! This module reuses OpenAI's Responses API types where compatible,
//! since xAI's API format is designed to be compatible with OpenAI.

use serde::{Deserialize, Serialize};

use crate::completion::{self, CompletionError};
use crate::message::{Message as RigMessage, MimeType, ReasoningContent};
use crate::providers::openai::responses_api::ReasoningSummary;

// ================================================================
// Request Types
// ================================================================

/// Input item for xAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
    /// A reasoning item returned by xAI/OpenAI-compatible Responses APIs.
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
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

    pub fn reasoning(
        id: String,
        summary: Vec<ReasoningSummary>,
        encrypted_content: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            encrypted_content,
        }
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

        fn reasoning_item(
            reasoning: crate::message::Reasoning,
        ) -> Result<Message, CompletionError> {
            let crate::message::Reasoning { id, content } = reasoning;
            let id = id.ok_or_else(|| {
                CompletionError::RequestError(
                    "Assistant reasoning `id` is required for xAI Responses replay".into(),
                )
            })?;
            let mut encrypted_content = None;
            let mut summary = Vec::new();
            for reasoning_content in content {
                match reasoning_content {
                    ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) => {
                        summary.push(ReasoningSummary::SummaryText { text });
                    }
                    // xAI's request shape has no dedicated redacted field, so preserve
                    // opaque reasoning data as encrypted content instead of plain text.
                    ReasoningContent::Redacted { data } | ReasoningContent::Encrypted(data) => {
                        encrypted_content.get_or_insert(data);
                    }
                }
            }

            Ok(Message::reasoning(id, summary, encrypted_content))
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
                            let call_id = tr.call_id.ok_or_else(|| {
                                CompletionError::RequestError(
                                    "Tool result `call_id` is required for xAI Responses API"
                                        .into(),
                                )
                            })?;
                            items.push(Message::function_call_output(call_id, output));
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
                let flush_assistant_text =
                    |items: &mut Vec<Message>, text_parts: &mut Vec<String>| {
                        if !text_parts.is_empty() {
                            items.push(Message::assistant(text_parts.join("\n")));
                            text_parts.clear();
                        }
                    };

                for c in content {
                    match c {
                        AssistantContent::Text(t) => text_parts.push(t.text),
                        AssistantContent::ToolCall(tc) => {
                            flush_assistant_text(&mut items, &mut text_parts);
                            // Tool call becomes FunctionCall
                            let call_id = tc.call_id.ok_or_else(|| {
                                CompletionError::RequestError(
                                    "Assistant tool call `call_id` is required for xAI Responses API"
                                        .into(),
                                )
                            })?;
                            items.push(Message::function_call(
                                call_id,
                                tc.function.name,
                                tc.function.arguments.to_string(),
                            ));
                        }
                        AssistantContent::Reasoning(r) => {
                            flush_assistant_text(&mut items, &mut text_parts);
                            items.push(reasoning_item(r)?);
                        }
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

#[cfg(test)]
mod tests {
    use super::Message;
    use crate::OneOrMany;
    use crate::completion::CompletionError;
    use crate::message::{AssistantContent, Message as RigMessage, Reasoning, ReasoningContent};
    use crate::providers::openai::responses_api::ReasoningSummary;

    #[test]
    fn assistant_redacted_reasoning_is_serialized_as_encrypted_content() {
        let reasoning = Reasoning {
            id: Some("rs_1".to_string()),
            content: vec![ReasoningContent::Redacted {
                data: "opaque-redacted".to_string(),
            }],
        };
        let message = RigMessage::Assistant {
            id: Some("assistant_1".to_string()),
            content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
        };

        let items = Vec::<Message>::try_from(message).expect("convert assistant message");
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items.first(),
            Some(Message::Reasoning {
                id,
                summary,
                encrypted_content: Some(encrypted_content),
            }) if id == "rs_1" && summary.is_empty() && encrypted_content == "opaque-redacted"
        ));
    }

    #[test]
    fn assistant_redacted_reasoning_does_not_leak_into_summary_text() {
        let reasoning = Reasoning {
            id: Some("rs_2".to_string()),
            content: vec![
                ReasoningContent::Text {
                    text: "explain".to_string(),
                    signature: None,
                },
                ReasoningContent::Redacted {
                    data: "opaque-redacted".to_string(),
                },
            ],
        };
        let message = RigMessage::Assistant {
            id: Some("assistant_2".to_string()),
            content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
        };

        let items = Vec::<Message>::try_from(message).expect("convert assistant message");
        let Some(Message::Reasoning {
            summary,
            encrypted_content,
            ..
        }) = items.first()
        else {
            panic!("Expected reasoning item");
        };

        assert_eq!(
            summary,
            &vec![ReasoningSummary::SummaryText {
                text: "explain".to_string()
            }]
        );
        assert_eq!(encrypted_content.as_deref(), Some("opaque-redacted"));
    }

    #[test]
    fn assistant_empty_reasoning_content_roundtrips_without_error() {
        let reasoning = Reasoning {
            id: Some("rs_empty".to_string()),
            content: vec![],
        };
        let message = RigMessage::Assistant {
            id: Some("assistant_2b".to_string()),
            content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
        };

        let items = Vec::<Message>::try_from(message).expect("convert assistant message");
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items.first(),
            Some(Message::Reasoning {
                id,
                summary,
                encrypted_content,
            }) if id == "rs_empty" && summary.is_empty() && encrypted_content.is_none()
        ));
    }

    #[test]
    fn assistant_reasoning_without_id_returns_request_error() {
        let message = RigMessage::Assistant {
            id: Some("assistant_no_reasoning_id".to_string()),
            content: OneOrMany::one(AssistantContent::Reasoning(Reasoning::new("thinking"))),
        };

        let converted = Vec::<Message>::try_from(message);
        assert!(matches!(
            converted,
            Err(CompletionError::RequestError(error))
                if error
                    .to_string()
                    .contains("Assistant reasoning `id` is required")
        ));
    }

    #[test]
    fn serialized_message_type_tags_are_snake_case() {
        let function_call = Message::function_call(
            "call_1".to_string(),
            "tool_name".to_string(),
            "{\"arg\":1}".to_string(),
        );
        let user_message = Message::user("hello");

        let function_call_json =
            serde_json::to_value(function_call).expect("serialize function_call");
        let user_message_json = serde_json::to_value(user_message).expect("serialize message");

        assert_eq!(
            function_call_json
                .get("type")
                .and_then(|value| value.as_str()),
            Some("function_call")
        );
        assert_eq!(
            user_message_json
                .get("type")
                .and_then(|value| value.as_str()),
            Some("message")
        );
    }

    #[test]
    fn user_tool_result_without_call_id_returns_request_error() {
        let message = RigMessage::tool_result("tool_1", "result payload");

        let converted = Vec::<Message>::try_from(message);
        assert!(matches!(
            converted,
            Err(CompletionError::RequestError(error))
                if error
                    .to_string()
                    .contains("Tool result `call_id` is required")
        ));
    }

    #[test]
    fn assistant_tool_call_without_call_id_returns_request_error() {
        let message = RigMessage::Assistant {
            id: Some("assistant_3".to_string()),
            content: OneOrMany::one(AssistantContent::tool_call(
                "tool_1",
                "my_tool",
                serde_json::json!({"arg":"value"}),
            )),
        };

        let converted = Vec::<Message>::try_from(message);
        assert!(matches!(
            converted,
            Err(CompletionError::RequestError(error))
                if error
                    .to_string()
                    .contains("Assistant tool call `call_id` is required")
        ));
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Error(ApiError),
}
