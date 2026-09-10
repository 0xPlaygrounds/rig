//! xAI Responses API types
//!
//! Types for the xAI Responses API: <https://docs.x.ai/docs/guides/chat>
//!
//! This module reuses OpenAI's Responses API types where compatible,
//! since xAI's API format is designed to be compatible with OpenAI.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::{self, CompletionError};
use crate::message::{Message as RigMessage, MimeType, ReasoningContent};
use crate::providers::openai::responses_api::ReasoningSummary;

#[derive(Debug, Serialize, Deserialize)]
struct CompletionRequest {
    model: String,
    input: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::responses_api::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<Value>,
}

fn normalize_strict_tool(mut tool: Value) -> Value {
    if tool.get("type").and_then(Value::as_str) == Some("function") {
        if let Some(parameters) = tool.get_mut("parameters") {
            crate::providers::openai::sanitize_schema(parameters);
        }
        if let Some(tool) = tool.as_object_mut() {
            tool.insert("strict".to_string(), Value::Bool(true));
        }
    }
    tool
}

pub(crate) fn create_completion_request(
    model: String,
    req: crate::completion::CompletionRequest,
    default_tools: &[crate::providers::openai::responses_api::ResponsesToolDefinition],
    strict_tools: bool,
    stream: bool,
) -> Result<(String, Value), CompletionError> {
    let chat_history = req.chat_history_with_documents();
    if req.output_schema.is_some() {
        tracing::warn!("Structured outputs currently not supported for xAI");
    }
    let model = req.model.clone().unwrap_or(model);
    let mut input = Vec::new();
    for message in chat_history {
        input.extend(Vec::<Message>::try_from(message)?);
    }
    let input = crate::message::require_non_empty(input, || {
        CompletionError::RequestError(
            "no message in the chat history converted to xAI input \
             (id-less reasoning-only content has no xAI representation)"
                .into(),
        )
    })?;

    let mut additional_params = req.additional_params.unwrap_or(Value::Null);
    let mut additional_tools = if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid xAI `additional_params.tools` payload: {err}").into(),
            )
        })?
    } else {
        Vec::new()
    };
    let mut tools = req
        .tools
        .into_iter()
        .map(ToolDefinition::from)
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?;
    tools.append(&mut additional_tools);
    tools.extend(
        default_tools
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?,
    );
    if strict_tools {
        tools = tools.into_iter().map(normalize_strict_tool).collect();
    }
    if stream {
        if additional_params.is_null() {
            additional_params = serde_json::json!({});
        }
        crate::json_utils::merge_inplace(
            &mut additional_params,
            serde_json::json!({"stream": true}),
        );
    }

    let request = CompletionRequest {
        model: model.clone(),
        input,
        temperature: req.temperature,
        max_output_tokens: req.max_tokens,
        tools,
        tool_choice: req
            .tool_choice
            .map(crate::providers::openai::responses_api::ToolChoice::try_from)
            .transpose()?,
        additional_params: (!additional_params.is_null()).then_some(additional_params),
    };
    Ok((model, serde_json::to_value(request)?))
}

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
                    let mime = img.media_type.map_or("image/png", |m| m.to_mime_type());
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
                        .map_or("application/pdf", |m| m.to_mime_type());
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

        fn reasoning_item(reasoning: crate::message::Reasoning) -> Option<Message> {
            let crate::message::Reasoning { id, content } = reasoning;
            // Only wire-genuine ids exist in durable histories (the streaming
            // layer populates `Reasoning::id` exclusively from
            // `StreamPartId::Wire`). An id-less reasoning item — a cross-provider
            // replay from a wire that issues no reasoning ids — drops from
            // request input, mirroring the OpenAI Responses handling, rather
            // than failing the whole request locally.
            let Some(id) = id else {
                tracing::warn!(
                    "xAI: dropping id-less reasoning item from request input \
                     (cross-provider replay; xAI reasoning requires a wire id)"
                );
                return None;
            };
            let mut encrypted_content = None;
            let mut summary = Vec::new();
            for reasoning_content in content {
                match reasoning_content {
                    ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) => {
                        summary.push(ReasoningSummary::SummaryText { text });
                    }
                    // xAI has a single encrypted_content field; only the first
                    // encrypted/redacted block can be preserved.
                    ReasoningContent::Redacted { data } | ReasoningContent::Encrypted(data) => {
                        if encrypted_content.is_some() {
                            tracing::warn!(
                                "xAI: dropping additional encrypted/redacted reasoning block \
                                 (API only supports one encrypted_content per item)"
                            );
                        }
                        encrypted_content.get_or_insert(data);
                    }
                }
            }

            Some(Message::reasoning(id, summary, encrypted_content))
        }

        match msg {
            RigMessage::System { content } => Ok(vec![Message::system(content)]),
            RigMessage::User { content } => {
                let mut items = Vec::new();
                let mut text_parts = Vec::new();
                let mut content_items = Vec::new();
                let mut has_images = false;

                for c in content {
                    match c {
                        UserContent::Text(Text { text, .. }) => text_parts.push(text),
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
                                text_parts.clear();
                            }
                            has_images = false;

                            // Provider-issued call id when one exists,
                            // else rig's minted handle — always present.
                            let call_id = tr.wire_call_id().to_owned();
                            // Tool result becomes FunctionCallOutput
                            let output = tr
                                .content
                                .into_iter()
                                .map(|tc| match tc {
                                    ToolResultContent::Text(t) => Ok(t.text),
                                    ToolResultContent::Json { value } => Ok(value.to_string()),
                                    ToolResultContent::Image(_) => {
                                        Err(CompletionError::RequestError(
                                            "xAI does not support images in tool results".into(),
                                        ))
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .join("\n");
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
                            let call_id = tc.wire_call_id().to_owned();
                            items.push(Message::function_call(
                                call_id,
                                tc.function.name,
                                tc.function.arguments.to_string(),
                            ));
                        }
                        AssistantContent::Reasoning(r) => {
                            flush_assistant_text(&mut items, &mut text_parts);
                            if let Some(item) = reasoning_item(r) {
                                items.push(item);
                            }
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

#[cfg(test)]
mod tests;
