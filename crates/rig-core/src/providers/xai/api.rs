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
    let mut input = req
        .preamble
        .as_ref()
        .map_or_else(Vec::new, |p| vec![Message::system(p)]);
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
        ) -> Result<Option<Message>, CompletionError> {
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
                return Ok(None);
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

            Ok(Some(Message::reasoning(id, summary, encrypted_content)))
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
                            if let Some(item) = reasoning_item(r)? {
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
mod tests {
    use super::{Content, Message, Role, create_completion_request};
    use crate::completion::{CompletionRequest, CompletionRequestBuilder, Document};
    use crate::message::{
        AssistantContent, Message as RigMessage, Reasoning, ReasoningContent, ToolChoice,
        ToolResultContent, UserContent,
    };
    use crate::providers::openai::responses_api::ReasoningSummary;
    use crate::test_utils::MockCompletionModel;

    fn request_value(request: CompletionRequest) -> serde_json::Value {
        create_completion_request("grok-4-0709".to_string(), request, &[], false, false)
            .expect("request conversion should succeed")
            .1
    }

    #[test]
    fn xai_request_includes_normalized_documents() {
        let request = CompletionRequestBuilder::new(
            MockCompletionModel::default(),
            "What does glarb-glarb mean?",
        )
        .document(Document {
            id: "doc_1".to_string(),
            text: "Definition of glarb-glarb: an ancient tool.".to_string(),
            additional_props: Default::default(),
        })
        .build();

        let serialized = request_value(request);
        let input = serialized["input"]
            .as_array()
            .expect("xAI request input should be an array");

        assert!(
            input
                .iter()
                .any(|message| message.to_string().contains("glarb-glarb")),
            "normalized documents should be forwarded into xAI input"
        );
    }

    #[test]
    fn xai_direct_request_keeps_documents_after_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![
                RigMessage::system("System prompt"),
                RigMessage::assistant("Earlier assistant turn"),
                RigMessage::system("Mid-conversation instruction"),
                RigMessage::user("What is glarb-glarb?"),
            ],
            documents: vec![Document {
                id: "doc_1".to_string(),
                text: "Definition of glarb-glarb: an ancient tool.".to_string(),
                additional_props: Default::default(),
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let serialized = request_value(request);
        let input = serialized["input"]
            .as_array()
            .expect("xAI request input should be an array");

        assert_eq!(input.len(), 5);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "user");
        assert!(input[1].to_string().contains("<file id: doc_1>"));
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[3]["role"], "system");
        assert_eq!(input[4]["role"], "user");
        assert_eq!(
            input
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc_1>"))
                .count(),
            1,
            "document input should appear exactly once: {input:?}"
        );
    }

    #[test]
    fn xai_request_uses_responses_tool_choice_for_specific_tool() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
            .tool(crate::completion::ToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool(crate::completion::ToolDefinition {
                name: "beta".to_string(),
                description: "Beta tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            })
            .build();

        let serialized = request_value(request);
        assert_eq!(
            serialized["tool_choice"],
            serde_json::json!({"type": "function", "name": "beta"})
        );
    }

    #[test]
    fn xai_stream_request_sets_stream_without_additional_params() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "hello").build();
        let (_, serialized) =
            create_completion_request("grok-4-0709".to_string(), request, &[], false, true)
                .expect("streaming request conversion should succeed");

        assert_eq!(serialized["stream"], true);
    }

    #[test]
    fn xai_strict_mode_normalizes_function_tools_from_every_source() {
        let mut request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "Use one of the tools.")
                .tool(crate::completion::ToolDefinition {
                    name: "request_tool".to_string(),
                    description: "A request tool".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"request": {"type": "string"}}
                    }),
                })
                .build();
        request.additional_params = Some(serde_json::json!({
            "tools": [
                {
                    "type": "function",
                    "name": "additional_tool",
                    "description": "An additional_params tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"additional": {"type": "string"}}
                    }
                },
                {"type": "web_search"}
            ]
        }));
        let default_tools = [
            crate::providers::openai::responses_api::ResponsesToolDefinition::function(
                "default_tool",
                "A model-level default tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {"default": {"type": "string"}}
                }),
            ),
        ];

        let (_, serialized) = create_completion_request(
            "grok-4-0709".to_string(),
            request,
            &default_tools,
            true,
            false,
        )
        .expect("request conversion should succeed");
        let tools = serialized["tools"]
            .as_array()
            .expect("tools should be an array");

        assert_eq!(tools.len(), 4);
        for tool in tools.iter().filter(|tool| tool["type"] == "function") {
            assert_eq!(tool["strict"], true);
            assert_eq!(tool["parameters"]["additionalProperties"], false);
            assert_eq!(
                tool["parameters"]["required"]
                    .as_array()
                    .expect("strict object schema should require every property")
                    .len(),
                1
            );
        }
        assert_eq!(tools[2], serde_json::json!({"type": "web_search"}));
    }

    #[test]
    fn mixed_user_content_preserves_order_without_duplicate_text() {
        let message = RigMessage::User {
            content: vec![
                UserContent::text("before"),
                UserContent::tool_result_with_call_id(
                    "result-id",
                    "call-id".to_string(),
                    "tool",
                    vec![ToolResultContent::json(serde_json::json!({ "ok": true }))],
                ),
                UserContent::text("after"),
            ],
        };

        let messages = Vec::<Message>::try_from(message).expect("mixed content should convert");
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            Message::Message {
                role: Role::User,
                content: Content::Text(text),
            } if text == "before"
        ));
        assert!(matches!(
            &messages[1],
            Message::FunctionCallOutput { call_id, output }
                if call_id == "call-id" && output == r#"{"ok":true}"#
        ));
        assert!(matches!(
            &messages[2],
            Message::Message {
                role: Role::User,
                content: Content::Text(text),
            } if text == "after"
        ));
    }

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
            content: vec![AssistantContent::Reasoning(reasoning)],
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
            content: vec![AssistantContent::Reasoning(reasoning)],
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
            content: vec![AssistantContent::Reasoning(reasoning)],
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
    fn assistant_reasoning_without_id_is_dropped_from_request_input() {
        // Only wire-genuine ids exist in durable histories; an id-less
        // reasoning item (a cross-provider replay from a wire that issues no
        // reasoning ids) drops from request input — mirroring the OpenAI
        // Responses handling — instead of failing the whole request or,
        // worse, fabricating an identifier xAI never issued (#2258 A1).
        let message = RigMessage::Assistant {
            id: Some("assistant_no_reasoning_id".to_string()),
            content: vec![AssistantContent::Reasoning(Reasoning::new("thinking"))],
        };

        let converted = Vec::<Message>::try_from(message).expect("conversion must not fail");
        assert!(
            converted
                .iter()
                .all(|item| !matches!(item, Message::Reasoning { .. })),
            "an id-less reasoning item must not reach the request: {converted:?}"
        );
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
    fn user_tool_result_without_call_id_replays_the_minted_handle() {
        // An empty wire id records no provider id and mints the correlation
        // handle; the minted handle (never an empty string) goes on the wire.
        let message = RigMessage::tool_result("", "tool_1", "result payload");

        let converted = Vec::<Message>::try_from(message).expect("id-less tool results convert");
        assert!(matches!(
            converted.as_slice(),
            [Message::FunctionCallOutput { call_id, output }]
                if !call_id.is_empty() && output == "result payload"
        ));
    }

    #[test]
    fn assistant_tool_call_without_call_id_replays_the_minted_handle() {
        // An empty wire id records no provider id and mints the correlation
        // handle; the minted handle (never an empty string) goes on the wire.
        let message = RigMessage::Assistant {
            id: Some("assistant_3".to_string()),
            content: vec![AssistantContent::tool_call(
                "",
                "my_tool",
                serde_json::json!({"arg":"value"}),
            )],
        };

        let converted = Vec::<Message>::try_from(message).expect("id-less tool calls convert");
        assert!(matches!(
            converted.as_slice(),
            [Message::FunctionCall { call_id, name, .. }]
                if !call_id.is_empty() && name == "my_tool"
        ));
    }
}
