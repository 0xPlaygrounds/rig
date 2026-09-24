//! Request and response types for the OpenAI Responses API.
//! Endpoint and dialect configuration lives in [`wire`].
//!
//! ```no_run
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.responses(openai::GPT_5_2);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use crate::error::EncodeError;
use crate::json_utils;
use crate::json_utils::string_or_vec;
use crate::message::{
    Document, DocumentMediaType, DocumentSourceKind, ImageDetail, MessageError, MimeType, Text,
};
use crate::{completion, message};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use std::convert::Infallible;
use std::ops::Add;
use std::str::FromStr;

pub mod streaming;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod websocket;
pub mod wire;

/// The completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionRequest {
    /// Message inputs
    pub input: Vec<InputItem>,
    /// The model name
    pub model: String,
    /// Top-level system instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The maximum number of output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Toggle to true for streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Sampling temperature. Supported values depend on the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Whether the LLM should be forced to use a tool before returning a response.
    /// If none provided, the default option is "auto".
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    /// The tools you want to use. This supports both function tools and hosted tools
    /// such as `web_search`, `file_search`, and `computer_use`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesToolDefinition>,
    /// Additional parameters
    #[serde(flatten)]
    pub additional_parameters: AdditionalParameters,
}

impl CompletionRequest {
    /// Appends a function or provider-hosted tool to the request.
    pub fn with_tool(mut self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Appends function or provider-hosted tools in iteration order.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }
}

/// An input item for [`CompletionRequest`].
#[derive(Debug, Deserialize, Clone)]
pub struct InputItem {
    /// The role of an input item/message.
    /// Input messages should be Some(Role::User), and output messages should be Some(Role::Assistant).
    /// Everything else should be None.
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<Role>,
    /// The input content itself.
    #[serde(flatten)]
    input: InputContent,
}

impl Serialize for InputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut value = serde_json::to_value(&self.input).map_err(serde::ser::Error::custom)?;
        let map = value.as_object_mut().ok_or_else(|| {
            serde::ser::Error::custom("Input content must serialize to an object")
        })?;

        if let Some(role) = &self.role
            && !map.contains_key("role")
        {
            map.insert(
                "role".to_string(),
                serde_json::to_value(role).map_err(serde::ser::Error::custom)?,
            );
        }

        value.serialize(serializer)
    }
}

impl InputItem {
    pub fn system_message(content: impl Into<String>) -> Self {
        Self {
            role: Some(Role::System),
            input: InputContent::Message(Message::System {
                content: vec![SystemContent::InputText {
                    text: content.into(),
                }],
                name: None,
            }),
        }
    }

    /// A user-role input item carrying one content part.
    fn user_content(content: UserContent) -> Self {
        Self {
            role: Some(Role::User),
            input: InputContent::Message(Message::User {
                content: vec![content],
                name: None,
            }),
        }
    }

    pub(crate) fn system_text(&self) -> Option<String> {
        match &self.input {
            InputContent::Message(Message::System { content, .. }) => Some(
                content
                    .iter()
                    .map(|item| match item {
                        SystemContent::InputText { text } => text.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        }
    }
}

/// Message roles. Used by OpenAI Responses API to determine who created a given message.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Content carried by an [`InputItem`].
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    Message(Message),
    Reasoning(OpenAIReasoning),
    FunctionCall(OutputFunctionCall),
    FunctionCallOutput(ToolResult),
    /// Opaque compaction data for replaying a compacted context. All fields
    /// other than the separately serialized `type` tag are preserved.
    Compaction(Map<String, Value>),
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct OpenAIReasoning {
    id: String,
    pub summary: Vec<ReasoningSummary>,
    #[serde(
        default,
        deserialize_with = "deserialize_reasoning_text_content",
        serialize_with = "serialize_reasoning_text_content",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub content: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    /// The upstream's signature over the reasoning text, which a gateway
    /// (OpenRouter relaying Claude) returns beside it and needs back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolStatus>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    SummaryText { text: String },
}

impl ReasoningSummary {
    fn new(input: &str) -> Self {
        Self::SummaryText {
            text: input.to_string(),
        }
    }

    pub fn text(&self) -> &str {
        let ReasoningSummary::SummaryText { text } = self;
        text
    }
}

fn reasoning_text_content_json(content: &[String]) -> Value {
    Value::Array(
        content
            .iter()
            .map(|text| {
                serde_json::json!({
                    "type": "reasoning_text",
                    "text": text,
                })
            })
            .collect(),
    )
}

fn serialize_reasoning_text_content<S>(content: &[String], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    reasoning_text_content_json(content).serialize(serializer)
}

fn deserialize_reasoning_text_content<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::Object(mut item) => item
                    .remove("text")
                    .and_then(|text| text.as_str().map(ToOwned::to_owned)),
                Value::String(text) => Some(text),
                _ => None,
            })
            .collect(),
        Value::String(text) => vec![text],
        _ => Vec::new(),
    })
}

/// A tool result.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolResult {
    /// The call ID of a tool (this should be linked to the call ID for a tool call, otherwise an error will be received)
    call_id: String,
    /// The result of a tool call.
    output: ToolResultOutput,
    /// The status of a tool call (if used in a completion request, this should always be Completed)
    status: ToolStatus,
}

/// Responses API function-call output, which accepts either plain text or an
/// ordered list of rich input blocks.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolResultOutput {
    /// A plain textual function result.
    Text(String),
    /// Ordered rich input blocks for a multimodal function result.
    Content(Vec<ToolResultOutputContent>),
}

/// Rich content supported by a Responses API function-call output.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultOutputContent {
    /// Textual function-output content.
    InputText {
        /// The text presented to the model.
        text: String,
    },
    /// Image function-output content.
    InputImage {
        /// A public URL or base64 data URL, mutually exclusive with `file_id`.
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        /// An uploaded OpenAI file identifier, mutually exclusive with
        /// `image_url`.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// Provider image-detail preference.
        #[serde(default)]
        detail: ImageDetail,
    },
}

/// Returns a request error for an unsupported source.
/// Callers must base64-encode raw bytes before request conversion.
fn unsupported_document_source(source: DocumentSourceKind) -> EncodeError {
    match source {
        DocumentSourceKind::Raw(_) => {
            EncodeError::request("Raw file data not supported, encode as base64 first")
        }
        source => EncodeError::request(format!("Unsupported document type: {source}")),
    }
}

fn responses_tool_result_output(
    content: Vec<message::ToolResultContent>,
) -> Result<ToolResultOutput, MessageError> {
    let mut rich_output = Vec::new();

    for content in content {
        match content {
            message::ToolResultContent::Text(Text { text, .. }) => {
                rich_output.push(ToolResultOutputContent::InputText { text });
            }
            message::ToolResultContent::Json { value } => {
                rich_output.push(ToolResultOutputContent::InputText {
                    text: value.to_string(),
                });
            }
            message::ToolResultContent::Image(message::Image {
                data,
                media_type,
                detail,
                ..
            }) => {
                let (image_url, file_id) = match data {
                    DocumentSourceKind::Base64(data) => {
                        let media_type = media_type.ok_or_else(|| {
                            MessageError::ConversionError(
                                "A media type is required for base64 tool-result images".into(),
                            )
                        })?;
                        (
                            Some(format!(
                                "data:{media_type};base64,{data}",
                                media_type = media_type.to_mime_type()
                            )),
                            None,
                        )
                    }
                    DocumentSourceKind::Url(url) => (Some(url), None),
                    DocumentSourceKind::FileId(file_id) => (None, Some(file_id)),
                    unsupported => {
                        return Err(MessageError::ConversionError(format!(
                            "Unsupported tool-result image source: {unsupported}"
                        )));
                    }
                };
                rich_output.push(ToolResultOutputContent::InputImage {
                    image_url,
                    file_id,
                    detail: detail.unwrap_or_default(),
                });
            }
        }
    }

    match rich_output.as_slice() {
        [ToolResultOutputContent::InputText { text }] => Ok(ToolResultOutput::Text(text.clone())),

        _ => Ok(ToolResultOutput::Content(rich_output)),
    }
}

impl TryFrom<crate::completion::Message> for Vec<InputItem> {
    type Error = EncodeError;

    fn try_from(value: crate::completion::Message) -> Result<Self, Self::Error> {
        match value {
            crate::completion::Message::System { content } => Ok(vec![InputItem {
                role: Some(Role::System),
                input: InputContent::Message(Message::System {
                    content: vec![content.into()],
                    name: None,
                }),
            }]),
            crate::completion::Message::User { content } => {
                let mut items = Vec::new();

                for user_content in content {
                    match user_content {
                        crate::message::UserContent::Text(Text { text, .. }) => {
                            items.push(InputItem::user_content(UserContent::InputText { text }));
                        }
                        crate::message::UserContent::ToolResult(tool_result) => {
                            // Prefer provider identity so results match replayed calls.
                            let call_id = tool_result.wire_call_id().into_owned();
                            let output = responses_tool_result_output(tool_result.content)?;
                            items.push(InputItem {
                                role: None,
                                input: InputContent::FunctionCallOutput(ToolResult {
                                    call_id,
                                    output,
                                    status: ToolStatus::Completed,
                                }),
                            });
                        }
                        crate::message::UserContent::Document(Document {
                            data: DocumentSourceKind::FileId(file_id),
                            ..
                        }) => items.push(InputItem::user_content(UserContent::InputFile {
                            file_id: Some(file_id),
                            file_data: None,
                            file_url: None,
                            filename: None,
                        })),
                        crate::message::UserContent::Document(Document {
                            data,
                            media_type: Some(DocumentMediaType::PDF),
                            ..
                        }) => {
                            let (file_data, file_url, filename) = match data {
                                DocumentSourceKind::Base64(data) => (
                                    Some(format!("data:application/pdf;base64,{data}")),
                                    None,
                                    Some("document.pdf".to_string()),
                                ),
                                DocumentSourceKind::Url(url) => (None, Some(url), None),
                                source => return Err(unsupported_document_source(source)),
                            };

                            items.push(InputItem::user_content(UserContent::InputFile {
                                file_id: None,
                                file_data,
                                file_url,
                                filename,
                            }));
                        }
                        crate::message::UserContent::Document(Document {
                            data:
                                DocumentSourceKind::Base64(text) | DocumentSourceKind::String(text),
                            ..
                        }) => items.push(InputItem::user_content(UserContent::InputText { text })),
                        crate::message::UserContent::Image(crate::message::Image {
                            data,
                            media_type,
                            detail,
                            ..
                        }) => {
                            let url = match data {
                                DocumentSourceKind::Base64(data) => {
                                    let media_type = media_type
                                        .map(|media_type| media_type.to_mime_type().to_string())
                                        .unwrap_or_default();
                                    format!("data:{media_type};base64,{data}")
                                }
                                DocumentSourceKind::Url(url) => url,
                                source => return Err(unsupported_document_source(source)),
                            };
                            items.push(InputItem::user_content(UserContent::InputImage {
                                image_url: url,
                                detail: detail.unwrap_or_default(),
                            }));
                        }
                        message => {
                            return Err(EncodeError::request(format!(
                                "Unsupported message: {message:?}"
                            )));
                        }
                    }
                }

                Ok(items)
            }
            crate::completion::Message::Assistant { id, content } => {
                let mut reasoning_items = Vec::new();
                let mut other_items: Vec<InputItem> = Vec::new();
                // Merge text parts under one message ID because duplicate input
                // item IDs are rejected.
                let mut message_item: Option<usize> = None;

                for assistant_content in content {
                    match assistant_content {
                        crate::message::AssistantContent::Text(Text {
                            text,
                            additional_params,
                        }) => {
                            let Some(message) =
                                assistant_text_replay_message(id.clone(), text, additional_params)
                            else {
                                continue;
                            };
                            match (message, message_item) {
                                (Message::Assistant { content: more, .. }, Some(at)) => {
                                    if let Some(InputItem {
                                        input:
                                            InputContent::Message(Message::Assistant {
                                                content, ..
                                            }),
                                        ..
                                    }) = other_items.get_mut(at)
                                    {
                                        content.extend(more);
                                    }
                                }
                                (message, _) => {
                                    let with_id = matches!(message, Message::Assistant { .. });
                                    other_items.push(InputItem {
                                        role: Some(Role::Assistant),
                                        input: InputContent::Message(message),
                                    });
                                    if with_id {
                                        message_item = Some(other_items.len() - 1);
                                    }
                                }
                            }
                        }
                        crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
                            id,
                            provider,
                            function,
                            ..
                        }) => {
                            let (call_id, item_id) = match provider {
                                Some(provider) => {
                                    let item_id = provider.item_id.clone().unwrap_or_default();
                                    (provider.call_id, item_id)
                                }
                                None => (id.wire_hint().into_owned(), String::new()),
                            };
                            other_items.push(InputItem {
                                role: None,
                                input: InputContent::FunctionCall(OutputFunctionCall {
                                    arguments: function.arguments.into(),
                                    call_id,
                                    id: item_id,
                                    name: function.name,
                                    status: ToolStatus::Completed,
                                }),
                            });
                        }
                        crate::message::AssistantContent::Reasoning(reasoning) => {
                            if let Some(openai_reasoning) = openai_reasoning_from_core(&reasoning) {
                                reasoning_items.push(InputItem {
                                    role: None,
                                    input: InputContent::Reasoning(openai_reasoning),
                                });
                            }
                        }
                        crate::message::AssistantContent::Image(_) => {
                            return Err(EncodeError::request(
                                "Assistant image content is not supported in OpenAI Responses API"
                                    .to_string(),
                            ));
                        }
                    }
                }

                let mut items = reasoning_items;
                items.extend(other_items);
                Ok(items)
            }
        }
    }
}

/// Builds reasoning blocks in summary, text, encrypted-content order.
/// Empty encrypted content contributes no block. A signature signs the
/// reasoning text, so it rides on the last text block, or on an empty one
/// when the item carried no text.
pub(crate) fn reasoning_content_blocks(
    summary: Vec<ReasoningSummary>,
    content: Vec<String>,
    encrypted_content: Option<String>,
    signature: Option<String>,
) -> Vec<message::ReasoningContent> {
    let mut blocks = summary
        .into_iter()
        .map(|summary| match summary {
            ReasoningSummary::SummaryText { text } => message::ReasoningContent::Summary(text),
        })
        .collect::<Vec<_>>();

    blocks.extend(
        content
            .into_iter()
            .map(|text| message::ReasoningContent::Text {
                text,
                signature: None,
            }),
    );
    if let Some(signature) = signature {
        match blocks.iter_mut().rev().find_map(|block| match block {
            message::ReasoningContent::Text { signature, .. } => Some(signature),
            _ => None,
        }) {
            Some(slot) => *slot = Some(signature),
            None => blocks.push(message::ReasoningContent::Text {
                text: String::new(),
                signature: Some(signature),
            }),
        }
    }

    if let Some(encrypted_content) = encrypted_content.filter(|content| !content.is_empty()) {
        blocks.push(message::ReasoningContent::Encrypted(encrypted_content));
    }

    blocks
}

fn openai_reasoning_from_core(reasoning: &crate::message::Reasoning) -> Option<OpenAIReasoning> {
    // Reasoning without a provider item ID cannot be replayed.
    let id = reasoning.id.clone()?;

    let mut summary = Vec::new();
    let mut reasoning_content = Vec::new();
    let mut encrypted_content = None;
    let mut text_signature = None;
    for content in &reasoning.content {
        match content {
            crate::message::ReasoningContent::Text { text, signature } => {
                // An empty block that only carries the signature adds no text.
                if !(text.is_empty() && signature.is_some()) {
                    reasoning_content.push(text.clone());
                }
                // A signature covers the reasoning before it, so the last
                // one is the item's, matching where decoding puts it.
                if let Some(signature) = signature {
                    text_signature = Some(signature.clone());
                }
            }
            crate::message::ReasoningContent::Summary(text) => {
                summary.push(ReasoningSummary::new(text));
            }
            // OpenAI reasoning input has one opaque payload field; preserve either
            // encrypted or redacted blocks there, preferring the first one seen.
            crate::message::ReasoningContent::Encrypted(data)
            | crate::message::ReasoningContent::Redacted { data } => {
                encrypted_content.get_or_insert_with(|| data.clone());
            }
        }
    }

    Some(OpenAIReasoning {
        id,
        summary,
        content: reasoning_content,
        encrypted_content,
        signature: text_signature,
        status: None,
    })
}

/// A function or hosted tool available to a Responses request.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ResponsesToolDefinition {
    /// The type of tool.
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Parameters - this should be a JSON schema. Strict function tools must use OpenAI's supported strict schema subset.
    #[serde(default, skip_serializing_if = "is_json_null")]
    pub parameters: serde_json::Value,
    /// Whether to use strict mode. Disabled by default; opt in with [`Self::with_strict`]
    /// or [`wire::Responses::with_strict_tools`].
    ///
    /// Always serialized: the Responses API treats an omitted `strict` as "attempt strict
    /// mode", so `false` must reach the wire for non-strict tools to actually be non-strict.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub strict: bool,
    /// Tool description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Additional provider-specific configuration for hosted tools.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub config: Map<String, Value>,
}

fn is_json_null(value: &Value) -> bool {
    value.is_null()
}

impl ResponsesToolDefinition {
    /// Creates a function tool definition with strict mode disabled.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            name: name.into(),
            parameters,
            strict: false,
            description: description.into(),
            config: Map::new(),
        }
    }

    /// Creates a strict function tool definition.
    ///
    /// The schema is sanitized to OpenAI's strict subset (`additionalProperties: false`
    /// added and every property forced into `required`).
    pub fn strict_function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self::function(name, description, parameters).with_strict()
    }

    /// Enables strict mode for this function tool.
    ///
    /// Function schemas are sanitized to OpenAI's strict subset. Hosted tools are
    /// returned unchanged because strict mode only applies to function tools.
    pub fn with_strict(mut self) -> Self {
        if self.kind == "function" {
            super::sanitize_schema(&mut self.parameters);
            self.strict = true;
        }
        self
    }

    /// Creates a hosted tool definition for an arbitrary hosted tool type.
    pub fn hosted(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            name: String::new(),
            parameters: Value::Null,
            strict: false,
            description: String::new(),
            config: Map::new(),
        }
    }

    /// Creates a hosted `web_search` tool definition.
    pub fn web_search() -> Self {
        Self::hosted("web_search")
    }

    /// Creates a hosted `file_search` tool definition.
    pub fn file_search() -> Self {
        Self::hosted("file_search")
    }

    /// Creates a hosted `computer_use` tool definition.
    pub fn computer_use() -> Self {
        Self::hosted("computer_use")
    }

    /// Adds hosted-tool configuration fields.
    pub fn with_config(mut self, key: impl Into<String>, value: Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    fn normalize(self) -> Self {
        self.with_strict()
    }
}

impl From<completion::ToolDefinition> for ResponsesToolDefinition {
    fn from(value: completion::ToolDefinition) -> Self {
        let completion::ToolDefinition {
            name,
            parameters,
            description,
        } = value;

        Self::function(name, description, parameters)
    }
}

/// Automatic, disabled, required, named-function, or restricted tool selection.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"`, `"none"`, or `"required"`. Do not use the wrapped enum's
    /// `Function` variant; use [`ToolChoiceDefinition::Function`] instead.
    Mode(super::completion::ToolChoice),
    /// A typed tool-choice object (`function` or `allowed_tools`).
    Definition(ToolChoiceDefinition),
}

/// A typed Responses API tool-choice object.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceDefinition {
    /// Force the model to call the named function tool.
    Function {
        /// Name of the function tool the model must call.
        name: String,
    },
    /// Restrict the model to a subset of the request's tools.
    AllowedTools {
        /// Whether the model may still answer without a tool call (`auto`)
        /// or must call one of the allowed tools (`required`).
        mode: AllowedToolsMode,
        /// The tools the model is allowed to call.
        tools: Vec<AllowedTool>,
    },
}

/// Constrains how the model may use the tools listed in
/// [`ToolChoiceDefinition::AllowedTools`].
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsMode {
    /// The model may call one of the allowed tools or answer directly.
    Auto,
    /// The model must call one of the allowed tools.
    Required,
}

/// One entry of a [`ToolChoiceDefinition::AllowedTools`] tool list.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AllowedTool {
    /// A function tool referenced by name.
    Function {
        /// Name of the allowed function tool.
        name: String,
    },
}

impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = EncodeError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let choice = match value {
            message::ToolChoice::Auto => Self::Mode(super::completion::ToolChoice::Auto),
            message::ToolChoice::None => Self::Mode(super::completion::ToolChoice::None),
            message::ToolChoice::Required => Self::Mode(super::completion::ToolChoice::Required),
            message::ToolChoice::Specific { function_names } => {
                let mut names = function_names.into_iter();
                let Some(first) = names.next() else {
                    return Err(EncodeError::request(
                        "ToolChoice::Specific requires at least one function name",
                    ));
                };

                match names.next() {
                    None => Self::Definition(ToolChoiceDefinition::Function { name: first }),
                    Some(second) => {
                        let tools = std::iter::once(first)
                            .chain(std::iter::once(second))
                            .chain(names)
                            .map(|name| AllowedTool::Function { name })
                            .collect();
                        Self::Definition(ToolChoiceDefinition::AllowedTools {
                            mode: AllowedToolsMode::Required,
                            tools,
                        })
                    }
                }
            }
        };

        Ok(choice)
    }
}

/// Response token counts and optional cached-input and reasoning breakdowns.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Input tokens
    pub input_tokens: u64,
    /// In-depth detail on input tokens (cached tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    /// Output tokens
    pub output_tokens: u64,
    /// In-depth detail on output tokens (reasoning tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Total tokens used (for a given prompt)
    pub total_tokens: u64,
}

impl From<&ResponsesUsage> for crate::completion::Usage {
    fn from(usage: &ResponsesUsage) -> Self {
        crate::completion::Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.total_tokens),
            cached_input_tokens: usage
                .input_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            reasoning_tokens: usage
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            ..Default::default()
        }
    }
}

impl From<ResponsesUsage> for crate::completion::Usage {
    fn from(usage: ResponsesUsage) -> Self {
        Self::from(&usage)
    }
}

/// Adds present breakdowns, preserving a lone value or joint absence.
fn add_optional_details<T: Add<Output = T>>(lhs: Option<T>, rhs: Option<T>) -> Option<T> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
        (lhs, rhs) => lhs.or(rhs),
    }
}

impl Add for ResponsesUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            input_tokens_details: add_optional_details(
                self.input_tokens_details,
                rhs.input_tokens_details,
            ),
            output_tokens: self.output_tokens + rhs.output_tokens,
            output_tokens_details: add_optional_details(
                self.output_tokens_details,
                rhs.output_tokens_details,
            ),
            total_tokens: self.total_tokens + rhs.total_tokens,
        }
    }
}

/// In-depth details on input tokens.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Cached tokens from OpenAI
    pub cached_tokens: u64,
}

impl Add for InputTokensDetails {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cached_tokens: self.cached_tokens + rhs.cached_tokens,
        }
    }
}

/// In-depth details on output tokens.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens
    pub reasoning_tokens: u64,
}

impl Add for OutputTokensDetails {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
        }
    }
}

/// Provider-reported reason for an incomplete response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IncompleteDetailsReason {
    /// The reason for an incomplete [`CompletionResponse`].
    pub reason: String,
}

/// A response error from OpenAI's Response API.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
}

/// A response object as an enum (ensures type validation)
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseObject {
    Response,
}

/// The response status as an enum (ensures type validation)
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Incomplete,
    /// A provider-specific status added after this client was released.
    Other(String),
}

/// The wire spelling of a [`ResponseStatus`].
///
/// Statuses outside the normalized finish-reason vocabulary are carried through
/// as [`completion::FinishReason::Other`], so they must keep OpenAI's own
/// spelling rather than a Rust `Debug` name.
fn response_status_wire_name(status: &ResponseStatus) -> &str {
    match status {
        ResponseStatus::InProgress => "in_progress",
        ResponseStatus::Completed => "completed",
        ResponseStatus::Failed => "failed",
        ResponseStatus::Cancelled => "cancelled",
        ResponseStatus::Queued => "queued",
        ResponseStatus::Incomplete => "incomplete",
        ResponseStatus::Other(status) => status,
    }
}

impl Serialize for ResponseStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(response_status_wire_name(self))
    }
}

impl<'de> Deserialize<'de> for ResponseStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            "queued" => Self::Queued,
            "incomplete" => Self::Incomplete,
            other => Self::Other(other.to_owned()),
        })
    }
}

/// Maps status and incomplete details to a finish reason, preserving unknown
/// terminal values. In-flight statuses and empty unknown statuses return `None`.
/// Completed responses map to `Stop`; response assembly upgrades tool-call turns.
pub(crate) fn map_finish_reason(
    status: &ResponseStatus,
    incomplete_details: Option<&IncompleteDetailsReason>,
) -> Option<completion::FinishReason> {
    match status {
        ResponseStatus::Completed => Some(completion::FinishReason::Stop),
        ResponseStatus::Incomplete => Some(
            match incomplete_details
                .map(|details| details.reason.as_str())
                .filter(|reason| !reason.is_empty())
            {
                Some("max_output_tokens") => completion::FinishReason::Length,
                Some("content_filter") => completion::FinishReason::ContentFilter,
                Some(other) => completion::FinishReason::Other(other.to_owned()),
                // Incomplete without a stated reason: the status itself is all
                // the provider told us.
                None => {
                    completion::FinishReason::Other(response_status_wire_name(status).to_owned())
                }
            },
        ),
        ResponseStatus::Other(status) if status.is_empty() => None,
        ResponseStatus::Failed | ResponseStatus::Cancelled | ResponseStatus::Other(_) => Some(
            completion::FinishReason::Other(response_status_wire_name(status).to_owned()),
        ),
        // The turn has not terminated, so there is genuinely no reason yet.
        ResponseStatus::InProgress | ResponseStatus::Queued => None,
    }
}

/// Controls where Rig system instructions are placed in an OpenAI Responses request.
///
/// Serialized because it is a field of the [`wire::Responses`] wire, which is
/// data a host may store.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemInstructionsPlacement {
    /// Send the leading run of system instructions (the preamble and any system
    /// messages that open the conversation) through the official top-level
    /// `instructions` field. Mid-conversation system messages keep their
    /// position in `input`.
    #[default]
    Instructions,
    /// Send every system message through the top-level `instructions` field,
    /// including mid-conversation ones.
    ///
    /// Use this for backends that reject the `system` role in `input` entirely.
    AllInstructions,
    /// Send system instructions as `system` messages in `input`.
    ///
    /// Use this only for OpenAI-compatible providers that do not support top-level
    /// `instructions`.
    InputSystemMessages,
}

/// Converts a Rig request using the default system-instruction placement.
impl TryFrom<(String, crate::completion::CompletionRequest)> for CompletionRequest {
    type Error = EncodeError;
    fn try_from(
        (model, request): (String, crate::completion::CompletionRequest),
    ) -> Result<Self, Self::Error> {
        Self::try_from(ResponsesRequestParams {
            model,
            request,
            system_instructions_placement: SystemInstructionsPlacement::default(),
        })
    }
}

/// Parameters for converting a [`crate::completion::CompletionRequest`] into a
/// Responses API [`CompletionRequest`] with a non-default configuration.
pub struct ResponsesRequestParams {
    pub model: String,
    pub request: crate::completion::CompletionRequest,
    pub system_instructions_placement: SystemInstructionsPlacement,
}

impl TryFrom<ResponsesRequestParams> for CompletionRequest {
    type Error = EncodeError;

    fn try_from(params: ResponsesRequestParams) -> Result<Self, Self::Error> {
        let ResponsesRequestParams {
            model,
            request: mut req,
            system_instructions_placement,
        } = params;
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or(model);
        let mut instruction_parts = Vec::new();
        let mut input = {
            let mut full_history: Vec<InputItem> = Vec::new();
            let tool_ids =
                crate::providers::internal::tool_call_ids::ToolCallIds::new(&chat_history)
                    .map_err(EncodeError::request)?;
            for (position, history_item) in chat_history.into_iter().enumerate() {
                let mut items = <Vec<InputItem>>::try_from(history_item)?;
                tool_ids
                    .apply(
                        position,
                        items.iter_mut().filter_map(|item| match &mut item.input {
                            InputContent::FunctionCall(call) => Some(&mut call.call_id),
                            InputContent::FunctionCallOutput(result) => Some(&mut result.call_id),
                            _ => None,
                        }),
                    )
                    .map_err(EncodeError::request)?;
                full_history.extend(items);
            }
            full_history
        };

        let mut lift_system_text = |text: String| {
            let text = text.trim();
            if !text.is_empty() {
                instruction_parts.push(text.to_string());
            }
        };
        let items_before_lift = input.len();
        match system_instructions_placement {
            SystemInstructionsPlacement::Instructions => {
                // Lift only the leading run of system items (the preamble and any
                // system messages that open the conversation) into the top-level
                // `instructions` field. Mid-conversation system messages keep
                // their position in `input`, and a request made up solely of
                // system messages keeps them in `input` so it stays non-empty.
                let leading_system_texts: Vec<String> =
                    input.iter().map_while(InputItem::system_text).collect();
                if leading_system_texts.len() < input.len() {
                    input.drain(..leading_system_texts.len());
                    leading_system_texts
                        .into_iter()
                        .for_each(&mut lift_system_text);
                }
            }
            SystemInstructionsPlacement::AllInstructions => {
                // Lift every system item, wherever it appears, for backends
                // that reject the `system` role in `input` entirely.
                let mut remaining = Vec::with_capacity(input.len());
                for item in input {
                    match item.system_text() {
                        Some(text) => lift_system_text(text),
                        None => remaining.push(item),
                    }
                }
                input = remaining;
            }
            SystemInstructionsPlacement::InputSystemMessages => {}
        }
        let instructions = (!instruction_parts.is_empty()).then(|| instruction_parts.join("\n\n"));
        let lifted_system_items = input.len() < items_before_lift;

        let input = crate::message::require_non_empty(input, || {
            EncodeError::request(if lifted_system_items {
                "OpenAI Responses request input must contain at least one non-system item \
                 (system messages were lifted into the top-level `instructions` field)"
            } else {
                "OpenAI Responses request input must contain at least one item"
            })
        })?;

        let mut additional_params_payload = req.additional_params.take().unwrap_or(Value::Null);
        let stream = match &additional_params_payload {
            Value::Bool(stream) => Some(*stream),
            Value::Object(map) => map.get("stream").and_then(Value::as_bool),
            _ => None,
        };

        let mut additional_tools = Vec::new();
        if let Some(additional_params_map) = additional_params_payload.as_object_mut() {
            if let Some(raw_tools) = additional_params_map.remove("tools") {
                additional_tools = serde_json::from_value::<Vec<ResponsesToolDefinition>>(
                    raw_tools,
                )
                .map_err(|err| {
                    EncodeError::request(format!(
                        "Invalid OpenAI Responses tools payload in additional_params: {err}"
                    ))
                })?;
            }
            additional_params_map.remove("stream");
        }

        if additional_params_payload.is_boolean() {
            additional_params_payload = Value::Null;
        }

        let mut additional_parameters = if additional_params_payload.is_null() {
            // If there's no additional parameters, initialise an empty object
            AdditionalParameters::default()
        } else {
            serde_json::from_value::<AdditionalParameters>(additional_params_payload).map_err(
                |err| {
                    EncodeError::request(format!(
                        "Invalid OpenAI Responses additional_params payload: {err}"
                    ))
                },
            )?
        };
        if additional_parameters.reasoning.is_some() {
            let include = additional_parameters.include.get_or_insert_with(Vec::new);
            if !include
                .iter()
                .any(|item| matches!(item, Include::ReasoningEncryptedContent))
            {
                include.push(Include::ReasoningEncryptedContent);
            }
        }

        // Apply output_schema as structured output if not already configured via additional_params
        if additional_parameters.text.is_none()
            && let Some(schema) = req.output_schema
        {
            let (name, schema_value) = super::structured_output_schema(schema);
            additional_parameters.text = Some(TextConfig::structured_output(name, schema_value));
        }

        let tool_choice = req.tool_choice.map(ToolChoice::try_from).transpose()?;
        let mut tools: Vec<ResponsesToolDefinition> = req
            .tools
            .into_iter()
            .map(ResponsesToolDefinition::from)
            .collect();
        tools.append(&mut additional_tools);

        Ok(Self {
            input,
            model,
            instructions,
            max_output_tokens: req.max_tokens,
            stream,
            tool_choice,
            tools,
            temperature: req.temperature,
            additional_parameters,
        })
    }
}

/// The standard response format from OpenAI's Responses API.
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    /// The ID of a completion response.
    pub id: String,
    /// The type of the object.
    pub object: ResponseObject,
    /// The time at which a given response has been created, in seconds from the UNIX epoch (01/01/1970 00:00:00).
    pub created_at: u64,
    /// The status of the response.
    pub status: ResponseStatus,
    /// Response error (optional)
    pub error: Option<ResponseError>,
    /// Incomplete response details (optional)
    pub incomplete_details: Option<IncompleteDetailsReason>,
    /// System prompt/preamble
    pub instructions: Option<String>,
    /// The maximum number of tokens the model should output
    pub max_output_tokens: Option<u64>,
    /// The model name
    pub model: String,
    /// Provider-specific top-level reasoning content returned by some
    /// OpenAI-compatible Responses implementations.
    pub provider_reasoning: Option<String>,
    /// Transport request ID from the `x-request-id` header, stamped by the driver.
    /// Body deserialization leaves it `None`; serialization omits it.
    pub provider_request_id: Option<String>,
    /// The complete object-shaped top-level reasoning metadata returned by the provider.
    ///
    /// Unknown fields, unknown values, and null-valued members inside the object
    /// are preserved value-equivalently. A top-level null, missing field, or
    /// unsupported non-object shape is normalized to no reasoning metadata.
    /// When serializing manually constructed responses, [`Self::provider_reasoning`]
    /// takes precedence over this field, and this field takes precedence over
    /// [`Self::reasoning_context`].
    pub reasoning_metadata: Option<Map<String, Value>>,
    /// The effective reasoning context returned by OpenAI.
    ///
    /// This is populated as a convenience projection of
    /// [`Self::reasoning_metadata`]. String-shaped reasoning returned by compatible
    /// providers remains available through [`Self::provider_reasoning`].
    pub reasoning_context: Option<String>,
    /// Token usage
    pub usage: Option<ResponsesUsage>,
    /// The model output (messages, etc will go here)
    pub output: Vec<Output>,
    /// Tools
    pub tools: Vec<ResponsesToolDefinition>,
    /// Additional parameters
    pub additional_parameters: AdditionalParameters,
}

#[derive(Serialize)]
#[serde(untagged)]
enum CompletionResponseReasoningRef<'a> {
    Text(&'a str),
    Metadata(&'a Map<String, Value>),
    Context { context: &'a str },
}

#[derive(Serialize)]
struct CompletionResponseWireRef<'a> {
    id: &'a str,
    object: &'a ResponseObject,
    created_at: u64,
    status: &'a ResponseStatus,
    error: &'a Option<ResponseError>,
    incomplete_details: &'a Option<IncompleteDetailsReason>,
    instructions: &'a Option<String>,
    max_output_tokens: &'a Option<u64>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<CompletionResponseReasoningRef<'a>>,
    usage: &'a Option<ResponsesUsage>,
    output: &'a Vec<Output>,
    tools: &'a Vec<ResponsesToolDefinition>,
    #[serde(flatten)]
    additional_parameters: &'a AdditionalParameters,
}

/// Response body with untyped echoed metadata. Metadata is decoded separately
/// so an incompatible optional field does not reject the response.
#[derive(Deserialize)]
struct CompletionResponseWire {
    id: String,
    object: ResponseObject,
    created_at: u64,
    status: ResponseStatus,
    error: Option<ResponseError>,
    incomplete_details: Option<IncompleteDetailsReason>,
    instructions: Option<String>,
    max_output_tokens: Option<u64>,
    model: String,
    #[serde(default)]
    reasoning: Option<Value>,
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    output: Vec<Output>,
    #[serde(default)]
    tools: Vec<ResponsesToolDefinition>,
    #[serde(flatten)]
    metadata: Map<String, Value>,
}

impl Serialize for CompletionResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Omit request reasoning configuration to avoid duplicate response keys.
        let mut additional_parameters = self.additional_parameters.clone();
        additional_parameters.reasoning = None;

        let reasoning = self
            .provider_reasoning
            .as_deref()
            .map(CompletionResponseReasoningRef::Text)
            .or_else(|| {
                self.reasoning_metadata
                    .as_ref()
                    .map(CompletionResponseReasoningRef::Metadata)
            })
            .or_else(|| {
                self.reasoning_context
                    .as_deref()
                    .map(|context| CompletionResponseReasoningRef::Context { context })
            });

        CompletionResponseWireRef {
            id: &self.id,
            object: &self.object,
            created_at: self.created_at,
            status: &self.status,
            error: &self.error,
            incomplete_details: &self.incomplete_details,
            instructions: &self.instructions,
            max_output_tokens: &self.max_output_tokens,
            model: &self.model,
            reasoning,
            usage: &self.usage,
            output: &self.output,
            tools: &self.tools,
            additional_parameters: &additional_parameters,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CompletionResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let response = CompletionResponseWire::deserialize(deserializer)?;
        let (provider_reasoning, reasoning_metadata) = match response.reasoning {
            Some(Value::String(reasoning)) => (Some(reasoning), None),
            Some(Value::Object(metadata)) => (None, Some(metadata)),
            // Unsupported reasoning shapes must not reject the response.
            _ => (None, None),
        };
        let reasoning_context = reasoning_metadata
            .as_ref()
            .and_then(|reasoning| reasoning.get("context"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        Ok(Self {
            id: response.id,
            object: response.object,
            created_at: response.created_at,
            status: response.status,
            error: response.error,
            incomplete_details: response.incomplete_details,
            instructions: response.instructions,
            max_output_tokens: response.max_output_tokens,
            model: response.model,
            provider_reasoning,
            provider_request_id: None,
            reasoning_metadata,
            reasoning_context,
            usage: response.usage,
            output: response.output,
            tools: response.tools,
            additional_parameters: AdditionalParameters::from_response_metadata(response.metadata),
        })
    }
}

/// Additional parameters for the completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct AdditionalParameters {
    /// Whether or not a given model task should run in the background (ie a detached process).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// The text response format. This is where you would add structured outputs (if you want them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    /// Additional response fields to request from the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<Include>>,
    /// `top_p`. Mutually exclusive with the `temperature` argument.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Whether or not the response should be truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    /// The username of the user (that you want to use).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// A stable cache routing key for prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    /// Any additional metadata you'd like to add. This will additionally be returned by the response.
    #[serde(
        skip_serializing_if = "Map::is_empty",
        default,
        deserialize_with = "deserialize_metadata"
    )]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Whether or not you want tool calls to run in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Previous response ID. If you are not sending a full conversation, this can help to track the message flow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Add thinking/reasoning to your response. The response will be emitted as a list member of the `output` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// The service tier you're using.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    /// Whether or not to store the response for later retrieval by API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

fn deserialize_metadata<'de, D>(
    deserializer: D,
) -> Result<serde_json::Map<String, serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(
        Option::<serde_json::Map<String, serde_json::Value>>::deserialize(deserializer)?
            .unwrap_or_default(),
    )
}

impl AdditionalParameters {
    /// Project echoed response metadata into the request-shaped parameters.
    ///
    /// Each key is decoded on its own; a key whose value does not fit its
    /// field is dropped rather than failing the response. A non-numeric
    /// `top_p` from a compatible endpoint therefore reads back as `None`.
    fn from_response_metadata(metadata: Map<String, Value>) -> Self {
        let mut accepted = Map::with_capacity(metadata.len());
        for (key, value) in metadata {
            let probe = Value::Object(Map::from_iter([(key.clone(), value.clone())]));
            if serde_json::from_value::<Self>(probe).is_ok() {
                accepted.insert(key, value);
            } else {
                tracing::debug!(
                    target: "rig::providers::openai",
                    field = %key,
                    "ignoring response metadata field that does not match its expected type"
                );
            }
        }
        // Every remaining key was individually accepted, so this cannot fail;
        // `unwrap_or_default` keeps the projection total without a panic path.
        serde_json::from_value(Value::Object(accepted)).unwrap_or_default()
    }

    pub fn to_json(self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|_| serde_json::Value::Object(Map::new()))
    }
}

/// The truncation strategy.
/// When using auto, if the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
/// Otherwise, does nothing (and is disabled by default).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    Auto,
    #[default]
    Disabled,
}

/// The model output format configuration.
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextConfig {
    pub format: TextFormat,
}

impl TextConfig {
    pub(crate) fn structured_output<S>(name: S, schema: serde_json::Value) -> Self
    where
        S: Into<String>,
    {
        Self {
            format: TextFormat::JsonSchema(StructuredOutputsInput {
                name: name.into(),
                schema,
                strict: true,
            }),
        }
    }
}

/// The text format (contained by [`TextConfig`]).
/// You can either have plain text by default, or attach a JSON schema for the purposes of structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    JsonSchema(StructuredOutputsInput),
    #[default]
    Text,
}

/// The inputs required for adding structured outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredOutputsInput {
    /// The name of your schema.
    ///
    /// Compatible providers may omit it when echoing a response configuration.
    #[serde(default)]
    pub name: String,
    /// Your required output schema. It is recommended that you use the JsonSchema macro, which you can check out at <https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html>.
    pub schema: serde_json::Value,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    #[serde(default)]
    pub strict: bool,
}

/// Add reasoning to a [`CompletionRequest`].
///
/// # Example
/// ```
/// use rig_core::providers::openai::responses_api::{
///     Reasoning, ReasoningContext, ReasoningEffort, ReasoningMode,
/// };
///
/// // GPT-5.6 reasoning controls: effort, pro mode, and persisted-reasoning context.
/// let reasoning = Reasoning::new()
///     .with_effort(ReasoningEffort::Max)
///     .with_mode(ReasoningMode::Pro)
///     .with_context(ReasoningContext::AllTurns);
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort you want the model to put into thinking/reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// How much effort you want the model to put into writing the reasoning summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryLevel>,
    /// The reasoning mode. Independent from `effort`; the standard mode is
    /// represented by omitting the field. Supported by the GPT-5.6 model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ReasoningMode>,
    /// How persisted reasoning is carried across turns. Supported by the
    /// GPT-5.6 model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
}

impl Reasoning {
    /// Creates a new Reasoning instantiation (with empty values).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds reasoning effort.
    pub fn with_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.effort = Some(reasoning_effort);

        self
    }

    /// Adds summary level (how detailed the reasoning summary will be).
    pub fn with_summary_level(mut self, reasoning_summary_level: ReasoningSummaryLevel) -> Self {
        self.summary = Some(reasoning_summary_level);

        self
    }

    /// Sets the reasoning mode (e.g. pro mode on GPT-5.6 models).
    pub fn with_mode(mut self, reasoning_mode: ReasoningMode) -> Self {
        self.mode = Some(reasoning_mode);

        self
    }

    /// Sets how persisted reasoning is carried across turns (GPT-5.6 models).
    pub fn with_context(mut self, reasoning_context: ReasoningContext) -> Self {
        self.context = Some(reasoning_context);

        self
    }
}

/// The billing service tier that will be used. On auto by default.
#[derive(Clone, Debug, Default)]
pub enum OpenAIServiceTier {
    /// Let OpenAI choose the service tier.
    #[default]
    Auto,
    /// Use the default service tier.
    Default,
    /// Use the flex service tier.
    Flex,
    /// Use the priority service tier.
    Priority,
    /// Use the standard service tier returned by OpenAI-compatible providers.
    Standard,
    /// Preserve an unknown provider-specific service tier.
    Other(String),
}

impl Serialize for OpenAIServiceTier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
            Self::Priority => "priority",
            Self::Standard => "standard",
            Self::Other(value) => value,
        })
    }
}

impl<'de> Deserialize<'de> for OpenAIServiceTier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "auto" => Self::Auto,
            "default" => Self::Default,
            "flex" => Self::Flex,
            "priority" => Self::Priority,
            "standard" => Self::Standard,
            _ => Self::Other(value),
        })
    }
}

/// The amount of reasoning effort that will be used by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
    /// The highest reasoning effort. Supported by the GPT-5.6 model family.
    Max,
}

/// The reasoning mode used by a given model. Independent from
/// [`ReasoningEffort`]; the standard mode is represented by omitting the field
/// (`None` on [`Reasoning::mode`]), so this enum only carries the documented
/// non-default modes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Pro mode. Supported by the GPT-5.6 model family.
    Pro,
}

/// How persisted reasoning is carried across turns. Supported by the GPT-5.6
/// model family.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    /// Let the model decide how much persisted reasoning to reuse.
    #[default]
    Auto,
    /// Reuse persisted reasoning from all previous turns.
    AllTurns,
    /// Only use reasoning from the current turn.
    CurrentTurn,
}

/// The amount of effort that will go into a reasoning summary by a given model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummaryLevel {
    #[default]
    Auto,
    Concise,
    Detailed,
}

/// Additional response fields requested through [`AdditionalParameters::include`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Include {
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    #[serde(rename = "computer_call.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
}

/// A Responses output item. Unrecognized types, including hosted tools, decode
/// to [`Output::Unknown`] with their JSON value preserved. Malformed known
/// types fail deserialization.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        content: Vec<String>,
        encrypted_content: Option<String>,
        /// The upstream's signature over the reasoning text, when a gateway
        /// relays one (OpenRouter for Claude).
        signature: Option<String>,
        status: Option<ToolStatus>,
    },
    /// An opaque compaction item (`"type": "compaction"`), preserved verbatim
    /// so it can be sent back as an input item on the next request. Kept
    /// distinct from [`Output::Unknown`] because OpenAI documents it as a
    /// must-replay item, and [`InputContent::Compaction`] is its input twin.
    Compaction(Map<String, Value>),
    /// Catch-all for output item types this version does not model. Holds the
    /// raw item object exactly as it appeared in the provider's `output[]`
    /// array, so hosted-tool payloads survive the typed decode.
    Unknown(Value),
}

/// Deserialization fields for [`Output::Reasoning`].
#[derive(Deserialize)]
struct ReasoningFields {
    id: String,
    #[serde(default)]
    summary: Vec<ReasoningSummary>,
    #[serde(default, deserialize_with = "deserialize_reasoning_text_content")]
    content: Vec<String>,
    #[serde(default)]
    encrypted_content: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    status: Option<ToolStatus>,
}

impl From<ReasoningFields> for Output {
    fn from(fields: ReasoningFields) -> Self {
        Output::Reasoning {
            id: fields.id,
            summary: fields.summary,
            content: fields.content,
            encrypted_content: fields.encrypted_content,
            signature: fields.signature,
            status: fields.status,
        }
    }
}

/// Serializes an object payload with a `type` tag. Non-object payloads fail.
/// Object key order is not preserved.
fn tagged_output_object<T>(tag: &str, payload: &T) -> Result<Value, serde_json::Error>
where
    T: Serialize,
{
    let mut value = serde_json::to_value(payload)?;
    let map = value.as_object_mut().ok_or_else(|| {
        <serde_json::Error as serde::ser::Error>::custom(
            "output payload must serialize to a JSON object",
        )
    })?;
    map.insert("type".to_string(), Value::String(tag.to_string()));
    Ok(value)
}

impl Serialize for Output {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Output::Message(message) => tagged_output_object("message", message),
            Output::FunctionCall(call) => tagged_output_object("function_call", call),
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                status,
            } => {
                let mut value = serde_json::json!({
                    "type": "reasoning",
                    "id": id,
                    "summary": summary,
                    "encrypted_content": encrypted_content,
                    "status": status,
                });
                let map = value.as_object_mut().ok_or_else(|| {
                    serde::ser::Error::custom("reasoning output must serialize to an object")
                })?;
                if !content.is_empty() {
                    map.insert("content".to_string(), reasoning_text_content_json(content));
                }
                if let Some(signature) = signature {
                    map.insert("signature".to_string(), Value::String(signature.clone()));
                }
                Ok(value)
            }
            Output::Compaction(fields) => {
                let mut map = fields.clone();
                map.insert("type".to_string(), Value::String("compaction".to_string()));
                return Value::Object(map).serialize(serializer);
            }
            Output::Unknown(value) => return value.serialize(serializer),
        };
        value
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Output {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Preserve unmodeled items, including absent or non-string tags.
        // Malformed bodies with known tags must still fail.
        let value = Value::deserialize(deserializer)?;
        let Some(tag) = value.get("type").and_then(Value::as_str) else {
            return Ok(Output::Unknown(value));
        };
        match tag {
            "message" => serde_json::from_value(value)
                .map(Output::Message)
                .map_err(serde::de::Error::custom),
            "function_call" => serde_json::from_value(value)
                .map(Output::FunctionCall)
                .map_err(serde::de::Error::custom),
            "reasoning" => serde_json::from_value::<ReasoningFields>(value)
                .map(Output::from)
                .map_err(serde::de::Error::custom),
            "compaction" => {
                let Value::Object(mut map) = value else {
                    return Ok(Output::Unknown(value));
                };
                map.remove("type");
                Ok(Output::Compaction(map))
            }
            _ => Ok(Output::Unknown(value)),
        }
    }
}

/// An OpenAI Responses API tool call. A call ID will be returned that must be used when creating a tool result to send back to OpenAI as a message input, otherwise an error will be received.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputFunctionCall {
    /// Provider-assigned `fc_...` item ID. The Responses API rejects
    /// `function_call` input IDs that are not native `fc` item IDs, so IDs
    /// minted outside the Responses API (by Rig's agent loop or another
    /// provider) are omitted on serialization and the call is paired with its
    /// output by `call_id` alone.
    #[serde(default, skip_serializing_if = "is_not_function_call_item_id")]
    pub id: String,
    pub arguments: FunctionCallArguments,
    pub call_id: String,
    pub name: String,
    pub status: ToolStatus,
}

/// Raw Responses function-call arguments, parsed as JSON at consumption time.
/// Truncated arguments remain valid wire data; parsing them can fail without
/// discarding the enclosing response or its terminal status.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCallArguments(String);

impl FunctionCallArguments {
    /// Parse the raw wire string into JSON arguments. An empty string is a
    /// parameterless invocation (`{}`); anything else must parse as JSON.
    pub fn parse(&self) -> serde_json::Result<serde_json::Value> {
        json_utils::parse_tool_arguments(&self.0)
    }

    /// The raw wire string, exactly as the provider sent it.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<serde_json::Value> for FunctionCallArguments {
    /// Encode already-parsed arguments (Rig's canonical tool-call form) in
    /// the wire's stringified-JSON spelling.
    fn from(value: serde_json::Value) -> Self {
        Self(value.to_string())
    }
}

impl Serialize for FunctionCallArguments {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for FunctionCallArguments {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // The wire spells arguments as a string; a non-string payload is
        // still a schema defect of the known `function_call` shape.
        String::deserialize(deserializer).map(Self)
    }
}

/// See [`OutputFunctionCall::id`]: only provider-native `fc` item IDs may be
/// sent back to the Responses API.
fn is_not_function_call_item_id(id: &str) -> bool {
    !id.starts_with("fc_")
}

/// The status of a given tool.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// An output message from OpenAI's Responses API.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputMessage {
    /// The message ID. Must be included when sending the message back to OpenAI
    pub id: String,
    /// The role (currently only Assistant is available as this struct is only created when receiving an LLM message as a response)
    pub role: OutputRole,
    /// The status of the response
    pub status: ResponseStatus,
    /// The actual message content
    pub content: Vec<AssistantContent>,
    /// Generation phase, such as `"final_answer"`, preserved for follow-up requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

/// The role of an output message.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputRole {
    Assistant,
}

/// An OpenAI Responses API message.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_vec")]
        content: Vec<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_vec")]
        content: Vec<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: Vec<AssistantContentType>,
        #[serde(skip_serializing_if = "String::is_empty")]
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        status: ToolStatus,
        /// Generation phase preserved from the output message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<String>,
    },
    #[serde(rename = "assistant", skip_deserializing)]
    AssistantInput {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: vec![content.to_owned().into()],
            name: None,
        }
    }
}

/// Text assistant content.
/// Note that the text type in comparison to the Completions API is actually `output_text` rather than `text`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    OutputText(OutputText),
    Refusal { refusal: String },
}

/// Responses `output_text` block with unmodeled sibling fields preserved as JSON.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct OutputText {
    pub text: String,
    /// OpenAI's sibling keys, preserved verbatim for value-equal replay.
    /// The `Map` form (not `Option<Value>`) makes absence and the empty map
    /// one value, so a decoded bare block equals a request-assembled one.
    #[serde(flatten, default, skip_serializing_if = "Map::is_empty")]
    pub extras: Map<String, Value>,
}

impl OutputText {
    /// A bare text block, as request assembly emits (no wire extras).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            extras: Map::new(),
        }
    }

    /// Rebuild a wire block from a rig text block, re-attaching only the
    /// extras this wire recognizes as its own: the sibling keys captured off
    /// an `output_text` block at ingest (see
    /// [`From<AssistantContent> for completion::AssistantContent`]).
    fn from_message_text(
        text: impl Into<String>,
        additional_params: Option<crate::message::AdditionalParams>,
    ) -> Self {
        let Some(params) = additional_params else {
            return Self::new(text);
        };
        // The caller diagnoses malformed extras before this conversion drops them.
        let extras = params
            .into_wire_extras(OPENAI_RESPONSES_EXTRAS_KEY)
            .map(|map| {
                map.into_iter()
                    // Reserved keys would duplicate the block's text or tag.
                    // Phase belongs on the message, not the content block.
                    .filter(|(key, _)| {
                        key != "text" && key != "type" && key != OPENAI_RESPONSES_PHASE_KEY
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            text: text.into(),
            extras,
        }
    }
}

/// Builds an assistant input using only Responses-owned extras. Empty text is
/// skipped unless both an ID and owned extras are present. Malformed extras
/// warn before skipping; extras on nonempty ID-less inputs warn and are dropped.
fn assistant_text_replay_message(
    id: Option<String>,
    text: String,
    additional_params: Option<crate::message::AdditionalParams>,
) -> Option<Message> {
    // Diagnose malformed extras before normalization makes them indistinguishable
    // from absent extras, including when empty text is skipped.
    if let Some(non_object) = additional_params
        .as_ref()
        .and_then(|params| params.get(OPENAI_RESPONSES_EXTRAS_KEY))
        .filter(|value| !value.is_object())
    {
        tracing::warn!(
            %non_object,
            "`additional_params[\"{OPENAI_RESPONSES_EXTRAS_KEY}\"]` must be a JSON \
             object — replaying without these extras"
        );
    }
    let own_extras = additional_params
        .as_ref()
        .and_then(|params| params.wire_extras(OPENAI_RESPONSES_EXTRAS_KEY))
        .is_some();
    if text.is_empty() && !(own_extras && id.is_some()) {
        return None;
    }
    // `phase` rides the text block's own-wire extras on ingest; it belongs
    // to the message, so it is lifted here and filtered from the block.
    let phase = additional_params
        .as_ref()
        .and_then(|params| params.wire_extras(OPENAI_RESPONSES_EXTRAS_KEY))
        .and_then(|extras| extras.get(OPENAI_RESPONSES_PHASE_KEY))
        .and_then(Value::as_str)
        .map(str::to_owned);
    match id {
        Some(id) => Some(Message::Assistant {
            content: vec![AssistantContentType::Text(AssistantContent::OutputText(
                OutputText::from_message_text(text, additional_params),
            ))],
            id,
            name: None,
            status: ToolStatus::Completed,
            phase,
        }),
        None => {
            if own_extras {
                tracing::warn!(
                    "own-wire extras cannot ride the id-less assistant form — \
                     replaying the text without them"
                );
            }
            Some(Message::AssistantInput {
                content: text,
                name: None,
            })
        }
    }
}

/// Responses-owned extras in [`Text::additional_params`](crate::message::Text).
/// Buffered output captures these fields for replay; streaming annotation events
/// are not captured.
pub(crate) const OPENAI_RESPONSES_EXTRAS_KEY: &str = "openai_responses";

/// Key inside the [`OPENAI_RESPONSES_EXTRAS_KEY`] object that carries the
/// output message's `phase`. It is message-level on the wire but rides the
/// text block's extras in rig history (the only own-wire seat), and is
/// lifted back onto the assistant input item at replay.
pub(crate) const OPENAI_RESPONSES_PHASE_KEY: &str = "phase";

/// Record an output message's `phase` on a text block's own-wire extras so
/// the follow-up request can re-send it.
pub(crate) fn stamp_phase(text: &mut Text, phase: Option<&str>) {
    let Some(phase) = phase else {
        return;
    };
    let mut extras = text
        .additional_params
        .as_ref()
        .and_then(|params| params.wire_extras(OPENAI_RESPONSES_EXTRAS_KEY))
        .cloned()
        .unwrap_or_default();
    extras.insert(
        OPENAI_RESPONSES_PHASE_KEY.to_string(),
        Value::String(phase.to_string()),
    );
    text.additional_params = crate::message::AdditionalParams::from_entries(Some((
        OPENAI_RESPONSES_EXTRAS_KEY,
        Value::Object(extras),
    )));
}

/// Converts output text or a refusal to a Rig text block, retaining nonempty
/// output-text extras under the Responses key.
pub(crate) fn text_block(value: AssistantContent) -> Text {
    match value {
        AssistantContent::Refusal { refusal } => Text::new(refusal),
        // Keep this destructuring exhaustive so new wire fields force an
        // explicit capture-or-drop decision.
        AssistantContent::OutputText(OutputText { text, extras }) => {
            // Empty metadata must not change replayed request bytes.
            let extras: Map<String, Value> = extras
                .into_iter()
                .filter(|(_, value)| {
                    !(value.is_null()
                        || value.as_array().is_some_and(Vec::is_empty)
                        || value.as_object().is_some_and(Map::is_empty))
                })
                .collect();
            Text {
                text,
                additional_params: crate::message::AdditionalParams::from_entries(
                    (!extras.is_empty())
                        .then_some((OPENAI_RESPONSES_EXTRAS_KEY, Value::Object(extras))),
                ),
            }
        }
    }
}

impl From<AssistantContent> for completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        completion::AssistantContent::Text(text_block(value))
    }
}

/// The type of assistant content.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum AssistantContentType {
    Text(AssistantContent),
    ToolCall(OutputFunctionCall),
    Reasoning(OpenAIReasoning),
}

/// System content for the OpenAI Responses API.
/// Uses `input_text` type to match the Responses API format.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContent {
    InputText { text: String },
}

impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent::InputText { text: s }
    }
}

impl std::str::FromStr for SystemContent {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent::InputText {
            text: s.to_string(),
        })
    }
}

/// Different types of user content.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default)]
        detail: ImageDetail,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::InputText {
            text: s.to_string(),
        })
    }
}

#[cfg(test)]
mod stateless_replay_tests;
#[cfg(test)]
mod tests;
