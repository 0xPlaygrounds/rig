//! The OpenAI Responses API.
//!
//! By default when creating a completion client, this is the API that gets used.
//!
//! If you'd like to switch back to the regular Completions API, you can do so by using the `.completions_api()` function - see below for an example:
//! ```rust
//! use rig_core::client::{CompletionClient, ProviderClient};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let openai_client = rig_core::providers::openai::Client::from_env()?;
//! let model = openai_client.completion_model("gpt-4o").completions_api();
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
use super::InputAudio;
use super::responses_api::streaming::StreamingCompletionResponse;
use crate::completion::{CompletionError, GetTokenUsage};
use crate::http_client;
use crate::http_client::HttpClientExt;
use crate::json_utils;
use crate::message::{
    AudioMediaType, Document, DocumentMediaType, DocumentSourceKind, ImageDetail, MessageError,
    MimeType, Text,
};
use crate::one_or_many::string_or_one_or_many;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, completion, message};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use tracing::{Instrument, Level, enabled};

use std::convert::Infallible;
use std::ops::Add;
use std::str::FromStr;

pub mod streaming;
#[cfg(all(not(target_family = "wasm"), feature = "websocket"))]
pub mod websocket;

/// The completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionRequest {
    /// Message inputs
    pub input: OneOrMany<InputItem>,
    /// The model name
    pub model: String,
    /// Instructions (also referred to as preamble, although in other APIs this would be the "system prompt")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The maximum number of output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Toggle to true for streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The temperature. Set higher (up to a max of 1.0) for more creative responses.
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
    pub fn with_structured_outputs<S>(mut self, schema_name: S, schema: serde_json::Value) -> Self
    where
        S: Into<String>,
    {
        self.additional_parameters.text = Some(TextConfig::structured_output(schema_name, schema));

        self
    }

    pub fn with_reasoning(mut self, reasoning: Reasoning) -> Self {
        self.additional_parameters.reasoning = Some(reasoning);

        self
    }

    /// Adds a provider-native hosted tool (e.g. `web_search`, `file_search`, `computer_use`)
    /// to the request. These tools are executed by OpenAI's infrastructure, not by Rig's
    /// agent loop.
    pub fn with_tool(mut self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Adds multiple provider-native hosted tools to the request. These tools are executed
    /// by OpenAI's infrastructure, not by Rig's agent loop.
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
                content: OneOrMany::one(SystemContent::InputText {
                    text: content.into(),
                }),
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

/// The type of content used in an [`InputItem`]. Additionally holds data for each type of input content.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    Message(Message),
    Reasoning(OpenAIReasoning),
    FunctionCall(OutputFunctionCall),
    FunctionCallOutput(ToolResult),
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

    pub fn text(&self) -> String {
        let ReasoningSummary::SummaryText { text } = self;
        text.clone()
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

fn responses_tool_result_output(
    content: OneOrMany<message::ToolResultContent>,
) -> Result<ToolResultOutput, MessageError> {
    let mut text_output = Vec::new();
    let mut rich_output = Vec::new();
    let mut has_image = false;

    for content in content {
        match content {
            message::ToolResultContent::Text(Text { text, .. }) => {
                text_output.push(text.clone());
                rich_output.push(ToolResultOutputContent::InputText { text });
            }
            message::ToolResultContent::Json { value } => {
                let text = value.to_string();
                text_output.push(text.clone());
                rich_output.push(ToolResultOutputContent::InputText { text });
            }
            message::ToolResultContent::Image(message::Image {
                data,
                media_type,
                detail,
                ..
            }) => {
                has_image = true;
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

    if has_image {
        Ok(ToolResultOutput::Content(rich_output))
    } else {
        Ok(ToolResultOutput::Text(text_output.join("\n")))
    }
}

impl From<Message> for InputItem {
    fn from(value: Message) -> Self {
        match value {
            Message::User { .. } => Self {
                role: Some(Role::User),
                input: InputContent::Message(value),
            },
            Message::Assistant { ref content, .. } => {
                let role = if content
                    .iter()
                    .any(|x| matches!(x, AssistantContentType::Reasoning(_)))
                {
                    None
                } else {
                    Some(Role::Assistant)
                };
                Self {
                    role,
                    input: InputContent::Message(value),
                }
            }
            Message::AssistantInput { .. } => Self {
                role: Some(Role::Assistant),
                input: InputContent::Message(value),
            },
            Message::System { .. } => Self {
                role: Some(Role::System),
                input: InputContent::Message(value),
            },
            Message::ToolResult {
                tool_call_id,
                output,
            } => Self {
                role: None,
                input: InputContent::FunctionCallOutput(ToolResult {
                    call_id: tool_call_id,
                    output,
                    status: ToolStatus::Completed,
                }),
            },
        }
    }
}

impl TryFrom<crate::completion::Message> for Vec<InputItem> {
    type Error = CompletionError;

    fn try_from(value: crate::completion::Message) -> Result<Self, Self::Error> {
        match value {
            crate::completion::Message::System { content } => Ok(vec![InputItem {
                role: Some(Role::System),
                input: InputContent::Message(Message::System {
                    content: OneOrMany::one(content.into()),
                    name: None,
                }),
            }]),
            crate::completion::Message::User { content } => {
                let mut items = Vec::new();

                for user_content in content {
                    match user_content {
                        crate::message::UserContent::Text(Text { text, .. }) => {
                            items.push(InputItem {
                                role: Some(Role::User),
                                input: InputContent::Message(Message::User {
                                    content: OneOrMany::one(UserContent::InputText { text }),
                                    name: None,
                                }),
                            });
                        }
                        crate::message::UserContent::ToolResult(
                            crate::completion::message::ToolResult {
                                call_id,
                                content: tool_content,
                                ..
                            },
                        ) => {
                            let output =
                                responses_tool_result_output(tool_content).map_err(|error| {
                                    CompletionError::ProviderError(error.to_string())
                                })?;
                            items.push(InputItem {
                                role: None,
                                input: InputContent::FunctionCallOutput(ToolResult {
                                    call_id: require_call_id(call_id, "Tool result")?,
                                    output,
                                    status: ToolStatus::Completed,
                                }),
                            });
                        }
                        crate::message::UserContent::Document(Document {
                            data: DocumentSourceKind::FileId(file_id),
                            ..
                        }) => items.push(InputItem {
                            role: Some(Role::User),
                            input: InputContent::Message(Message::User {
                                content: OneOrMany::one(UserContent::InputFile {
                                    file_id: Some(file_id),
                                    file_data: None,
                                    file_url: None,
                                    filename: None,
                                }),
                                name: None,
                            }),
                        }),
                        crate::message::UserContent::Document(Document {
                            data,
                            media_type: Some(DocumentMediaType::PDF),
                            ..
                        }) => {
                            let (file_data, file_url) = match data {
                                DocumentSourceKind::Base64(data) => {
                                    (Some(format!("data:application/pdf;base64,{data}")), None)
                                }
                                DocumentSourceKind::Url(url) => (None, Some(url)),
                                DocumentSourceKind::Raw(_) => {
                                    return Err(CompletionError::RequestError(
                                        "Raw file data not supported, encode as base64 first"
                                            .into(),
                                    ));
                                }
                                doc => {
                                    return Err(CompletionError::RequestError(
                                        format!("Unsupported document type: {doc}").into(),
                                    ));
                                }
                            };

                            items.push(InputItem {
                                role: Some(Role::User),
                                input: InputContent::Message(Message::User {
                                    content: OneOrMany::one(UserContent::InputFile {
                                        file_id: None,
                                        file_data,
                                        file_url,
                                        filename: Some("document.pdf".to_string()),
                                    }),
                                    name: None,
                                }),
                            })
                        }
                        crate::message::UserContent::Document(Document {
                            data:
                                DocumentSourceKind::Base64(text) | DocumentSourceKind::String(text),
                            ..
                        }) => items.push(InputItem {
                            role: Some(Role::User),
                            input: InputContent::Message(Message::User {
                                content: OneOrMany::one(UserContent::InputText { text }),
                                name: None,
                            }),
                        }),
                        crate::message::UserContent::Image(crate::message::Image {
                            data,
                            media_type,
                            detail,
                            ..
                        }) => {
                            let url = match data {
                                DocumentSourceKind::Base64(data) => {
                                    let media_type = if let Some(media_type) = media_type {
                                        media_type.to_mime_type().to_string()
                                    } else {
                                        String::new()
                                    };
                                    format!("data:{media_type};base64,{data}")
                                }
                                DocumentSourceKind::Url(url) => url,
                                DocumentSourceKind::Raw(_) => {
                                    return Err(CompletionError::RequestError(
                                        "Raw file data not supported, encode as base64 first"
                                            .into(),
                                    ));
                                }
                                doc => {
                                    return Err(CompletionError::RequestError(
                                        format!("Unsupported document type: {doc}").into(),
                                    ));
                                }
                            };
                            items.push(InputItem {
                                role: Some(Role::User),
                                input: InputContent::Message(Message::User {
                                    content: OneOrMany::one(UserContent::InputImage {
                                        image_url: url,
                                        detail: detail.unwrap_or_default(),
                                    }),
                                    name: None,
                                }),
                            });
                        }
                        message => {
                            return Err(CompletionError::ProviderError(format!(
                                "Unsupported message: {message:?}"
                            )));
                        }
                    }
                }

                Ok(items)
            }
            crate::completion::Message::Assistant { id, content } => {
                let mut reasoning_items = Vec::new();
                let mut other_items = Vec::new();

                for assistant_content in content {
                    match assistant_content {
                        crate::message::AssistantContent::Text(Text { text, .. }) => {
                            if text.is_empty() {
                                continue;
                            }
                            let message = if let Some(id) = id.clone() {
                                Message::Assistant {
                                    content: OneOrMany::one(AssistantContentType::Text(
                                        AssistantContent::OutputText(Text::new(text)),
                                    )),
                                    id,
                                    name: None,
                                    status: ToolStatus::Completed,
                                }
                            } else {
                                Message::AssistantInput {
                                    content: text,
                                    name: None,
                                }
                            };

                            other_items.push(InputItem {
                                role: Some(Role::Assistant),
                                input: InputContent::Message(message),
                            });
                        }
                        crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
                            id: tool_id,
                            call_id,
                            function,
                            ..
                        }) => {
                            other_items.push(InputItem {
                                role: None,
                                input: InputContent::FunctionCall(OutputFunctionCall {
                                    arguments: function.arguments,
                                    call_id: require_call_id(call_id, "Assistant tool call")?,
                                    id: tool_id,
                                    name: function.name,
                                    status: ToolStatus::Completed,
                                }),
                            });
                        }
                        crate::message::AssistantContent::Reasoning(reasoning) => {
                            let openai_reasoning = openai_reasoning_from_core(&reasoning)
                                .map_err(|err| CompletionError::ProviderError(err.to_string()))?;
                            if let Some(openai_reasoning) = openai_reasoning {
                                reasoning_items.push(InputItem {
                                    role: None,
                                    input: InputContent::Reasoning(openai_reasoning),
                                });
                            }
                        }
                        crate::message::AssistantContent::Image(_) => {
                            return Err(CompletionError::ProviderError(
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

impl From<OneOrMany<String>> for Vec<ReasoningSummary> {
    fn from(value: OneOrMany<String>) -> Self {
        value.iter().map(|x| ReasoningSummary::new(x)).collect()
    }
}

fn require_call_id(call_id: Option<String>, context: &str) -> Result<String, CompletionError> {
    call_id.ok_or_else(|| {
        CompletionError::RequestError(
            format!("{context} `call_id` is required for OpenAI Responses API").into(),
        )
    })
}

fn openai_reasoning_from_core(
    reasoning: &crate::message::Reasoning,
) -> Result<Option<OpenAIReasoning>, MessageError> {
    let Some(id) = reasoning.id.clone() else {
        return Ok(None);
    };

    let mut summary = Vec::new();
    let mut reasoning_content = Vec::new();
    let mut encrypted_content = None;
    for content in &reasoning.content {
        match content {
            crate::message::ReasoningContent::Text { text, .. } => {
                reasoning_content.push(text.clone());
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

    Ok(Some(OpenAIReasoning {
        id,
        summary,
        content: reasoning_content,
        encrypted_content,
        status: None,
    }))
}

/// The definition of a tool response, repurposed for OpenAI's Responses API.
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
    /// or [`GenericResponsesCompletionModel::with_strict_tools`].
    #[serde(
        default,
        skip_serializing_if = "is_false",
        deserialize_with = "json_utils::null_or_default"
    )]
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

fn is_false(value: &bool) -> bool {
    !value
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

/// Tool choice for the OpenAI Responses API.
///
/// The Responses API accepts the `"auto"`/`"none"`/`"required"` modes shared
/// with the Chat Completions API, and additionally supports forcing one
/// specific function (`{"type": "function", "name": "..."}`) or restricting
/// the model to a subset of the request's tools
/// (`{"type": "allowed_tools", ...}`).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"`, `"none"`, or `"required"`. The wrapped chat-completions
    /// enum also has a `Function` variant whose nested wire shape the
    /// Responses API rejects — use [`ToolChoiceDefinition::Function`] to
    /// force a function here.
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
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let choice = match value {
            message::ToolChoice::Auto => Self::Mode(super::completion::ToolChoice::Auto),
            message::ToolChoice::None => Self::Mode(super::completion::ToolChoice::None),
            message::ToolChoice::Required => Self::Mode(super::completion::ToolChoice::Required),
            message::ToolChoice::Specific { function_names } => {
                let mut names = function_names.into_iter();
                let Some(first) = names.next() else {
                    return Err(CompletionError::RequestError(
                        "ToolChoice::Specific requires at least one function name".into(),
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

/// Token usage.
/// Token usage from the OpenAI Responses API generally shows the input tokens and output tokens (both with more in-depth details) as well as a total tokens field.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl ResponsesUsage {
    /// Create a new ResponsesUsage instance
    pub(crate) fn new() -> Self {
        Self {
            input_tokens: 0,
            input_tokens_details: Some(InputTokensDetails::new()),
            output_tokens: 0,
            output_tokens_details: Some(OutputTokensDetails::new()),
            total_tokens: 0,
        }
    }
}

impl GetTokenUsage for ResponsesUsage {
    fn token_usage(&self) -> crate::completion::Usage {
        crate::completion::Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            total_tokens: self.total_tokens,
            cached_input_tokens: self
                .input_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens)
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens)
                .unwrap_or(0),
        }
    }
}

impl Add for ResponsesUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let input_tokens = self.input_tokens + rhs.input_tokens;
        let input_tokens_details = match (self.input_tokens_details, rhs.input_tokens_details) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        };
        let output_tokens = self.output_tokens + rhs.output_tokens;
        let output_tokens_details = match (self.output_tokens_details, rhs.output_tokens_details) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        };
        let total_tokens = self.total_tokens + rhs.total_tokens;
        Self {
            input_tokens,
            input_tokens_details,
            output_tokens,
            output_tokens_details,
            total_tokens,
        }
    }
}

/// In-depth details on input tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Cached tokens from OpenAI
    pub cached_tokens: u64,
}

impl InputTokensDetails {
    pub(crate) fn new() -> Self {
        Self { cached_tokens: 0 }
    }
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens
    pub reasoning_tokens: u64,
}

impl OutputTokensDetails {
    pub(crate) fn new() -> Self {
        Self {
            reasoning_tokens: 0,
        }
    }
}

impl Add for OutputTokensDetails {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
        }
    }
}

/// Occasionally, when using OpenAI's Responses API you may get an incomplete response. This struct holds the reason as to why it happened.
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
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Incomplete,
}

/// Controls where Rig system instructions are placed in an OpenAI Responses request.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Provider extensions that drive the OpenAI Responses request conversion.
///
/// Implemented by the `Ext` type of a [`crate::client::Client`] used with
/// [`GenericResponsesCompletionModel`], so a client-level configuration can
/// control request shaping for every model created from that client.
pub trait ResponsesProviderExt {
    /// Provider name recorded on `gen_ai.provider.name` telemetry spans.
    ///
    /// Defaults to `"openai"`. OpenAI-compatible backends on other hosts (e.g.
    /// Amazon Bedrock Mantle) override this so metrics are not attributed to
    /// OpenAI.
    const PROVIDER_NAME: &'static str = "openai";

    /// Where Rig system instructions are placed in requests built from this
    /// provider. See [`SystemInstructionsPlacement`].
    ///
    /// Deliberately has no default body: each provider must state its
    /// placement explicitly, so a backend that can't handle the default
    /// (top-level `instructions`) is never inherited by accident.
    fn system_instructions_placement(&self) -> SystemInstructionsPlacement;
}

/// Attempt to try and create a `NewCompletionRequest` from a model name and [`crate::completion::CompletionRequest`]
impl TryFrom<(String, crate::completion::CompletionRequest)> for CompletionRequest {
    type Error = CompletionError;
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
    type Error = CompletionError;

    fn try_from(params: ResponsesRequestParams) -> Result<Self, Self::Error> {
        let ResponsesRequestParams {
            model,
            request: mut req,
            system_instructions_placement,
        } = params;
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or(model);
        let preamble = req.preamble.take();
        let mut instruction_parts = Vec::new();
        let mut input = {
            let mut partial_history = vec![];
            partial_history.extend(chat_history);

            let mut full_history: Vec<InputItem> = preamble
                .map(InputItem::system_message)
                .into_iter()
                .collect();

            for history_item in partial_history {
                full_history.extend(<Vec<InputItem>>::try_from(history_item)?);
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

        let input = OneOrMany::many(input).map_err(|_| {
            CompletionError::RequestError(if lifted_system_items {
                "OpenAI Responses request input must contain at least one non-system item \
                 (system messages were lifted into the top-level `instructions` field)"
                    .into()
            } else {
                "OpenAI Responses request input must contain at least one item".into()
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
                    CompletionError::RequestError(
                        format!(
                            "Invalid OpenAI Responses tools payload in additional_params: {err}"
                        )
                        .into(),
                    )
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
                    CompletionError::RequestError(
                        format!("Invalid OpenAI Responses additional_params payload: {err}").into(),
                    )
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
            let name = schema
                .as_object()
                .and_then(|o| o.get("title"))
                .and_then(|v| v.as_str())
                .unwrap_or("response_schema")
                .to_string();
            let mut schema_value = schema.to_value();
            super::sanitize_schema(&mut schema_value);
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

/// The completion model struct for OpenAI's response API.
#[doc(hidden)]
#[derive(Clone)]
pub struct GenericResponsesCompletionModel<Ext = super::OpenAIResponsesExt, H = reqwest::Client> {
    /// The OpenAI client
    pub(crate) client: crate::client::Client<Ext, H>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
    /// Model-level default tools that are always added to outgoing requests.
    pub tools: Vec<ResponsesToolDefinition>,
    /// Whether function tools should use strict mode. Disabled by default to match
    /// the Chat Completions API; enable with [`Self::with_strict_tools`].
    pub strict_tools: bool,
    system_instructions_placement: SystemInstructionsPlacement,
}

/// The completion model struct for OpenAI's Responses API.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type ResponsesCompletionModel<H = reqwest::Client> =
    GenericResponsesCompletionModel<super::OpenAIResponsesExt, H>;

impl<Ext, H> GenericResponsesCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + std::fmt::Debug + 'static,
    Ext: crate::client::Provider + ResponsesProviderExt + Clone + 'static,
    H: Clone + Default + std::fmt::Debug + 'static,
{
    /// Creates a new [`ResponsesCompletionModel`].
    pub fn new(client: crate::client::Client<Ext, H>, model: impl Into<String>) -> Self {
        let system_instructions_placement = client.ext().system_instructions_placement();
        Self {
            client,
            model: model.into(),
            tools: Vec::new(),
            strict_tools: false,
            system_instructions_placement,
        }
    }

    pub fn with_model(client: crate::client::Client<Ext, H>, model: &str) -> Self {
        Self::new(client, model)
    }

    /// Enable strict mode for function tool schemas.
    ///
    /// When enabled, function tool schemas are sanitized to meet OpenAI's strict
    /// mode requirements and `strict: true` is set on each function definition.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Sets where Rig system instructions are placed in requests from this
    /// model, overriding the client-level default. See
    /// [`SystemInstructionsPlacement`] for when each placement applies.
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.system_instructions_placement = placement;
        self
    }

    /// Sends Rig system instructions as `system` messages in `input` instead of
    /// as top-level Responses API `instructions`.
    ///
    /// OpenAI's Responses API supports `instructions`, and Rig uses it by
    /// default. Use this compatibility fallback for OpenAI-compatible providers
    /// that reject or ignore top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// Adds a default tool to all requests from this model.
    pub fn with_tool(mut self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Adds default tools to all requests from this model.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Attempt to create a completion request from [`crate::completion::CompletionRequest`].
    pub(crate) fn create_completion_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<CompletionRequest, CompletionError> {
        let mut req = CompletionRequest::try_from(ResponsesRequestParams {
            model: self.model.clone(),
            request: completion_request,
            system_instructions_placement: self.system_instructions_placement,
        })?;
        req.tools.extend(self.tools.clone());

        if self.strict_tools {
            req.tools = req
                .tools
                .into_iter()
                .map(ResponsesToolDefinition::normalize)
                .collect();
        }

        Ok(req)
    }
}

impl<T> GenericResponsesCompletionModel<super::OpenAIResponsesExt, T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    /// Use the Completions API instead of Responses.
    pub fn completions_api(self) -> crate::providers::openai::completion::CompletionModel<T> {
        super::completion::CompletionModel::new(self.client.completions_api(), &self.model)
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

#[derive(Deserialize)]
struct CompletionResponseWire {
    id: String,
    object: ResponseObject,
    /// Some OpenAI-compatible hosts (e.g. Bedrock Mantle) emit floats.
    #[serde(deserialize_with = "deserialize_created_at")]
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
    additional_parameters: AdditionalParameters,
}

/// Accept integer or float Unix timestamps; keep the public field as `u64`.
///
/// Bedrock Mantle (and some other OpenAI-compatible gateways) return
/// `created_at` as a JSON float (e.g. `1784497980.0`). Truncate toward zero.
fn deserialize_created_at<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                return Ok(u);
            }
            if let Some(i) = n.as_i64() {
                if i < 0 {
                    return Err(serde::de::Error::custom(format!(
                        "created_at must be non-negative, got {i}"
                    )));
                }
                return Ok(i as u64);
            }
            if let Some(f) = n.as_f64() {
                if !f.is_finite() || f < 0.0 {
                    return Err(serde::de::Error::custom(format!(
                        "created_at must be a non-negative finite number, got {f}"
                    )));
                }
                return Ok(f as u64);
            }
            Err(serde::de::Error::custom(format!(
                "created_at number could not be converted to u64: {n}"
            )))
        }
        other => Err(serde::de::Error::custom(format!(
            "created_at must be a number, got {other}"
        ))),
    }
}

impl Serialize for CompletionResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // `AdditionalParameters::reasoning` models request configuration. A
        // response's top-level `reasoning` field is represented by the three
        // response surfaces above, so omit the request field here to avoid
        // serializing duplicate `reasoning` keys.
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
            // Null, arrays, numbers, and booleans are not documented top-level
            // reasoning response shapes. Ignore them to preserve the lenient
            // behavior that predates object-shaped metadata support.
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
            reasoning_metadata,
            reasoning_context,
            usage: response.usage,
            output: response.output,
            tools: response.tools,
            additional_parameters: response.additional_parameters,
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
    /// What types of extra data you would like to include. This is mostly useless at the moment since the types of extra data to add is currently unsupported, but this will be coming soon!
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

/// Results to additionally include in the OpenAI Responses API.
/// Note that most of these are currently unsupported, but have been added for completeness.
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

/// A modeled output item from the OpenAI Responses API.
///
/// Unrecognized output items — notably provider-native hosted tools such as
/// `web_search_call`, `file_search_call`, `computer_call`, and
/// `code_interpreter_call` — decode to [`Output::Unknown`], which preserves
/// the verbatim item object so callers can inspect or forward it. This keeps
/// unknown item types from breaking deserialization of the entire
/// `CompletionResponse` (the invariant that previously caused streaming token
/// usage to be silently dropped) without discarding the payload along the way.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        content: Vec<String>,
        encrypted_content: Option<String>,
        status: Option<ToolStatus>,
    },
    /// Catch-all for output item types this version does not model. Holds the
    /// raw item object exactly as it appeared in the provider's `output[]`
    /// array, so hosted-tool payloads survive the typed decode.
    Unknown(Value),
}

/// Deserialize helper for the inline-field [`Output::Reasoning`] variant.
///
/// `Output`'s (de)serialization is hand-written so [`Output::Unknown`] can carry
/// a raw [`Value`] (`#[serde(other)]` only applies to a unit variant, which
/// would force the payload to be dropped). The modeled `Message`/`FunctionCall`
/// variants deserialize straight into their payload structs; `Reasoning` has no
/// payload struct of its own, so this mirrors its fields. Same approach as
/// Anthropic's `Citation`.
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
    status: Option<ToolStatus>,
}

impl From<ReasoningFields> for Output {
    fn from(fields: ReasoningFields) -> Self {
        Output::Reasoning {
            id: fields.id,
            summary: fields.summary,
            content: fields.content,
            encrypted_content: fields.encrypted_content,
            status: fields.status,
        }
    }
}

/// Serialize a modeled payload as its tagged wire object — the payload's own
/// fields plus the internally tagged `"type"`. The key is appended, so the
/// result is value-equal (not byte-for-byte ordered) to the original item.
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
        // Hand-written to keep `Unknown` verbatim (mirrors Anthropic's
        // `Citation`). Known variants emit their modeled fields plus the
        // internally tagged `type`; `Unknown` re-emits its raw value. The result
        // is value-equal — not byte-for-byte — to the wire item, since `type` is
        // appended rather than threaded in declaration order.
        let value = match self {
            Output::Message(message) => tagged_output_object("message", message),
            Output::FunctionCall(call) => tagged_output_object("function_call", call),
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                status,
            } => {
                let mut value = serde_json::json!({
                    "type": "reasoning",
                    "id": id,
                    "summary": summary,
                    "encrypted_content": encrypted_content,
                    "status": status,
                });
                if !content.is_empty() {
                    let map = value.as_object_mut().ok_or_else(|| {
                        serde::ser::Error::custom("reasoning output must serialize to an object")
                    })?;
                    map.insert("content".to_string(), reasoning_text_content_json(content));
                }
                Ok(value)
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
        // Decode to a `Value` first so an unmodeled item is captured verbatim as
        // `Unknown`. A modeled `type` with a malformed body still errors (rather
        // than silently degrading to `Unknown`); an absent or non-string `type`
        // is itself unmodeled and is captured as `Unknown`. Mirrors `Citation`.
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
            _ => Ok(Output::Unknown(value)),
        }
    }
}

impl From<Output> for Vec<completion::AssistantContent> {
    fn from(value: Output) -> Self {
        let res: Vec<completion::AssistantContent> = match value {
            Output::Message(OutputMessage { content, .. }) => content
                .into_iter()
                .map(completion::AssistantContent::from)
                .collect(),
            Output::FunctionCall(OutputFunctionCall {
                id,
                arguments,
                call_id,
                name,
                ..
            }) => vec![completion::AssistantContent::tool_call_with_call_id(
                id, call_id, name, arguments,
            )],
            Output::Reasoning {
                id,
                summary,
                content: reasoning_content,
                encrypted_content,
                ..
            } => {
                let mut content = summary
                    .into_iter()
                    .map(|summary| match summary {
                        ReasoningSummary::SummaryText { text } => {
                            message::ReasoningContent::Summary(text)
                        }
                    })
                    .collect::<Vec<_>>();
                content.extend(reasoning_content.into_iter().map(|text| {
                    message::ReasoningContent::Text {
                        text,
                        signature: None,
                    }
                }));
                if let Some(encrypted_content) = encrypted_content {
                    content.push(message::ReasoningContent::Encrypted(encrypted_content));
                }
                vec![completion::AssistantContent::Reasoning(
                    message::Reasoning {
                        id: Some(id),
                        content,
                    },
                )]
            }
            Output::Unknown(_) => Vec::new(),
        };

        res
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputReasoning {
    id: String,
    summary: Vec<ReasoningSummary>,
    status: ToolStatus,
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
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
    pub call_id: String,
    pub name: String,
    pub status: ToolStatus,
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
}

/// The role of an output message.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputRole {
    Assistant,
}

impl<Ext, H> completion::CompletionModel for GenericResponsesCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: crate::client::Provider
        + ResponsesProviderExt
        + crate::client::DebugExt
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    H: Clone + Default + std::fmt::Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = crate::client::Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    // The OpenAI Responses API constrains only the final assistant message via
    // `text.format`; tools are still called across turns, so native structured
    // output composes with tool calls. See issue #1928.
    fn composes_native_output_with_tools(&self) -> bool {
        true
    }

    async fn completion(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = self.create_completion_request(completion_request)?;
        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &request.model,
            CompletionOperation::Chat,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();
        let body = serde_json::to_vec(&request)?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Responses completion request: {request}",
                request = serde_json::to_string_pretty(&request)?
            );
        }

        let req = self
            .client
            .post("/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let t = http_client::text(response).await?;
                let response = serde_json::from_str::<Self::Response>(&t)?;
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &response.id);
                span.record("gen_ai.response.model", &response.model);
                if let Some(ref usage) = response.usage {
                    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                    let cached_tokens = usage
                        .input_tokens_details
                        .as_ref()
                        .map(|d| d.cached_tokens)
                        .unwrap_or(0);
                    span.record("gen_ai.usage.cache_read.input_tokens", cached_tokens);
                }
                if enabled!(Level::TRACE) {
                    tracing::trace!(
                        target: "rig::completions",
                        "OpenAI Responses completion response: {response}",
                        response = serde_json::to_string_pretty(&response)?
                    );
                }
                response.try_into()
            } else {
                let status = response.status();
                let text = http_client::text(response).await?;
                Err(CompletionError::from_http_response(status, text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        GenericResponsesCompletionModel::stream(self, request).await
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        // Extract the msg_ ID from the first Output::Message item
        let message_id = response.output.iter().find_map(|item| match item {
            Output::Message(msg) => Some(msg.id.clone()),
            _ => None,
        });

        let output_content: Vec<completion::AssistantContent> = response
            .output
            .iter()
            .cloned()
            .flat_map(<Vec<completion::AssistantContent>>::from)
            .collect();
        let has_structured_reasoning = response
            .output
            .iter()
            .any(|item| matches!(item, Output::Reasoning { .. }));
        let content = response
            .provider_reasoning
            .as_ref()
            .filter(|reasoning| !has_structured_reasoning && !reasoning.is_empty())
            .map(|reasoning| {
                let mut content = Vec::with_capacity(output_content.len() + 1);
                content.push(completion::AssistantContent::Reasoning(
                    message::Reasoning::new(reasoning),
                ));
                content.extend(output_content.clone());
                content
            })
            .unwrap_or(output_content);

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(GetTokenUsage::token_usage)
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id,
        })
    }
}

/// An OpenAI Responses API message.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        content: OneOrMany<AssistantContentType>,
        #[serde(skip_serializing_if = "String::is_empty")]
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        status: ToolStatus,
    },
    #[serde(rename = "assistant", skip_deserializing)]
    AssistantInput {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        output: ToolResultOutput,
    },
}

/// The type of a tool result content item.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultContentType {
    #[default]
    Text,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

/// Text assistant content.
/// Note that the text type in comparison to the Completions API is actually `output_text` rather than `text`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    OutputText(Text),
    Refusal { refusal: String },
}

impl From<AssistantContent> for completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        match value {
            AssistantContent::Refusal { refusal } => {
                completion::AssistantContent::Text(Text::new(refusal))
            }
            AssistantContent::OutputText(Text { text, .. }) => {
                completion::AssistantContent::Text(Text::new(text))
            }
        }
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
    Audio {
        input_audio: InputAudio,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        output: String,
    },
}

fn flush_responses_user_content(
    messages: &mut Vec<Message>,
    pending: &mut Vec<UserContent>,
) -> Result<(), MessageError> {
    if pending.is_empty() {
        return Ok(());
    }

    let content = OneOrMany::many(std::mem::take(pending)).map_err(|_| {
        MessageError::ConversionError(
            "User message did not contain OpenAI Responses-compatible content".to_string(),
        )
    })?;
    messages.push(Message::User {
        content,
        name: None,
    });
    Ok(())
}

fn responses_user_content(content: message::UserContent) -> Result<UserContent, MessageError> {
    match content {
        message::UserContent::Text(message::Text { text, .. }) => {
            Ok(UserContent::InputText { text })
        }
        message::UserContent::Image(message::Image {
            data,
            detail,
            media_type,
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
                DocumentSourceKind::Raw(_) => {
                    return Err(MessageError::ConversionError(
                        "Raw files not supported, encode as base64 first".into(),
                    ));
                }
                doc => {
                    return Err(MessageError::ConversionError(format!(
                        "Unsupported document type: {doc}"
                    )));
                }
            };

            Ok(UserContent::InputImage {
                image_url: url,
                detail: detail.unwrap_or_default(),
            })
        }
        message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId(file_id),
            ..
        }) => Ok(UserContent::InputFile {
            file_id: Some(file_id),
            file_url: None,
            file_data: None,
            filename: None,
        }),
        message::UserContent::Document(message::Document {
            media_type: Some(DocumentMediaType::PDF),
            data,
            ..
        }) => {
            let (file_data, file_url, filename) = match data {
                DocumentSourceKind::Base64(data) => (
                    Some(format!("data:application/pdf;base64,{data}")),
                    None,
                    Some("document.pdf".to_string()),
                ),
                DocumentSourceKind::Url(url) => (None, Some(url), None),
                DocumentSourceKind::Raw(_) => {
                    return Err(MessageError::ConversionError(
                        "Raw files not supported, encode as base64 first".into(),
                    ));
                }
                doc => {
                    return Err(MessageError::ConversionError(format!(
                        "Unsupported document type: {doc}"
                    )));
                }
            };

            Ok(UserContent::InputFile {
                file_id: None,
                file_url,
                file_data,
                filename,
            })
        }
        message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Base64(text),
            ..
        }) => Ok(UserContent::InputText { text }),
        message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Base64(data),
            media_type,
            ..
        }) => Ok(UserContent::Audio {
            input_audio: InputAudio {
                data,
                format: media_type.unwrap_or(AudioMediaType::MP3),
            },
        }),
        message::UserContent::Audio(_) => Err(MessageError::ConversionError(
            "Audio must be base64 encoded data".into(),
        )),
        _ => Err(MessageError::ConversionError(
            "Unsupported user content for OpenAI Responses API".into(),
        )),
    }
}

fn responses_tool_result(tool_result: message::ToolResult) -> Result<Message, MessageError> {
    let output = responses_tool_result_output(tool_result.content)?;

    Ok(Message::ToolResult {
        tool_call_id: tool_result.call_id.ok_or_else(|| {
            MessageError::ConversionError(
                "Tool result `call_id` is required for OpenAI Responses API".into(),
            )
        })?,
        output,
    })
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::System {
                content: OneOrMany::one(content.into()),
                name: None,
            }]),
            message::Message::User { content } => {
                let mut messages = Vec::new();
                let mut pending = Vec::new();

                for content in content {
                    match content {
                        message::UserContent::ToolResult(tool_result) => {
                            flush_responses_user_content(&mut messages, &mut pending)?;
                            messages.push(responses_tool_result(tool_result)?);
                        }
                        content => pending.push(responses_user_content(content)?),
                    }
                }

                flush_responses_user_content(&mut messages, &mut pending)?;
                Ok(messages)
            }
            message::Message::Assistant {
                content,
                id: assistant_message_id,
            } => {
                let mut messages = Vec::new();

                for assistant_content in content {
                    match assistant_content {
                        crate::message::AssistantContent::Text(Text { text, .. }) => {
                            if text.is_empty() {
                                continue;
                            }
                            if let Some(id) = assistant_message_id.clone() {
                                messages.push(Message::Assistant {
                                    id,
                                    status: ToolStatus::Completed,
                                    content: OneOrMany::one(AssistantContentType::Text(
                                        AssistantContent::OutputText(Text::new(text)),
                                    )),
                                    name: None,
                                });
                            } else {
                                messages.push(Message::AssistantInput {
                                    content: text,
                                    name: None,
                                });
                            }
                        }
                        crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
                            id: tool_id,
                            call_id,
                            function,
                            ..
                        }) => {
                            messages.push(Message::Assistant {
                                content: OneOrMany::one(AssistantContentType::ToolCall(
                                    OutputFunctionCall {
                                        call_id: call_id.ok_or_else(|| {
                                            MessageError::ConversionError(
                                                "Tool call `call_id` is required for OpenAI Responses API"
                                                    .into(),
                                            )
                                        })?,
                                        arguments: function.arguments,
                                        id: tool_id,
                                        name: function.name,
                                        status: ToolStatus::Completed,
                                    },
                                )),
                                id: assistant_message_id.clone().unwrap_or_default(),
                                name: None,
                                status: ToolStatus::Completed,
                            });
                        }
                        crate::message::AssistantContent::Reasoning(reasoning) => {
                            if let Some(openai_reasoning) = openai_reasoning_from_core(&reasoning)?
                            {
                                messages.push(Message::Assistant {
                                    content: OneOrMany::one(AssistantContentType::Reasoning(
                                        openai_reasoning,
                                    )),
                                    id: assistant_message_id.clone().unwrap_or_default(),
                                    name: None,
                                    status: ToolStatus::Completed,
                                });
                            }
                        }
                        crate::message::AssistantContent::Image(_) => {
                            return Err(MessageError::ConversionError(
                                "Assistant image content is not supported in OpenAI Responses API"
                                    .into(),
                            ));
                        }
                    }
                }

                Ok(messages)
            }
        }
    }
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
mod tests {
    use super::*;
    use crate::completion::CompletionRequestBuilder;
    use crate::message;
    use crate::test_utils::MockCompletionModel;
    use serde_json::json;
    use std::collections::HashMap;

    fn test_document(id: &str, text: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: HashMap::new(),
        }
    }

    fn weather_tool_definition() -> completion::ToolDefinition {
        completion::ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }),
        }
    }

    fn rig_tool_result(content: message::ToolResultContent) -> message::Message {
        message::Message::User {
            content: OneOrMany::one(message::UserContent::ToolResult(message::ToolResult {
                id: "result-id".to_string(),
                call_id: Some("call-id".to_string()),
                content: OneOrMany::one(content),
            })),
        }
    }

    #[test]
    fn mixed_user_content_preserves_order_around_tool_results() {
        let input = message::Message::User {
            content: OneOrMany::many(vec![
                message::UserContent::text("before"),
                message::UserContent::tool_result_with_call_id(
                    "result-id",
                    "call-id".to_string(),
                    OneOrMany::one(message::ToolResultContent::text("tool output")),
                ),
                message::UserContent::text("after"),
            ])
            .expect("mixed content should be non-empty"),
        };

        let messages = Vec::<Message>::try_from(input).expect("message conversion");

        assert!(matches!(
            messages.as_slice(),
            [
                Message::User { content: before, .. },
                Message::ToolResult { tool_call_id, .. },
                Message::User { content: after, .. },
            ] if matches!(before.first(), UserContent::InputText { text } if text == "before")
                && tool_call_id == "call-id"
                && matches!(after.first(), UserContent::InputText { text } if text == "after")
        ));
    }

    #[test]
    fn tool_result_literal_text_and_structured_json_render_without_reparsing() {
        let cases = [
            (
                message::ToolResultContent::text(r#"{"status":"ok"}"#),
                r#"{"status":"ok"}"#.to_string(),
            ),
            (
                message::ToolResultContent::json(json!({ "status": "ok" })),
                r#"{"status":"ok"}"#.to_string(),
            ),
        ];

        for (content, expected) in cases {
            let input = rig_tool_result(content);

            let messages: Vec<Message> = input.clone().try_into().expect("message conversion");
            assert!(matches!(
                messages.as_slice(),
                [Message::ToolResult {
                    output: ToolResultOutput::Text(output),
                    ..
                }] if output == &expected
            ));

            let items: Vec<InputItem> = input.try_into().expect("input item conversion");
            assert!(matches!(
                items.as_slice(),
                [InputItem {
                    input: InputContent::FunctionCallOutput(ToolResult {
                        output: ToolResultOutput::Text(output),
                        ..
                    }),
                    ..
                }] if output == &expected
            ));
        }
    }

    #[test]
    fn tool_result_images_and_text_preserve_order_as_rich_function_output() {
        let content = OneOrMany::many(vec![
            message::ToolResultContent::text("before"),
            message::ToolResultContent::image_base64(
                "aW1hZ2U=",
                Some(message::ImageMediaType::PNG),
                None,
            ),
            message::ToolResultContent::json(json!({ "after": true })),
        ])
        .expect("mixed tool output is non-empty");
        let input = message::Message::User {
            content: OneOrMany::one(message::UserContent::ToolResult(message::ToolResult {
                id: "result-id".to_string(),
                call_id: Some("call-id".to_string()),
                content,
            })),
        };

        let assert_output = |output: &ToolResultOutput| {
            assert!(matches!(
                output,
                ToolResultOutput::Content(content)
                    if matches!(content.as_slice(), [
                        ToolResultOutputContent::InputText { text: before },
                        ToolResultOutputContent::InputImage { image_url, .. },
                        ToolResultOutputContent::InputText { text: after },
                    ] if before == "before"
                        && image_url.as_deref() == Some("data:image/png;base64,aW1hZ2U=")
                        && after == r#"{"after":true}"#)
            ));
        };

        let messages: Vec<Message> = input.clone().try_into().expect("message conversion");
        match messages.as_slice() {
            [Message::ToolResult { output, .. }] => assert_output(output),
            other => panic!("expected one rich tool result, got {other:?}"),
        }

        let items: Vec<InputItem> = input.try_into().expect("input item conversion");
        match items.as_slice() {
            [
                InputItem {
                    input: InputContent::FunctionCallOutput(ToolResult { output, .. }),
                    ..
                },
            ] => assert_output(output),
            other => panic!("expected one rich function output, got {other:?}"),
        }
    }

    #[test]
    fn tool_result_file_id_image_uses_the_native_wire_field() {
        let input = rig_tool_result(message::ToolResultContent::Image(message::Image {
            data: message::DocumentSourceKind::FileId("file-image-123".to_string()),
            media_type: None,
            detail: None,
            additional_params: None,
        }));

        let items: Vec<InputItem> = input.try_into().expect("input item conversion");
        let wire = serde_json::to_value(&items[0]).expect("serialize input item");
        assert_eq!(
            wire,
            json!({
                "type": "function_call_output",
                "call_id": "call-id",
                "output": [{
                    "type": "input_image",
                    "file_id": "file-image-123",
                    "detail": "auto"
                }],
                "status": "completed"
            })
        );
    }

    fn weather_tool_request() -> completion::CompletionRequest {
        completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one(message::Message::user("what's the weather?")),
            documents: Vec::new(),
            tools: vec![weather_tool_definition()],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn responses_tool_choice_modes_serialize_as_plain_strings() {
        for (choice, expected) in [
            (message::ToolChoice::Auto, json!("auto")),
            (message::ToolChoice::None, json!("none")),
            (message::ToolChoice::Required, json!("required")),
        ] {
            let converted = ToolChoice::try_from(choice).expect("mode should convert");
            assert_eq!(
                serde_json::to_value(&converted).expect("serialize tool choice"),
                expected
            );
        }
    }

    #[test]
    fn responses_tool_choice_specific_single_name_serializes_as_named_function() {
        let converted = ToolChoice::try_from(message::ToolChoice::Specific {
            function_names: vec!["get_weather".to_string()],
        })
        .expect("single specific tool should convert");

        assert_eq!(
            serde_json::to_value(&converted).expect("serialize tool choice"),
            json!({"type": "function", "name": "get_weather"})
        );
    }

    #[test]
    fn responses_tool_choice_specific_multiple_names_serialize_as_allowed_tools() {
        let converted = ToolChoice::try_from(message::ToolChoice::Specific {
            function_names: vec!["add".to_string(), "subtract".to_string()],
        })
        .expect("multiple specific tools should convert");

        assert_eq!(
            serde_json::to_value(&converted).expect("serialize tool choice"),
            json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "add"},
                    {"type": "function", "name": "subtract"}
                ]
            })
        );
    }

    #[test]
    fn responses_tool_choice_specific_empty_names_error() {
        let converted = ToolChoice::try_from(message::ToolChoice::Specific {
            function_names: vec![],
        });

        assert!(matches!(
            converted,
            Err(CompletionError::RequestError(error))
                if error.to_string().contains("at least one function name")
        ));
    }

    #[test]
    fn responses_request_with_specific_tool_choice_serializes_named_function() {
        let mut request = weather_tool_request();
        request.tool_choice = Some(message::ToolChoice::Specific {
            function_names: vec!["get_weather".to_string()],
        });

        let request =
            CompletionRequest::try_from(("gpt-test".to_string(), request)).expect("convert");
        let request_json = serde_json::to_value(&request).expect("serialize request");

        assert_eq!(
            request_json.get("tool_choice"),
            Some(&json!({"type": "function", "name": "get_weather"}))
        );
    }

    #[test]
    fn responses_function_tools_are_non_strict_by_default() {
        let tool = ResponsesToolDefinition::function(
            "get_weather",
            "Get the weather",
            weather_tool_definition().parameters,
        );

        assert!(!tool.strict);
        assert_eq!(tool.parameters["required"], json!(["location"]));
        assert!(tool.parameters.get("additionalProperties").is_none());

        let serialized = serde_json::to_value(tool).expect("tool should serialize");
        assert!(serialized.get("strict").is_none());
    }

    #[test]
    fn responses_tool_definitions_accept_nullable_strict() {
        let cases = [
            (
                json!({
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {}
                }),
                false,
            ),
            (
                json!({
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {},
                    "strict": null
                }),
                false,
            ),
            (
                json!({
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {},
                    "strict": false
                }),
                false,
            ),
            (
                json!({
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {},
                    "strict": true
                }),
                true,
            ),
        ];

        for (value, expected) in cases {
            let tool: ResponsesToolDefinition =
                serde_json::from_value(value).expect("tool definition should deserialize");
            assert_eq!(tool.strict, expected);
        }
    }

    #[test]
    fn responses_strict_function_tools_sanitize_schema() {
        let tool = ResponsesToolDefinition::strict_function(
            "get_weather",
            "Get the weather",
            weather_tool_definition().parameters,
        );

        assert!(tool.strict);
        assert_eq!(tool.parameters["additionalProperties"], json!(false));
        assert_eq!(tool.parameters["required"], json!(["location", "unit"]));
    }

    fn request_with_preamble(preamble: &str) -> completion::CompletionRequest {
        completion::CompletionRequest {
            model: None,
            preamble: Some(preamble.to_string()),
            chat_history: crate::OneOrMany::one(message::Message::user("Hello")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    fn system_only_request(system_text: &str) -> completion::CompletionRequest {
        completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one(completion::Message::system(system_text)),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn responses_request_uses_top_level_instructions_for_preamble_by_default() {
        let req = CompletionRequest::try_from((
            "gpt-4o-mini".to_string(),
            request_with_preamble("You are concise."),
        ))
        .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert_eq!(serialized["instructions"], json!("You are concise."));
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn responses_request_drops_whitespace_only_preamble() {
        let req = CompletionRequest::try_from((
            "gpt-4o-mini".to_string(),
            request_with_preamble("  \n "),
        ))
        .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert!(
            serialized.get("instructions").is_none(),
            "a whitespace-only preamble carries no content and is dropped"
        );
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn responses_request_lifts_system_messages_to_top_level_instructions_by_default() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Hello")
            .preamble("System one".to_string())
            .message(completion::Message::system("System two"))
            .build();

        let req = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
            .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert_eq!(
            serialized["instructions"],
            json!("System one\n\nSystem two")
        );
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn responses_request_with_only_system_messages_keeps_them_in_input() {
        let req = CompletionRequest::try_from((
            "gpt-4o-mini".to_string(),
            system_only_request("System only"),
        ))
        .expect("request conversion should succeed");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert!(
            serialized.get("instructions").is_none(),
            "lifting a system-only history would leave input empty, so it stays in input"
        );
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "system");
        assert!(input[0].to_string().contains("System only"));
    }

    #[test]
    fn responses_model_can_fallback_to_system_messages_in_input() {
        let client = crate::providers::openai::Client::new("dummy-key").expect("client");
        let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
            .with_system_instructions_as_messages();

        let req = model
            .create_completion_request(request_with_preamble("You are concise."))
            .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert!(serialized.get("instructions").is_none());
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "system");
        assert!(input[0].to_string().contains("You are concise."));
        assert_eq!(input[1]["role"], "user");
    }

    #[test]
    fn responses_client_can_fallback_to_system_messages_in_input() {
        use crate::prelude::CompletionClient;

        let client = crate::providers::openai::Client::new("dummy-key")
            .expect("client")
            .with_system_instructions_as_messages();
        let model = client.completion_model("gpt-4o-mini");

        let req = model
            .create_completion_request(request_with_preamble("You are concise."))
            .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert!(serialized.get("instructions").is_none());
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "system");
        assert!(input[0].to_string().contains("You are concise."));
        assert_eq!(input[1]["role"], "user");
    }

    #[test]
    fn responses_model_can_lift_all_system_messages_via_placement() {
        let client = crate::providers::openai::Client::new("dummy-key").expect("client");
        let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
            .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions);

        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "again")
            .preamble("System one".to_string())
            .message(completion::Message::user("hi"))
            .message(completion::Message::system("Mid-conversation instruction"))
            .build();

        let req = model
            .create_completion_request(request)
            .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be array");

        assert_eq!(
            serialized["instructions"],
            json!("System one\n\nMid-conversation instruction")
        );
        assert!(
            input.iter().all(|item| item["role"] != "system"),
            "AllInstructions should leave no system items in input: {input:?}"
        );
    }

    #[test]
    fn responses_client_placement_survives_completions_api_round_trip() {
        use crate::prelude::CompletionClient;

        let client = crate::providers::openai::Client::new("dummy-key")
            .expect("client")
            .with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
            .completions_api()
            .responses_api();
        let model = client.completion_model("gpt-4o-mini");

        let req = model
            .create_completion_request(request_with_preamble("You are concise."))
            .expect("request should convert");
        let serialized = serde_json::to_value(&req).expect("request should serialize");

        assert!(
            serialized.get("instructions").is_none(),
            "placement configured before completions_api() should survive responses_api()"
        );
        assert_eq!(serialized["input"][0]["role"], "system");
    }

    #[test]
    fn all_instructions_system_only_input_reports_non_system_requirement() {
        let err = CompletionRequest::try_from(ResponsesRequestParams {
            model: "gpt-4o-mini".to_string(),
            request: system_only_request("System only"),
            system_instructions_placement: SystemInstructionsPlacement::AllInstructions,
        })
        .expect_err("system-only input should fail once every item is lifted");

        assert!(
            err.to_string().contains("non-system item"),
            "error should explain that lifted system messages left input empty: {err}"
        );
    }

    #[test]
    fn all_instructions_whitespace_only_system_input_reports_non_system_requirement() {
        let err = CompletionRequest::try_from(ResponsesRequestParams {
            model: "gpt-4o-mini".to_string(),
            request: system_only_request("   "),
            system_instructions_placement: SystemInstructionsPlacement::AllInstructions,
        })
        .expect_err("whitespace-only system input should fail once every item is lifted");

        assert!(
            err.to_string().contains("non-system item"),
            "even when lifted system text is whitespace-only (so no `instructions` field is \
             produced), the error should explain that system messages were lifted: {err}"
        );
    }

    #[test]
    fn responses_request_conversion_keeps_tools_non_strict_by_default() {
        let req = CompletionRequest::try_from(("gpt-4o-mini".to_string(), weather_tool_request()))
            .expect("request should convert");

        let tool = &req.tools[0];
        assert!(!tool.strict);
        assert_eq!(tool.parameters["required"], json!(["location"]));
        assert!(tool.parameters.get("additionalProperties").is_none());
    }

    #[test]
    fn responses_model_strict_tools_opt_in_sanitizes_all_function_tools() {
        let client = crate::providers::openai::Client::new("dummy-key").expect("client");
        let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
            .with_strict_tools()
            .with_tool(completion::ToolDefinition {
                name: "lookup".to_string(),
                description: "Look something up".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }),
            });

        let mut request = weather_tool_request();
        request.additional_params = Some(json!({
            "tools": [{
                "type": "function",
                "name": "extra",
                "description": "An additional_params tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
            }]
        }));

        let req = model
            .create_completion_request(request)
            .expect("request should convert");

        assert_eq!(req.tools.len(), 3);
        for tool in &req.tools {
            assert!(tool.strict, "{} should be strict", tool.name);
            assert_eq!(tool.parameters["additionalProperties"], json!(false));
        }
    }

    #[test]
    fn responses_model_default_preserves_all_function_tools_as_constructed() {
        let client = crate::providers::openai::Client::new("dummy-key").expect("client");
        let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
            .with_tool(weather_tool_definition());

        let mut request = weather_tool_request();
        request.additional_params = Some(json!({
            "tools": [{
                "type": "function",
                "name": "extra",
                "description": "An additional_params tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
            }]
        }));

        let req = model
            .create_completion_request(request)
            .expect("request should convert");

        assert_eq!(req.tools.len(), 3);
        for tool in &req.tools {
            assert!(!tool.strict, "{} should not be strict", tool.name);
            assert!(tool.parameters.get("additionalProperties").is_none());
        }
    }

    #[test]
    fn responses_explicit_strict_tool_stays_strict_on_default_model() {
        let client = crate::providers::openai::Client::new("dummy-key").expect("client");
        let model = ResponsesCompletionModel::new(client, "gpt-4o-mini").with_tool(
            ResponsesToolDefinition::strict_function(
                "lookup",
                "Look something up",
                json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            ),
        );

        let req = model
            .create_completion_request(weather_tool_request())
            .expect("request should convert");

        assert!(!req.tools[0].strict);
        assert!(req.tools[1].strict);
        assert_eq!(
            req.tools[1].parameters["additionalProperties"],
            json!(false)
        );
    }

    fn response_with_service_tier(service_tier: &str) -> Value {
        json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
            "service_tier": service_tier,
        })
    }

    #[test]
    fn completion_response_deserializes_float_created_at() {
        // Bedrock Mantle GPT-OSS Responses returns created_at as a float.
        let raw = json!({
            "id": "resp_mantle_float",
            "object": "response",
            "created_at": 1784497980.0,
            "status": "completed",
            "model": "openai.gpt-oss-20b",
            "output": [],
        });

        let response: CompletionResponse =
            serde_json::from_value(raw).expect("float created_at should deserialize");
        assert_eq!(response.created_at, 1_784_497_980);
    }

    #[test]
    fn completion_response_deserializes_integer_created_at() {
        let raw = json!({
            "id": "resp_int",
            "object": "response",
            "created_at": 1_700_000_000_u64,
            "status": "completed",
            "model": "gpt-4o",
            "output": [],
        });

        let response: CompletionResponse =
            serde_json::from_value(raw).expect("integer created_at should deserialize");
        assert_eq!(response.created_at, 1_700_000_000);
    }

    #[test]
    fn completion_response_deserializes_standard_service_tier() {
        let response: CompletionResponse =
            serde_json::from_value(response_with_service_tier("standard"))
                .expect("response should deserialize");

        assert!(matches!(
            response.additional_parameters.service_tier,
            Some(OpenAIServiceTier::Standard)
        ));
    }

    #[test]
    fn completion_response_deserializes_priority_service_tier() {
        let response: CompletionResponse =
            serde_json::from_value(response_with_service_tier("priority"))
                .expect("response should deserialize");

        assert!(matches!(
            response.additional_parameters.service_tier,
            Some(OpenAIServiceTier::Priority)
        ));
    }

    #[test]
    fn completion_response_preserves_unknown_service_tier() {
        let response: CompletionResponse =
            serde_json::from_value(response_with_service_tier("provider_experimental"))
                .expect("response should deserialize");

        let Some(OpenAIServiceTier::Other(service_tier)) =
            response.additional_parameters.service_tier
        else {
            panic!("expected provider-specific service tier");
        };

        assert_eq!(service_tier, "provider_experimental");
    }

    #[test]
    fn responses_request_keeps_documents_after_lifted_system_messages() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Prompt")
            .message(completion::Message::system("System prompt"))
            .message(completion::Message::user("Earlier user turn"))
            .message(completion::Message::assistant("Earlier assistant turn"))
            .document(test_document("doc1", "Document text."))
            .build();

        let responses_request = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
            .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(&responses_request).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be an array");

        assert_eq!(serialized["instructions"], json!("System prompt"));
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert!(
            input[0].to_string().contains("<file id: doc1>"),
            "document input should be first after system instructions are lifted: {input:?}"
        );
        assert_eq!(input[1]["role"], "user");
        assert!(
            input[1].to_string().contains("Earlier user turn"),
            "prior user history should follow document input: {input:?}"
        );
        assert_eq!(input[2]["role"], "assistant");
        assert!(
            input[2].to_string().contains("Earlier assistant turn"),
            "prior assistant history should follow prior user history: {input:?}"
        );
        assert_eq!(input[3]["role"], "user");
        assert!(
            input[3].to_string().contains("Prompt"),
            "prompt should remain last: {input:?}"
        );
    }

    #[test]
    fn responses_direct_request_keeps_mid_conversation_system_messages_in_input() {
        let request = crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::many(vec![
                completion::Message::system("System prompt"),
                completion::Message::assistant("Earlier assistant turn"),
                completion::Message::system("Mid-conversation instruction"),
                completion::Message::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![test_document("doc1", "Document text.")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let responses_request = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
            .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(&responses_request).expect("request should serialize");
        let input = serialized["input"]
            .as_array()
            .expect("input should be an array");

        assert_eq!(
            serialized["instructions"],
            json!("System prompt"),
            "only the leading run of system messages should be lifted"
        );
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert!(
            input[0].to_string().contains("<file id: doc1>"),
            "document input should follow lifted system instructions: {input:?}"
        );
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["role"], "system");
        assert!(
            input[2]
                .to_string()
                .contains("Mid-conversation instruction"),
            "mid-conversation system messages should keep their position: {input:?}"
        );
        assert_eq!(input[3]["role"], "user");
        assert_eq!(
            input
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc1>"))
                .count(),
            1,
            "document input should appear exactly once: {input:?}"
        );
    }

    #[test]
    fn service_tier_serializes_expected_strings() {
        let cases = [
            (OpenAIServiceTier::Auto, "auto"),
            (OpenAIServiceTier::Default, "default"),
            (OpenAIServiceTier::Flex, "flex"),
            (OpenAIServiceTier::Priority, "priority"),
            (OpenAIServiceTier::Standard, "standard"),
        ];

        for (service_tier, expected) in cases {
            assert_eq!(
                serde_json::to_value(service_tier).expect("service tier should serialize"),
                json!(expected)
            );
        }

        assert_eq!(
            serde_json::to_value(OpenAIServiceTier::Other(
                "provider_experimental".to_string()
            ))
            .expect("provider-specific service tier should serialize"),
            json!("provider_experimental")
        );
    }

    #[test]
    fn responses_usage_token_usage_preserves_reasoning_tokens() {
        let usage = ResponsesUsage {
            input_tokens: 100,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 25 }),
            output_tokens: 50,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 15,
            }),
            total_tokens: 150,
        };

        let token_usage = usage.token_usage();

        assert_eq!(token_usage.input_tokens, 100);
        assert_eq!(token_usage.cached_input_tokens, 25);
        assert_eq!(token_usage.output_tokens, 50);
        assert_eq!(token_usage.reasoning_tokens, 15);
        assert_eq!(token_usage.total_tokens, 150);
    }

    #[test]
    fn responses_usage_deserializes_without_output_token_details() {
        let usage: ResponsesUsage = serde_json::from_value(json!({
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 25
            },
            "output_tokens": 50,
            "total_tokens": 150
        }))
        .expect("usage should deserialize when output token details are omitted");

        assert!(usage.output_tokens_details.is_none());

        let token_usage = usage.token_usage();

        assert_eq!(token_usage.input_tokens, 100);
        assert_eq!(token_usage.cached_input_tokens, 25);
        assert_eq!(token_usage.output_tokens, 50);
        assert_eq!(token_usage.reasoning_tokens, 0);
        assert_eq!(token_usage.total_tokens, 150);
    }

    #[test]
    fn completion_response_accepts_top_level_reasoning_string() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "Qwen/Qwen3-4B",
            "reasoning": "thinking through the answer",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3
            },
            "output": [{
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "text": "done"
                }]
            }],
            "tools": []
        }))
        .expect("mistral.rs-style reasoning string should deserialize");

        assert_eq!(
            response.provider_reasoning.as_deref(),
            Some("thinking through the answer")
        );
        assert_eq!(response.reasoning_metadata, None);
        assert_eq!(response.reasoning_context, None);
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            json!("thinking through the answer")
        );

        let completion: completion::CompletionResponse<CompletionResponse> =
            response.try_into().expect("response should convert");
        let items = completion.choice.iter().collect::<Vec<_>>();
        assert!(matches!(
            items[0],
            completion::AssistantContent::Reasoning(_)
        ));
        assert!(matches!(items[1], completion::AssistantContent::Text(_)));
    }

    #[test]
    fn completion_response_accepts_null_metadata() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "openai-compatible-model",
            "metadata": null,
            "output": [{
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "text": "done"
                }]
            }],
            "tools": []
        }))
        .expect("response with null metadata should deserialize");

        assert!(response.additional_parameters.metadata.is_empty());
    }

    #[test]
    fn completion_response_accepts_reasoning_only_response() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "Qwen/Qwen3-4B",
            "reasoning": "thinking only",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3
            },
            "output": [],
            "tools": []
        }))
        .expect("reasoning-only response should deserialize");

        let completion: completion::CompletionResponse<CompletionResponse> = response
            .try_into()
            .expect("reasoning-only response should convert");
        let items = completion.choice.iter().collect::<Vec<_>>();

        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            completion::AssistantContent::Reasoning(_)
        ));
    }

    #[test]
    fn completion_response_rejects_empty_response_without_reasoning() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "Qwen/Qwen3-4B",
            "output": [],
            "tools": []
        }))
        .expect("empty response shape should deserialize");

        let err = completion::CompletionResponse::<CompletionResponse>::try_from(response)
            .expect_err("empty response without reasoning should be rejected");

        assert!(
            err.to_string()
                .contains("Response contained no message or tool call")
        );
    }

    #[test]
    fn completion_response_preserves_context_without_treating_config_as_text() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "Qwen/Qwen3-4B",
            "reasoning": {
                "context": "all_turns",
                "effort": "high",
                "mode": "standard",
                "summary": null
            },
            "output": [{
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "text": "done"
                }]
            }],
            "tools": []
        }))
        .expect("object-shaped reasoning should be tolerated");

        assert!(response.provider_reasoning.is_none());
        assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
        assert_eq!(
            response.reasoning_metadata.as_ref(),
            json!({
                "context": "all_turns",
                "effort": "high",
                "mode": "standard",
                "summary": null
            })
            .as_object()
        );
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            json!({
                "context": "all_turns",
                "effort": "high",
                "mode": "standard",
                "summary": null
            })
        );

        let completion: completion::CompletionResponse<CompletionResponse> =
            response.try_into().expect("response should convert");
        let items = completion.choice.iter().collect::<Vec<_>>();
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], completion::AssistantContent::Text(_)));
    }

    #[test]
    fn completion_response_preserves_unknown_reasoning_metadata_and_nulls() {
        let metadata = json!({
            "context": "future_context",
            "effort": "ultra",
            "summary": null,
            "future_control": { "depth": 3 }
        });
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-future",
            "reasoning": metadata.clone(),
            "output": [],
            "tools": []
        }))
        .expect("unknown reasoning metadata should deserialize");

        assert_eq!(
            response.reasoning_context.as_deref(),
            Some("future_context")
        );
        assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            metadata
        );
    }

    #[test]
    fn completion_response_ignores_unsupported_reasoning_shapes() {
        for reasoning in [Value::Null, json!(["unexpected"]), json!(42), json!(true)] {
            let response: CompletionResponse = serde_json::from_value(json!({
                "id": "resp_123",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "openai-compatible-model",
                "reasoning": reasoning,
                "output": [],
                "tools": []
            }))
            .expect("unsupported reasoning shapes should remain non-fatal");

            assert_eq!(response.provider_reasoning, None);
            assert_eq!(response.reasoning_metadata, None);
            assert_eq!(response.reasoning_context, None);
            let serialized = serde_json::to_value(&response).expect("response should serialize");
            assert!(
                !serialized
                    .as_object()
                    .expect("response should serialize as an object")
                    .contains_key("reasoning")
            );
        }
    }

    #[test]
    fn completion_response_reasoning_serialization_precedence_is_stable() {
        let mut response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6",
            "reasoning": { "context": "all_turns", "effort": "max" },
            "output": [],
            "tools": []
        }))
        .expect("reasoning metadata should deserialize");

        response.reasoning_context = Some("current_turn".to_owned());
        response.additional_parameters.reasoning =
            Some(Reasoning::new().with_effort(ReasoningEffort::Low));
        let serialized = serde_json::to_string(&response).expect("response should serialize");
        assert_eq!(serialized.matches("\"reasoning\":").count(), 1);
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            json!({ "context": "all_turns", "effort": "max" })
        );

        let metadata = response.reasoning_metadata.take();
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            json!({ "context": "current_turn" })
        );

        response.reasoning_metadata = metadata;
        response.provider_reasoning = Some("compatible-provider text".to_owned());
        assert_eq!(
            serde_json::to_value(&response).expect("response should serialize")["reasoning"],
            json!("compatible-provider text")
        );
    }

    fn request_with_reasoning_params(reasoning: Value) -> CompletionRequest {
        let mut request = request_with_preamble("You are concise.");
        request.additional_params = Some(json!({ "reasoning": reasoning }));

        CompletionRequest::try_from(("gpt-5.6".to_string(), request))
            .expect("request with reasoning params should convert")
    }

    #[test]
    fn reasoning_effort_max_survives_request_conversion() {
        let request = request_with_reasoning_params(json!({ "effort": "max" }));
        let serialized = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(serialized["reasoning"], json!({ "effort": "max" }));
    }

    #[test]
    fn reasoning_mode_pro_composes_with_independent_effort() {
        let request = request_with_reasoning_params(json!({ "effort": "high", "mode": "pro" }));
        let serialized = serde_json::to_value(&request).expect("request should serialize");

        assert_eq!(
            serialized["reasoning"],
            json!({ "effort": "high", "mode": "pro" })
        );
    }

    #[test]
    fn reasoning_context_values_survive_request_conversion() {
        for (context, wire_value) in [
            (ReasoningContext::Auto, "auto"),
            (ReasoningContext::AllTurns, "all_turns"),
            (ReasoningContext::CurrentTurn, "current_turn"),
        ] {
            let typed = serde_json::to_value(Reasoning::new().with_context(context))
                .expect("typed reasoning should serialize");
            assert_eq!(typed, json!({ "context": wire_value }));

            let request = request_with_reasoning_params(json!({ "context": wire_value }));
            let serialized = serde_json::to_value(&request).expect("request should serialize");
            assert_eq!(serialized["reasoning"], json!({ "context": wire_value }));
        }
    }

    #[test]
    fn reasoning_omits_unset_optional_fields() {
        let reasoning = serde_json::to_value(Reasoning::new().with_mode(ReasoningMode::Pro))
            .expect("reasoning should serialize");

        assert_eq!(reasoning, json!({ "mode": "pro" }));

        let reasoning = serde_json::to_value(
            Reasoning::new()
                .with_effort(ReasoningEffort::Max)
                .with_mode(ReasoningMode::Pro)
                .with_context(ReasoningContext::CurrentTurn)
                .with_summary_level(ReasoningSummaryLevel::Detailed),
        )
        .expect("reasoning should serialize");

        assert_eq!(
            reasoning,
            json!({
                "effort": "max",
                "mode": "pro",
                "context": "current_turn",
                "summary": "detailed"
            })
        );
    }

    #[test]
    fn completion_response_does_not_duplicate_structured_reasoning() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.4",
            "reasoning": "provider top-level text",
            "output": [{
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{
                    "type": "summary_text",
                    "text": "structured summary"
                }]
            }, {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "text": "done"
                }]
            }],
            "tools": []
        }))
        .expect("response should deserialize");

        let completion: completion::CompletionResponse<CompletionResponse> =
            response.try_into().expect("response should convert");
        let reasoning_count = completion
            .choice
            .iter()
            .filter(|item| matches!(item, completion::AssistantContent::Reasoning(_)))
            .count();

        assert_eq!(reasoning_count, 1);
    }

    #[test]
    fn idless_reasoning_is_skipped_when_converting_responses_history() {
        let assistant = message::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::one(message::AssistantContent::Reasoning(
                message::Reasoning::new("provider reasoning"),
            )),
        };

        let converted = Vec::<Message>::try_from(assistant)
            .expect("idless reasoning should degrade gracefully");

        assert!(converted.is_empty());
    }

    #[test]
    fn idless_reasoning_only_is_skipped_without_empty_input_item() {
        let assistant = completion::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::Reasoning(
                message::Reasoning::new("provider reasoning"),
            )),
        };

        let converted = Vec::<InputItem>::try_from(assistant)
            .expect("idless reasoning should degrade gracefully");

        assert!(converted.is_empty());
    }

    #[test]
    fn idless_reasoning_plus_text_preserves_text_for_responses_history() {
        let assistant = message::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::many(vec![
                message::AssistantContent::Reasoning(message::Reasoning::new("provider reasoning")),
                message::AssistantContent::Text(Text::new("final answer")),
            ])
            .expect("assistant content should be non-empty"),
        };

        let converted =
            Vec::<Message>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 1);
        let Message::Assistant { content, .. } = &converted[0] else {
            panic!("expected assistant message");
        };
        assert!(matches!(
            content.first_ref(),
            AssistantContentType::Text(AssistantContent::OutputText(Text { text, .. })) if text == "final answer"
        ));
    }

    #[test]
    fn completion_history_idless_reasoning_plus_text_preserves_text_input_item() {
        let assistant = completion::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::many(vec![
                message::AssistantContent::Reasoning(message::Reasoning::new("provider reasoning")),
                message::AssistantContent::Text(Text::new("final answer")),
            ])
            .expect("assistant content should be non-empty"),
        };

        let converted =
            Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 1);
        assert!(matches!(converted[0].role, Some(Role::Assistant)));
        let InputContent::Message(Message::Assistant { content, .. }) = &converted[0].input else {
            panic!("expected assistant message input item");
        };
        assert!(matches!(
            content.first_ref(),
            AssistantContentType::Text(AssistantContent::OutputText(Text { text, .. })) if text == "final answer"
        ));
    }

    #[test]
    fn assistant_text_without_idless_reasoning_replays_as_output_text() {
        let assistant = completion::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::one(message::AssistantContent::Text(Text::new("final answer"))),
        };

        let converted =
            Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 1);
        let InputContent::Message(Message::Assistant { content, .. }) = &converted[0].input else {
            panic!("expected assistant message input item");
        };
        assert!(matches!(
            content.first_ref(),
            AssistantContentType::Text(AssistantContent::OutputText(Text { text, .. })) if text == "final answer"
        ));
    }

    #[test]
    fn idless_completion_assistant_text_replays_as_easy_input_message() {
        let assistant = completion::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::Text(Text::new("final answer"))),
        };

        let converted =
            Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 1);
        assert!(matches!(converted[0].role, Some(Role::Assistant)));
        let InputContent::Message(Message::AssistantInput { content, .. }) = &converted[0].input
        else {
            panic!("expected assistant input message item");
        };
        assert_eq!(content, "final answer");

        let serialized =
            serde_json::to_value(&converted[0]).expect("input item should serialize to JSON");
        assert_eq!(serialized["type"], json!("message"));
        assert_eq!(serialized["role"], json!("assistant"));
        assert_eq!(serialized["content"], json!("final answer"));
        assert!(serialized.get("id").is_none());
        assert!(serialized.get("status").is_none());
    }

    #[test]
    fn idless_message_assistant_text_replays_as_easy_input_message() {
        let assistant = message::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::Text(Text::new("final answer"))),
        };

        let converted =
            Vec::<Message>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 1);
        let Message::AssistantInput { content, .. } = &converted[0] else {
            panic!("expected assistant input message");
        };
        assert_eq!(content, "final answer");

        let serialized = serde_json::to_value(&converted[0])
            .expect("assistant message should serialize to JSON");
        assert_eq!(serialized["role"], json!("assistant"));
        assert_eq!(serialized["content"], json!("final answer"));
        assert!(serialized.get("id").is_none());
        assert!(serialized.get("status").is_none());
    }

    #[test]
    fn structured_reasoning_with_id_still_converts_for_responses_history() {
        let assistant = message::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::one(message::AssistantContent::Reasoning(message::Reasoning {
                id: Some("rs_123".to_string()),
                content: vec![message::ReasoningContent::Summary(
                    "structured summary".to_string(),
                )],
            })),
        };

        let converted =
            Vec::<Message>::try_from(assistant).expect("structured reasoning should still convert");

        assert_eq!(converted.len(), 1);
        let Message::Assistant { content, .. } = &converted[0] else {
            panic!("expected assistant message");
        };
        assert!(matches!(
            content.first_ref(),
            AssistantContentType::Reasoning(OpenAIReasoning { id, .. }) if id == "rs_123"
        ));
    }

    #[test]
    fn structured_reasoning_with_id_still_converts_to_input_item() {
        let assistant = completion::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::one(message::AssistantContent::Reasoning(message::Reasoning {
                id: Some("rs_123".to_string()),
                content: vec![message::ReasoningContent::Summary(
                    "structured summary".to_string(),
                )],
            })),
        };

        let converted =
            Vec::<InputItem>::try_from(assistant).expect("structured reasoning should convert");

        assert_eq!(converted.len(), 1);
        assert!(converted[0].role.is_none());
        assert!(matches!(
            &converted[0].input,
            InputContent::Reasoning(OpenAIReasoning { id, .. }) if id == "rs_123"
        ));
    }

    #[test]
    fn assistant_reasoning_text_tool_call_convert_in_responses_replay_order() {
        let assistant = completion::Message::Assistant {
            id: Some("msg_123".to_string()),
            content: OneOrMany::many(vec![
                message::AssistantContent::Reasoning(message::Reasoning {
                    id: Some("rs_123".to_string()),
                    content: vec![message::ReasoningContent::Summary(
                        "structured summary".to_string(),
                    )],
                }),
                message::AssistantContent::Text(Text::new("final answer")),
                message::AssistantContent::tool_call_with_call_id(
                    "fc_123",
                    "call_123".to_string(),
                    "lookup",
                    json!({"query": "rig"}),
                ),
            ])
            .expect("assistant content should be non-empty"),
        };

        let converted =
            Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

        assert_eq!(converted.len(), 3);
        assert!(converted[0].role.is_none());
        assert!(matches!(
            &converted[0].input,
            InputContent::Reasoning(OpenAIReasoning { id, .. }) if id == "rs_123"
        ));

        assert!(matches!(converted[1].role, Some(Role::Assistant)));
        let InputContent::Message(Message::Assistant { content, id, .. }) = &converted[1].input
        else {
            panic!("expected assistant output message");
        };
        assert_eq!(id, "msg_123");
        assert!(matches!(
            content.first_ref(),
            AssistantContentType::Text(AssistantContent::OutputText(Text { text, .. }))
                if text == "final answer"
        ));

        assert!(converted[2].role.is_none());
        let InputContent::FunctionCall(OutputFunctionCall {
            id, call_id, name, ..
        }) = &converted[2].input
        else {
            panic!("expected function call input item");
        };
        assert_eq!(id, "fc_123");
        assert_eq!(call_id, "call_123");
        assert_eq!(name, "lookup");
    }

    #[test]
    fn mocked_second_turn_request_omits_unreplayable_reasoning() {
        let request = crate::completion::CompletionRequest {
            model: None,
            preamble: Some("You are concise.".to_string()),
            chat_history: OneOrMany::many(vec![
                completion::Message::User {
                    content: OneOrMany::one(message::UserContent::Text(Text::new(
                        "Think briefly, then answer.",
                    ))),
                },
                completion::Message::Assistant {
                    id: Some("msg_123".to_string()),
                    content: OneOrMany::many(vec![
                        message::AssistantContent::Reasoning(message::Reasoning::new(
                            "provider reasoning",
                        )),
                        message::AssistantContent::Text(Text::new("final answer")),
                    ])
                    .expect("assistant content should be non-empty"),
                },
                completion::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(vec![
                        message::AssistantContent::Reasoning(message::Reasoning::new(
                            "provider reasoning only",
                        )),
                        message::AssistantContent::Text(Text::new("")),
                    ])
                    .expect("assistant content should be non-empty"),
                },
                completion::Message::User {
                    content: OneOrMany::one(message::UserContent::Text(Text::new(
                        "/no_think Reply with exactly: OK",
                    ))),
                },
            ])
            .expect("history should be non-empty"),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let request = CompletionRequest::try_from(("Qwen/Qwen3-4B".to_string(), request))
            .expect("request should convert");
        let value = serde_json::to_value(&request).expect("request should serialize");
        let input = value["input"]
            .as_array()
            .expect("mocked multi-turn request should serialize input as an array");

        assert!(!input.iter().any(|item| {
            item.get("type") == Some(&json!("reasoning")) && item.get("id").is_none()
        }));
        assert!(!input.iter().any(|item| {
            item.get("role") == Some(&json!("assistant"))
                && item
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
        }));

        let assistant_items = input
            .iter()
            .filter(|item| item.get("role") == Some(&json!("assistant")))
            .collect::<Vec<_>>();

        assert_eq!(assistant_items.len(), 1);
        assert_eq!(assistant_items[0]["content"][0]["type"], "output_text");
        assert_eq!(assistant_items[0]["content"][0]["text"], "final answer");
    }

    #[test]
    fn responses_usage_add_preserves_rhs_details_when_lhs_details_are_absent() {
        let lhs = ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 20,
            output_tokens_details: None,
            total_tokens: 30,
        };
        let rhs = ResponsesUsage {
            input_tokens: 3,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 2 }),
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 4,
            }),
            total_tokens: 8,
        };

        let usage = lhs + rhs;
        let token_usage = usage.token_usage();

        assert_eq!(token_usage.input_tokens, 13);
        assert_eq!(token_usage.cached_input_tokens, 2);
        assert_eq!(token_usage.output_tokens, 25);
        assert_eq!(token_usage.reasoning_tokens, 4);
        assert_eq!(token_usage.total_tokens, 38);
    }

    #[test]
    fn file_id_document_serializes_as_input_file_content() {
        let message = message::Message::User {
            content: OneOrMany::one(message::UserContent::Document(message::Document {
                data: DocumentSourceKind::FileId("file_abc".to_string()),
                media_type: None,
                additional_params: None,
            })),
        };

        let converted: Vec<Message> = message.try_into().expect("conversion should succeed");
        let Message::User { content, .. } = &converted[0] else {
            panic!("expected user message");
        };

        let json = serde_json::to_value(content.first_ref()).expect("serialize content");

        assert_eq!(json["type"], "input_file");
        assert_eq!(json["file_id"], "file_abc");
        assert!(json.get("file_data").is_none());
        assert!(json.get("file_url").is_none());
    }

    #[test]
    fn file_id_document_serializes_as_input_item_content() {
        let message = completion::Message::User {
            content: OneOrMany::one(message::UserContent::Document(message::Document {
                data: DocumentSourceKind::FileId("file_abc".to_string()),
                media_type: None,
                additional_params: None,
            })),
        };

        let converted: Vec<InputItem> = message.try_into().expect("conversion should succeed");
        let json = serde_json::to_value(&converted[0]).expect("serialize input item");

        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "input_file");
        assert_eq!(json["content"][0]["file_id"], "file_abc");
        assert!(json["content"][0].get("file_data").is_none());
        assert!(json["content"][0].get("file_url").is_none());
    }

    #[tokio::test]
    async fn responses_completion_http_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::providers::openai::Client;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"bad image","type":"invalid_request_error","code":"invalid_value"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o-mini");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
        let json = error
            .provider_response_json()
            .expect("raw body should be valid JSON")
            .expect("parsed JSON should be present");
        assert_eq!(json["error"]["code"], "invalid_value");
    }

    #[test]
    fn output_unknown_preserves_hosted_tool_payload() {
        let item = json!({
            "type": "web_search_call",
            "id": "ws_001",
            "status": "completed",
            "action": { "type": "search", "queries": ["rig framework"] },
        });

        let output: Output =
            serde_json::from_value(item.clone()).expect("unknown output should deserialize");

        let Output::Unknown(value) = output else {
            panic!("expected Output::Unknown for an unmodeled item type");
        };
        assert_eq!(value, item);
    }

    #[test]
    fn output_unknown_round_trips_value_equal() {
        let item = json!({
            "type": "file_search_call",
            "id": "fs_007",
            "status": "in_progress",
            "queries": ["lifecycle"],
        });

        let output: Output =
            serde_json::from_value(item.clone()).expect("unknown output should deserialize");
        let serialized = serde_json::to_value(&output).expect("unknown output should serialize");

        assert_eq!(serialized, item);
    }

    #[test]
    fn output_known_variant_with_bad_body_errors() {
        // A recognized `type` tag with a malformed body must still error rather
        // than silently degrading to `Output::Unknown`.
        let malformed = json!({
            "type": "function_call",
            "id": "call_1",
            // missing `arguments`, `call_id`, `name`
        });

        let result: Result<Output, _> = serde_json::from_value(malformed);
        assert!(result.is_err());
    }

    #[test]
    fn completion_response_with_unknown_output_keeps_usage() {
        // Guards the original reason the catch-all exists: an unknown item must
        // not break decoding of the whole response or drop token usage.
        let response = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_001",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": { "cached_tokens": 25 },
                "output_tokens": 50,
                "output_tokens_details": { "reasoning_tokens": 15 },
                "total_tokens": 150,
            },
        });

        let response: CompletionResponse =
            serde_json::from_value(response).expect("response should deserialize");

        assert!(matches!(response.output.first(), Some(Output::Unknown(_))));
        let usage = response.usage.expect("usage should be present");
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn output_known_variant_round_trips_value_equal() {
        // The hand-written Serialize must reproduce the modeled wire shape, so a
        // decoded known item re-serializes value-equal to what it came from
        // (guards the `function_call` arm, including its stringified `arguments`).
        // The item ID uses the provider-native `fc_` prefix; other IDs are
        // intentionally dropped on serialization (see `OutputFunctionCall::id`).
        let item = json!({
            "type": "function_call",
            "id": "fc_1",
            "arguments": "{}",
            "call_id": "c1",
            "name": "search",
            "status": "completed",
        });

        let output: Output =
            serde_json::from_value(item.clone()).expect("known output should deserialize");
        assert!(matches!(output, Output::FunctionCall(_)));

        let serialized = serde_json::to_value(&output).expect("known output should serialize");
        assert_eq!(serialized, item);
    }

    #[test]
    fn output_reasoning_round_trips_value_equal() {
        // Highest-value parity guard: the `Reasoning` struct variant threads its
        // fields by hand in *both* directions. Populated `encrypted_content` /
        // `status` (the `#[serde(default)]` optionals) must survive
        // serialize -> deserialize unchanged — catching a dropped field or a
        // forgotten `reasoning` dispatch arm (which would degrade to `Unknown`).
        let original = Output::Reasoning {
            id: "reasoning_1".to_string(),
            summary: vec![ReasoningSummary::SummaryText {
                text: "weighing options".to_string(),
            }],
            content: vec!["private reasoning".to_string()],
            encrypted_content: Some("ENCRYPTED".to_string()),
            status: Some(ToolStatus::Completed),
        };

        let value = serde_json::to_value(&original).expect("reasoning should serialize");
        let round_tripped: Output =
            serde_json::from_value(value).expect("reasoning should deserialize");

        assert_eq!(round_tripped, original);
    }

    #[test]
    fn output_reasoning_none_optionals_serialize_as_explicit_null() {
        // Wire-anchored complement to the round-trip test: with `None`
        // optionals, the keys must still be emitted as explicit `null` (the
        // derived behavior this hand-written serde replaced has no
        // `skip_serializing_if`). Guards against a future refactor silently
        // dropping the keys and changing the wire shape.
        let value = serde_json::to_value(Output::Reasoning {
            id: "reasoning_1".to_string(),
            summary: vec![],
            content: vec![],
            encrypted_content: None,
            status: None,
        })
        .expect("reasoning should serialize");

        assert_eq!(value["type"], "reasoning");
        assert_eq!(value["encrypted_content"], Value::Null);
        assert_eq!(value["status"], Value::Null);
        assert!(value.get("encrypted_content").is_some());
        assert!(value.get("status").is_some());
    }

    #[test]
    fn output_message_round_trips_value_equal() {
        // Wire-anchored serialize check for the `message` arm (only
        // `function_call` was anchored): a decoded message item re-serializes
        // value-equal to the input, tag included.
        let item = json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [ { "type": "output_text", "text": "hello", "annotations": [] } ],
        });

        let output: Output =
            serde_json::from_value(item.clone()).expect("message item should deserialize");
        assert!(matches!(output, Output::Message(_)));

        let serialized = serde_json::to_value(&output).expect("message should serialize");
        assert_eq!(serialized, item);
    }

    #[test]
    fn each_known_tag_decodes_to_its_modeled_variant() {
        // Guards every modeled dispatch arm: a well-formed item for each known
        // `type` must decode to its specific variant, never to `Unknown`. Adding
        // an `Output` variant without a matching deserialize arm fails here
        // instead of silently routing real items to `Unknown`.
        let message: Output = serde_json::from_value(json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
        }))
        .expect("message item should decode");
        assert!(matches!(message, Output::Message(_)));

        let function_call: Output = serde_json::from_value(json!({
            "type": "function_call", "id": "call_1", "arguments": "{}",
            "call_id": "c1", "name": "f", "status": "completed",
        }))
        .expect("function_call item should decode");
        assert!(matches!(function_call, Output::FunctionCall(_)));

        let reasoning: Output =
            serde_json::from_value(json!({ "type": "reasoning", "id": "r1", "summary": [] }))
                .expect("reasoning item should decode");
        assert!(matches!(reasoning, Output::Reasoning { .. }));
    }

    #[test]
    fn output_without_usable_type_tag_decodes_to_unknown() {
        // An absent or non-string `type` is itself unmodeled, so it is captured
        // verbatim as `Unknown` rather than erroring.
        for item in [
            json!({ "id": "x", "note": "no type field" }),
            json!({ "type": 7, "id": "x" }),
        ] {
            let output: Output =
                serde_json::from_value(item.clone()).expect("should decode to Unknown");
            assert_eq!(output, Output::Unknown(item));
        }
    }
}
