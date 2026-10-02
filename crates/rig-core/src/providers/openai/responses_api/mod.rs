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

/// An input item for [`CompletionRequest`]: a system or user message, a
/// function result, or an item the provider stated, sent back as it is.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// A system or user message.
    Message(Message),
    /// A function call's result.
    FunctionCallOutput(ToolResult),
    /// Any other item, in the API's JSON shape: an output item replayed
    /// verbatim, or one rebuilt from a turn's canonical fields.
    #[serde(untagged)]
    Item(Value),
}

impl InputItem {
    pub fn system_message(content: impl Into<String>) -> Self {
        Self::Message(Message::System {
            content: vec![SystemContent::InputText {
                text: content.into(),
            }],
            name: None,
        })
    }

    /// A user-role input item carrying one content part.
    fn user_content(content: UserContent) -> Self {
        Self::Message(Message::User {
            content: vec![content],
            name: None,
        })
    }

    pub(crate) fn system_text(&self) -> Option<String> {
        match self {
            Self::Message(Message::System { content, .. }) => Some(
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

    /// The call id fields a request spells: a call's and a result's.
    fn call_ids(&mut self) -> Vec<&mut String> {
        match self {
            Self::FunctionCallOutput(result) => vec![&mut result.call_id],
            Self::Item(Value::Object(item))
                if item
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| {
                        matches!(
                            kind,
                            "function_call" | "custom_tool_call" | "custom_tool_call_output"
                        )
                    }) =>
            {
                match item.get_mut("call_id") {
                    Some(Value::String(call_id)) => vec![call_id],
                    _ => Vec::new(),
                }
            }
            Self::Message(_) | Self::Item(_) => Vec::new(),
        }
    }
}

/// A tool result.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
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

/// The ids of the calls a request sends as `custom_tool_call` items, whose
/// results go back as `custom_tool_call_output`.
type CustomCalls = std::collections::HashSet<String>;

/// The request message at `position` of the history as input items.
///
/// Only a turn the target model produced still holds provider items (the
/// adapter clears the rest), and it is sent as pi sends it. Reasoning goes
/// as its item whatever its text, which is display only. A block whose item
/// is current goes verbatim. An edited text or call is rebuilt from its
/// canonical fields but keeps its item's `id` (and `phase`), so the
/// reasoning before it stays paired. A block with no item is rebuilt as pi
/// rebuilds another model's turn: text as a completed output message under
/// a synthetic id, a call as a `function_call` with no item id, and
/// reasoning, which only its item can carry, not at all.
fn input_items(
    message: crate::completion::Message,
    position: usize,
    custom: &mut CustomCalls,
) -> Result<Vec<InputItem>, EncodeError> {
    match message {
        crate::completion::Message::System { content } => {
            Ok(vec![InputItem::Message(Message::System {
                content: vec![content.into()],
                name: None,
            })])
        }
        crate::completion::Message::User { content } => {
            let mut parts = Vec::with_capacity(content.len());
            for content in content {
                parts.extend(user_input_item(content, custom)?);
            }
            Ok(parts)
        }
        crate::completion::Message::Assistant(turn) => {
            let mut items = Vec::new();
            let mut texts = 0usize;
            for block in turn.content {
                if let Some(item) = block.native_item() {
                    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call")
                        && let Some(call_id) = item.get("call_id").and_then(Value::as_str)
                    {
                        custom.insert(call_id.to_owned());
                    }
                    items.push(InputItem::Item(item.clone()));
                    continue;
                }
                match block {
                    crate::message::AssistantContent::Text(text) => {
                        // An empty message says nothing.
                        if text.text.is_empty() {
                            continue;
                        }
                        let native = text.native.as_ref().map(|native| &native.item);
                        let id = match native.and_then(|item| item_id(item, "")) {
                            Some(id) => id,
                            None => synthetic_id(position, &mut texts),
                        };
                        let phase = native.and_then(|item| item.get("phase")).cloned();
                        items.push(rebuilt_message(&text.text, id, phase));
                    }
                    crate::message::AssistantContent::ToolCall(call) => {
                        let native = call.native.as_ref().map(|native| &native.item);
                        let custom_call = native
                            .and_then(|item| item.get("type"))
                            .and_then(Value::as_str)
                            == Some("custom_tool_call");
                        let call_id = call.id.wire().into_owned();
                        let mut item = if custom_call {
                            custom.insert(call_id.clone());
                            let input = match call.function.arguments.get("input") {
                                Some(Value::String(input)) => input.clone(),
                                _ => call.function.arguments_value().to_string(),
                            };
                            serde_json::json!({
                                "type": "custom_tool_call",
                                "call_id": call_id,
                                "name": call.function.name.as_str(),
                                "input": input,
                            })
                        } else {
                            serde_json::json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": call.function.name.as_str(),
                                "arguments": call.function.arguments_value().to_string(),
                            })
                        };
                        let prefix = if custom_call { "ctc_" } else { "fc_" };
                        if let (Some(id), Some(fields)) = (
                            native.and_then(|item| item_id(item, prefix)),
                            item.as_object_mut(),
                        ) {
                            fields.insert("id".to_owned(), Value::String(id));
                        }
                        items.push(InputItem::Item(item));
                    }
                    // The same model's reasoning goes as its item even when
                    // its text was edited; without one there is nothing to
                    // send.
                    crate::message::AssistantContent::Reasoning(reasoning) => {
                        if let Some(native) = reasoning.native {
                            items.push(InputItem::Item(native.item));
                        }
                    }
                    crate::message::AssistantContent::Opaque(opaque) => {
                        items.push(InputItem::Item(opaque.item));
                    }
                    // Responses takes no image in an assistant turn; the
                    // adapter downgrades another model's, and this one names
                    // what was there.
                    crate::message::AssistantContent::Image(_) => {
                        items.push(rebuilt_message(
                            crate::completion::history::ASSISTANT_IMAGE_OMITTED,
                            synthetic_id(position, &mut texts),
                            None,
                        ));
                    }
                }
            }
            Ok(items)
        }
    }
}

/// The id of the next rebuilt message of the turn at `position`, as pi
/// numbers them.
fn synthetic_id(position: usize, texts: &mut usize) -> String {
    let id = match *texts {
        0 => format!("msg_rig_{position}"),
        n => format!("msg_rig_{position}_{n}"),
    };
    *texts += 1;
    id
}

/// A completed assistant output message holding `text`.
fn rebuilt_message(text: &str, id: String, phase: Option<Value>) -> InputItem {
    let mut item = serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "status": "completed",
        "id": id,
    });
    if let (Some(phase @ Value::String(_)), Some(fields)) = (phase, item.as_object_mut()) {
        fields.insert("phase".to_owned(), phase);
    }
    InputItem::Item(item)
}

/// The `id` of a stored item, when it has the `prefix` its replayed type
/// requires and fits the API's 64 characters.
fn item_id(item: &Value, prefix: &str) -> Option<String> {
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && id.starts_with(prefix) && id.len() <= 64)
        .map(str::to_owned)
}

/// One user content part as an input item.
fn user_input_item(
    content: crate::message::UserContent,
    custom: &CustomCalls,
) -> Result<Option<InputItem>, EncodeError> {
    Ok(Some(match content {
        // Blank text says nothing, and some backends reject it.
        crate::message::UserContent::Text(Text { text, .. }) if text.trim().is_empty() => {
            return Ok(None);
        }
        crate::message::UserContent::Text(Text { text, .. }) => {
            InputItem::user_content(UserContent::InputText { text })
        }
        // A function output has no error field: a failed result says so in
        // its text, and `is_error` is not sent.
        crate::message::UserContent::ToolResult(tool_result) => {
            let call_id = tool_result.call.wire().into_owned();
            let output = responses_tool_result_output(tool_result.content)?;
            if custom.contains(&call_id) {
                InputItem::Item(serde_json::json!({
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": output,
                }))
            } else {
                InputItem::FunctionCallOutput(ToolResult {
                    call_id,
                    output,
                    status: ToolStatus::Completed,
                })
            }
        }
        crate::message::UserContent::Document(Document {
            data: DocumentSourceKind::FileId(file_id),
            ..
        }) => InputItem::user_content(UserContent::InputFile {
            file_id: Some(file_id),
            file_data: None,
            file_url: None,
            filename: None,
        }),
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
            InputItem::user_content(UserContent::InputFile {
                file_id: None,
                file_data,
                file_url,
                filename,
            })
        }
        // A URL whose type the caller did not name: `input_file` fetches it
        // and reads the type itself.
        crate::message::UserContent::Document(Document {
            data: DocumentSourceKind::Url(url),
            media_type: None,
            ..
        }) => InputItem::user_content(UserContent::InputFile {
            file_id: None,
            file_data: None,
            file_url: Some(url),
            filename: None,
        }),
        crate::message::UserContent::Document(Document {
            data: DocumentSourceKind::Base64(text) | DocumentSourceKind::String(text),
            ..
        }) => InputItem::user_content(UserContent::InputText { text }),
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
            InputItem::user_content(UserContent::InputImage {
                image_url: url,
                detail: detail.unwrap_or_default(),
            })
        }
        message => {
            return Err(EncodeError::request(format!(
                "Unsupported message: {message:?}"
            )));
        }
    }))
}

/// A function or hosted tool available to a Responses request.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct ResponsesToolDefinition {
    /// The type of tool.
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name
    #[serde(default)]
    pub name: String,
    /// Parameters - this should be a JSON schema. Strict function tools must use OpenAI's supported strict schema subset.
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Whether to use strict mode. Disabled by default; opt in with [`Self::with_strict`]
    /// or [`wire::Responses::with_strict_tools`].
    ///
    /// Always serialized on a function tool: the Responses API treats an omitted `strict`
    /// as "attempt strict mode", so `false` must reach the wire for non-strict tools to
    /// actually be non-strict. Never serialized on a hosted tool, which answers the field
    /// with a 400 (`Unknown parameter: 'tools[0].strict'`).
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub strict: bool,
    /// Tool description.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub description: String,
    /// Additional provider-specific configuration for hosted tools.
    #[serde(flatten, default)]
    pub config: Map<String, Value>,
}

impl Serialize for ResponsesToolDefinition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", &self.kind)?;
        if !self.name.is_empty() {
            map.serialize_entry("name", &self.name)?;
        }
        if !self.parameters.is_null() {
            map.serialize_entry("parameters", &self.parameters)?;
        }
        if self.kind == "function" {
            map.serialize_entry("strict", &self.strict)?;
        }
        if !self.description.is_empty() {
            map.serialize_entry("description", &self.description)?;
        }
        for (key, value) in &self.config {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
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
                let mut names = function_names.into_iter().map(String::from);
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
        let mut custom = CustomCalls::new();
        let mut position = 0;
        let mut input = crate::providers::internal::wire_ids::WireIds::convert(
            chat_history,
            |message| {
                position += 1;
                input_items(message, position - 1, &mut custom)
            },
            InputItem::call_ids,
        )?;

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
            if let Some(raw_tools) = additional_params_map.shift_remove("tools") {
                additional_tools = serde_json::from_value::<Vec<ResponsesToolDefinition>>(
                    raw_tools,
                )
                .map_err(|err| {
                    EncodeError::request(format!(
                        "Invalid OpenAI Responses tools payload in additional_params: {err}"
                    ))
                })?;
            }
            additional_params_map.shift_remove("stream");
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
        // Reasoning replays without stored state only with its ciphertext.
        if additional_parameters.reasoning.is_some() || additional_parameters.store == Some(false) {
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

/// The status of a given tool.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
    /// A status rig does not model, kept as it came.
    #[serde(untagged)]
    Other(String),
}

/// A system or user message of a Responses request.
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
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: vec![content.to_owned().into()],
            name: None,
        }
    }
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
