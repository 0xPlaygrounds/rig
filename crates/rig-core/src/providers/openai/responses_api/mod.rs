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

use crate::completion::history::Replay;
use crate::error::EncodeError;
use crate::json_utils;
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Message, MimeType,
    ToolResultContent, UserContent,
};
use crate::providers::internal::wire_ids::WireIds;
use crate::{completion, message};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

pub mod streaming;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod websocket;
pub mod wire;

/// The `input_image` part `image` goes as: a data URL for typed base64
/// data, its URL, or its file id. `None` for any other source, which
/// [`wire::Responses`] tells the adapter it does not carry.
fn image_part(image: &message::Image) -> Option<Value> {
    let (key, source) = match &image.data {
        DocumentSourceKind::Base64(data) => (
            "image_url",
            format!(
                "data:{};base64,{data}",
                image.media_type.as_ref()?.to_mime_type()
            ),
        ),
        DocumentSourceKind::Url(url) => ("image_url", url.clone()),
        DocumentSourceKind::FileId(file_id) => ("file_id", file_id.clone()),
        _ => return None,
    };
    let mut part =
        json!({"type": "input_image", "detail": image.detail.clone().unwrap_or_default()});
    part[key] = Value::String(source);
    Some(part)
}

/// The content part `document` goes as: an `input_file` for a file id, a
/// URL or base64 PDF data, and its text for a string. `None` for any other
/// form; the adapter sends a text document's text instead.
fn document_part(document: &Document) -> Option<Value> {
    Some(match &document.data {
        DocumentSourceKind::FileId(file_id) => json!({"type": "input_file", "file_id": file_id}),
        // `input_file` reads the type of a file it fetches itself.
        DocumentSourceKind::Url(url) => json!({"type": "input_file", "file_url": url}),
        DocumentSourceKind::Base64(data) if document.media_type == Some(DocumentMediaType::PDF) => {
            json!({
                "type": "input_file",
                "file_data": format!("data:application/pdf;base64,{data}"),
                "filename": "document.pdf",
            })
        }
        DocumentSourceKind::String(text) => json!({"type": "input_text", "text": text}),
        _ => return None,
    })
}

/// The refusal for a part a request was not prepared to carry.
fn unsendable(part: &str) -> EncodeError {
    EncodeError::request(format!(
        "the Responses API cannot carry this {part}; prepare the request first"
    ))
}

/// A tool result's `output`: one text as a string, anything else as its
/// ordered parts.
fn result_output(content: &[ToolResultContent]) -> Result<Value, EncodeError> {
    let mut parts = content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(json!({"type": "input_text", "text": text.text})),
            ToolResultContent::Json { value } => {
                Ok(json!({"type": "input_text", "text": value.to_string()}))
            }
            ToolResultContent::Image(image) => {
                image_part(image).ok_or_else(|| unsendable("tool-result image"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match parts.as_mut_slice() {
        [part] if part.str("type") == Some("input_text") => part["text"].take(),
        _ => Value::Array(parts),
    })
}

/// The custom tools of a request: a call to one is a `custom_tool_call`,
/// answered by a `custom_tool_call_output`, as pi decides by name.
struct Custom {
    /// The custom tools the request declares.
    tools: std::collections::HashSet<String>,
    /// The call ids the request sends as custom calls.
    calls: std::collections::HashSet<String>,
}

/// The input items of `history` for `target`, which addresses `model`. A
/// turn the target model produced keeps its provider items (the adapter
/// cleared the rest), sent as pi sends them: a current item verbatim with
/// its call id spelled, an edited block rebuilt under its item's identity,
/// and any other block rebuilt as pi rebuilds another model's turn: text as
/// a completed output message under a synthetic id, and a call with no
/// item id. Reasoning only its item can carry goes only with its identity.
fn input(
    history: &[Message],
    target: &wire::Responses,
    model: &str,
    custom: &mut Custom,
) -> Result<Vec<Value>, EncodeError> {
    let ids = WireIds::for_target(history, target, model);
    let spelled = |call: &message::CallId| {
        ids.of(call)
            .map_or_else(|| call.wire().into_owned(), str::to_owned)
    };
    let mut items = Vec::new();
    for (position, message) in history.iter().enumerate() {
        match message {
            Message::System { content } => items.push(json!({
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": content}],
            })),
            Message::User { content } => {
                for part in content {
                    let part = match part {
                        // Blank text says nothing, and some backends reject it.
                        UserContent::Text(text) if text.text.trim().is_empty() => continue,
                        UserContent::Text(text) => json!({"type": "input_text", "text": text.text}),
                        // A function output has no error field: a failed
                        // result says so in its text.
                        UserContent::ToolResult(result) => {
                            let call_id = spelled(&result.call);
                            let output = result_output(&result.content)?;
                            items.push(
                                if custom.calls.contains(&call_id)
                                    || custom.tools.contains(result.name.as_str())
                                {
                                    json!({"type": "custom_tool_call_output", "call_id": call_id, "output": output})
                                } else {
                                    json!({"type": "function_call_output", "call_id": call_id, "output": output, "status": "completed"})
                                },
                            );
                            continue;
                        }
                        UserContent::Image(image) => {
                            image_part(image).ok_or_else(|| unsendable("image"))?
                        }
                        UserContent::Document(document) => {
                            document_part(document).ok_or_else(|| unsendable("document"))?
                        }
                        UserContent::Audio(_) => return Err(unsendable("audio")),
                        UserContent::Video(_) => return Err(unsendable("video")),
                    };
                    items.push(json!({"type": "message", "role": "user", "content": [part]}));
                }
            }
            Message::Assistant(turn) => {
                let mut texts = 0usize;
                for block in &turn.content {
                    let identity = match block.replay(target, &ids) {
                        Replay::Item(item) => {
                            if item.str("type") == Some("custom_tool_call")
                                && let Some(call_id) = item.str("call_id")
                            {
                                custom.calls.insert(call_id.to_owned());
                            }
                            items.push(item.into_owned());
                            continue;
                        }
                        Replay::Identity(identity) => identity,
                        Replay::Rebuild => Map::new(),
                    };
                    let id = |prefix: &str| {
                        identity
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty() && id.starts_with(prefix) && id.len() <= 64)
                            .map(str::to_owned)
                    };
                    let item = match block {
                        AssistantContent::Text(text) => {
                            let synthetic = match texts {
                                0 => format!("msg_rig_{position}"),
                                n => format!("msg_rig_{position}_{n}"),
                            };
                            texts += 1;
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text.text, "annotations": []}],
                                "status": "completed",
                                "id": id("").unwrap_or(synthetic),
                            });
                            if let Some(phase) =
                                identity.get("phase").filter(|phase| phase.is_string())
                            {
                                item["phase"] = phase.clone();
                            }
                            items.push(item);
                            continue;
                        }
                        AssistantContent::ToolCall(call) => {
                            let call_id = spelled(&call.id);
                            let name = call.function.name.as_str();
                            let kind = identity.get("type").and_then(Value::as_str);
                            if kind == Some("custom_tool_call")
                                || (kind.is_none() && custom.tools.contains(name))
                            {
                                custom.calls.insert(call_id.clone());
                                let input = match call.function.arguments.get("input") {
                                    Some(Value::String(input)) => input.clone(),
                                    _ => call.function.arguments_value().to_string(),
                                };
                                let mut item = json!({"type": "custom_tool_call", "call_id": call_id, "name": name, "input": input});
                                if let Some(id) = id("ctc_") {
                                    item["id"] = json!(id);
                                }
                                item
                            } else {
                                let arguments = call.function.arguments_value().to_string();
                                let mut item = json!({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments});
                                if let Some(id) = id("fc_") {
                                    item["id"] = json!(id);
                                }
                                item
                            }
                        }
                        // An edited reasoning block keeps its item's id and
                        // ciphertext, so the items after it stay paired.
                        AssistantContent::Reasoning(reasoning) => {
                            let Some(rs) = id("") else {
                                continue;
                            };
                            let summary: Vec<Value> = (!reasoning.text.is_empty())
                                .then(|| json!({"type": "summary_text", "text": reasoning.text}))
                                .into_iter()
                                .collect();
                            let mut item =
                                json!({"type": "reasoning", "id": rs, "summary": summary});
                            if let Some(ciphertext) = identity.get("encrypted_content") {
                                item["encrypted_content"] = ciphertext.clone();
                            }
                            item
                        }
                        AssistantContent::Opaque(opaque) if opaque.replay => opaque.item.clone(),
                        AssistantContent::Opaque(_) => continue,
                        AssistantContent::Image(_) => return Err(unsendable("assistant image")),
                    };
                    items.push(item);
                }
            }
        }
    }
    Ok(items)
}

/// The JSON of a tool choice.
fn tool_choice(choice: message::ToolChoice) -> Result<Value, EncodeError> {
    Ok(match choice {
        message::ToolChoice::Auto => json!("auto"),
        message::ToolChoice::None => json!("none"),
        message::ToolChoice::Required => json!("required"),
        message::ToolChoice::Specific { function_names } => match function_names.as_slice() {
            [] => {
                return Err(EncodeError::request(
                    "ToolChoice::Specific requires at least one function name",
                ));
            }
            [name] => json!({"type": "function", "name": name}),
            names => json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": names.iter().map(|name| json!({"type": "function", "name": name})).collect::<Vec<_>>(),
            }),
        },
    })
}

/// Ask for the reasoning ciphertext, without which reasoning replays only
/// from stored state.
fn include_ciphertext(body: &mut Value) {
    const CIPHERTEXT: &str = "reasoning.encrypted_content";
    if !body["include"].is_array() {
        body["include"] = json!([]);
    }
    if let Some(include) = body["include"].as_array_mut()
        && !include.iter().any(|item| item == CIPHERTEXT)
    {
        include.push(json!(CIPHERTEXT));
    }
}

/// The request body `wire` sends for `request`.
fn body(
    wire: &wire::Responses,
    mut request: completion::CompletionRequest,
    streaming: bool,
) -> Result<Value, EncodeError> {
    let model = request.model.take().unwrap_or_else(|| wire.model.clone());
    let mut params = match request.additional_params.take() {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(EncodeError::request(
                "Invalid OpenAI Responses additional_params payload: not an object",
            ));
        }
    };
    params.shift_remove("stream");
    let mut tools: Vec<ResponsesToolDefinition> = request
        .tools
        .into_iter()
        .map(ResponsesToolDefinition::from)
        .collect();
    if let Some(extra) = params.shift_remove("tools") {
        tools.extend(
            serde_json::from_value::<Vec<ResponsesToolDefinition>>(extra).map_err(|err| {
                EncodeError::request(format!(
                    "Invalid OpenAI Responses tools payload in additional_params: {err}"
                ))
            })?,
        );
    }
    tools.extend(wire.tools.iter().cloned());
    if wire.strict_tools {
        tools = tools
            .into_iter()
            .map(ResponsesToolDefinition::with_strict)
            .collect();
    }
    let mut custom = Custom {
        tools: tools
            .iter()
            .filter(|tool| tool.kind == "custom")
            .map(|tool| tool.name.clone())
            .collect(),
        calls: Default::default(),
    };
    let mut input = input(&request.chat_history, wire, &model, &mut custom)?;

    let system = |item: &Value| {
        (item.str("role") == Some("system")).then(|| {
            item.at("/content/0/text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        })
    };
    let before = input.len();
    let mut lifted = Vec::new();
    match wire.system_instructions {
        // The leading run of system messages, unless it is the whole
        // request, which then keeps them in `input` so it is not empty.
        SystemInstructionsPlacement::Instructions => {
            let leading = input
                .iter()
                .take_while(|item| system(item).is_some())
                .count();
            if leading < input.len() {
                lifted.extend(input.drain(..leading).filter_map(|item| system(&item)));
            }
        }
        SystemInstructionsPlacement::AllInstructions => input.retain(|item| match system(item) {
            Some(text) => {
                lifted.push(text);
                false
            }
            None => true,
        }),
        SystemInstructionsPlacement::InputSystemMessages => {}
    }
    if input.is_empty() {
        return Err(EncodeError::request(if input.len() < before {
            "OpenAI Responses request input must contain at least one non-system item \
             (system messages were lifted into the top-level `instructions` field)"
        } else {
            "OpenAI Responses request input must contain at least one item"
        }));
    }
    let lifted: Vec<&str> = lifted
        .iter()
        .map(|text| text.trim())
        .filter(|text| !text.is_empty())
        .collect();
    let instructions = match (&wire.provider.instructions, lifted.is_empty()) {
        (Some(gateway), _) => Some(wire::merge_instructions(
            gateway,
            Some(&lifted.join("\n\n")),
        )),
        (None, false) => Some(lifted.join("\n\n")),
        (None, true) => None,
    };

    let mut body = json!({"model": model, "input": input});
    let fields = [
        ("instructions", instructions.map(Value::from)),
        ("max_output_tokens", request.max_tokens.map(Value::from)),
        ("temperature", request.temperature.map(Value::from)),
        (
            "tool_choice",
            request.tool_choice.map(tool_choice).transpose()?,
        ),
        ("tools", (!tools.is_empty()).then(|| json!(tools))),
        ("stream", streaming.then_some(Value::Bool(true))),
    ];
    for (key, value) in fields {
        if let Some(value) = value {
            body[key] = value;
        }
    }
    for (key, value) in params {
        if !value.is_null() && body.get(&key).is_none() {
            body[key.as_str()] = value;
        }
    }
    if body.get("text").is_none()
        && let Some(schema) = request.output_schema
    {
        let (name, schema) = super::structured_output_schema(schema);
        body["text"] = json!({"format": {"type": "json_schema", "name": name, "schema": schema, "strict": true}});
    }
    let codex = wire.provider.dialect.quirks.responses.contract == wire::ResponsesContract::Codex;
    if codex {
        // The codex gateway takes the turn and the tools; sampling, storage,
        // metadata and structured output are not its to accept, and
        // `store: false` is the one value it wants stated.
        if let Some(fields) = body.as_object_mut() {
            for key in [
                "temperature",
                "max_output_tokens",
                "background",
                "metadata",
                "parallel_tool_calls",
                "service_tier",
                "text",
                "top_p",
                "user",
            ] {
                fields.shift_remove(key);
            }
        }
        body["store"] = json!(false);
    }
    // Reasoning replays without stored state only with its ciphertext.
    if codex || body.get("reasoning").is_some() || body.get("store") == Some(&json!(false)) {
        include_ciphertext(&mut body);
    }
    Ok(body)
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
        deserialize_with = "json_utils::null_or_default"
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    #[serde(untagged)]
    Other(String),
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

#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod stateless_replay_tests;
#[cfg(test)]
mod tests;
