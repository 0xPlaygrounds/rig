//! The OpenAI Responses API.
//!
//! By default when creating a completion client, this is the API that gets used.
//!
//! If you'd like to switch back to the regular Completions API, you can do so by using the `.completions_api()` function - see below for an example:
//! ```rust
//! let openai_client = rig::providers::openai::Client::from_env();
//! let model = openai_client.completion_model("gpt-4o").completions_api();
//! ```
use super::InputAudio;
use super::completion::ToolChoice;
use super::{Client, responses_api::streaming::StreamingCompletionResponse};
use crate::completion::CompletionError;
use crate::http_client;
use crate::http_client::HttpClientExt;
use crate::json_utils;
use crate::message::{
    AudioMediaType, Document, DocumentMediaType, DocumentSourceKind, ImageDetail, MessageError,
    MimeType, Text,
};
use crate::one_or_many::string_or_one_or_many;

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, completion, message};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::{Instrument, Level, enabled, info_span};

use std::convert::Infallible;
use std::ops::Add;
use std::str::FromStr;

pub mod streaming;

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
    /// The tools you want to use. Currently this is limited to functions, but will be expanded on in future.
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

/// A tool result.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolResult {
    /// The call ID of a tool (this should be linked to the call ID for a tool call, otherwise an error will be received)
    call_id: String,
    /// The result of a tool call.
    output: String,
    /// The status of a tool call (if used in a completion request, this should always be Completed)
    status: ToolStatus,
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
            crate::completion::Message::User { content } => {
                let mut items = Vec::new();

                for user_content in content {
                    match user_content {
                        crate::message::UserContent::Text(Text { text }) => {
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
                            for tool_result_content in tool_content {
                                let crate::completion::message::ToolResultContent::Text(Text {
                                    text,
                                }) = tool_result_content
                                else {
                                    return Err(CompletionError::ProviderError(
                                        "This thing only supports text!".to_string(),
                                    ));
                                };
                                // let output = serde_json::from_str(&text)?;
                                items.push(InputItem {
                                    role: None,
                                    input: InputContent::FunctionCallOutput(ToolResult {
                                        call_id: require_call_id(call_id.clone(), "Tool result")?,
                                        output: text,
                                        status: ToolStatus::Completed,
                                    }),
                                });
                            }
                        }
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
                        crate::message::AssistantContent::Text(Text { text }) => {
                            let id = id.as_ref().unwrap_or(&String::default()).clone();
                            other_items.push(InputItem {
                                role: Some(Role::Assistant),
                                input: InputContent::Message(Message::Assistant {
                                    content: OneOrMany::one(AssistantContentType::Text(
                                        AssistantContent::OutputText(Text { text }),
                                    )),
                                    id,
                                    name: None,
                                    status: ToolStatus::Completed,
                                }),
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
                            reasoning_items.push(InputItem {
                                role: None,
                                input: InputContent::Reasoning(openai_reasoning),
                            });
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
) -> Result<OpenAIReasoning, MessageError> {
    let id = reasoning.id.clone().ok_or_else(|| {
        MessageError::ConversionError(
            "An OpenAI-generated ID is required when using OpenAI reasoning items".to_string(),
        )
    })?;
    let mut summary = Vec::new();
    let mut encrypted_content = None;
    for content in &reasoning.content {
        match content {
            crate::message::ReasoningContent::Text { text, .. }
            | crate::message::ReasoningContent::Summary(text) => {
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

    Ok(OpenAIReasoning {
        id,
        summary,
        encrypted_content,
        status: None,
    })
}

/// The definition of a tool response, repurposed for OpenAI's Responses API.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponsesToolDefinition {
    /// Tool name
    pub name: String,
    /// Parameters - this should be a JSON schema. Tools should additionally ensure an "additionalParameters" field has been added with the value set to false, as this is required if using OpenAI's strict mode (enabled by default).
    pub parameters: serde_json::Value,
    /// Whether to use strict mode. Enabled by default as it allows for improved efficiency.
    pub strict: bool,
    /// The type of tool. This should always be "function".
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool description.
    pub description: String,
}

impl From<completion::ToolDefinition> for ResponsesToolDefinition {
    fn from(value: completion::ToolDefinition) -> Self {
        let completion::ToolDefinition {
            name,
            mut parameters,
            description,
        } = value;

        super::sanitize_schema(&mut parameters);

        Self {
            name,
            parameters,
            description,
            kind: "function".to_string(),
            strict: true,
        }
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
    pub output_tokens_details: OutputTokensDetails,
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
            output_tokens_details: OutputTokensDetails::new(),
            total_tokens: 0,
        }
    }
}

impl Add for ResponsesUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let input_tokens = self.input_tokens + rhs.input_tokens;
        let input_tokens_details = self.input_tokens_details.map(|lhs| {
            if let Some(tokens) = rhs.input_tokens_details {
                lhs + tokens
            } else {
                lhs
            }
        });
        let output_tokens = self.output_tokens + rhs.output_tokens;
        let output_tokens_details = self.output_tokens_details + rhs.output_tokens_details;
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

/// Attempt to try and create a `NewCompletionRequest` from a model name and [`crate::completion::CompletionRequest`]
impl TryFrom<(String, crate::completion::CompletionRequest)> for CompletionRequest {
    type Error = CompletionError;
    fn try_from(
        (model, req): (String, crate::completion::CompletionRequest),
    ) -> Result<Self, Self::Error> {
        let model = req.model.clone().unwrap_or(model);
        let input = {
            let mut partial_history = vec![];
            if let Some(docs) = req.normalized_documents() {
                partial_history.push(docs);
            }
            partial_history.extend(req.chat_history);

            // Initialize full history with preamble (or empty if non-existent)
            // Some "Responses API compatible" providers don't support `instructions` field
            // so we need to add a system message until further notice
            let mut full_history: Vec<InputItem> = if let Some(content) = req.preamble {
                vec![InputItem::system_message(content)]
            } else {
                Vec::new()
            };

            for history_item in partial_history {
                full_history.extend(<Vec<InputItem>>::try_from(history_item)?);
            }

            full_history
        };

        let input = OneOrMany::many(input).map_err(|_| {
            CompletionError::RequestError(
                "OpenAI Responses request input must contain at least one item".into(),
            )
        })?;

        let stream = req
            .additional_params
            .clone()
            .unwrap_or(Value::Null)
            .as_bool();

        let mut additional_parameters = if let Some(map) = req.additional_params {
            serde_json::from_value::<AdditionalParameters>(map).map_err(|err| {
                CompletionError::RequestError(
                    format!("Invalid OpenAI Responses additional_params payload: {err}").into(),
                )
            })?
        } else {
            // If there's no additional parameters, initialise an empty object
            AdditionalParameters::default()
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

        Ok(Self {
            input,
            model,
            instructions: None, // is currently None due to lack of support in compliant providers
            max_output_tokens: req.max_tokens,
            stream,
            tool_choice,
            tools: req
                .tools
                .into_iter()
                .map(ResponsesToolDefinition::from)
                .collect(),
            temperature: req.temperature,
            additional_parameters,
        })
    }
}

/// The completion model struct for OpenAI's response API.
#[derive(Clone)]
pub struct ResponsesCompletionModel<T = reqwest::Client> {
    /// The OpenAI client
    pub(crate) client: Client<T>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl<T> ResponsesCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    /// Creates a new [`ResponsesCompletionModel`].
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    /// Use the Completions API instead of Responses.
    pub fn completions_api(self) -> crate::providers::openai::completion::CompletionModel<T> {
        super::completion::CompletionModel::with_model(self.client.completions_api(), &self.model)
    }

    /// Attempt to create a completion request from [`crate::completion::CompletionRequest`].
    pub(crate) fn create_completion_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<CompletionRequest, CompletionError> {
        let req = CompletionRequest::try_from((self.model.clone(), completion_request))?;

        Ok(req)
    }
}

/// The standard response format from OpenAI's Responses API.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// Token usage
    pub usage: Option<ResponsesUsage>,
    /// The model output (messages, etc will go here)
    pub output: Vec<Output>,
    /// Tools
    #[serde(default)]
    pub tools: Vec<ResponsesToolDefinition>,
    /// Additional parameters
    #[serde(flatten)]
    pub additional_parameters: AdditionalParameters,
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
    /// Any additional metadata you'd like to add. This will additionally be returned by the response.
    #[serde(skip_serializing_if = "Map::is_empty", default)]
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
    pub strict: bool,
}

/// Add reasoning to a [`CompletionRequest`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// How much effort you want the model to put into thinking/reasoning.
    pub effort: Option<ReasoningEffort>,
    /// How much effort you want the model to put into writing the reasoning summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryLevel>,
}

impl Reasoning {
    /// Creates a new Reasoning instantiation (with empty values).
    pub fn new() -> Self {
        Self {
            effort: None,
            summary: None,
        }
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
}

/// The billing service tier that will be used. On auto by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
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

/// A currently non-exhaustive list of output types.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Output {
    Message(OutputMessage),
    #[serde(alias = "function_call")]
    FunctionCall(OutputFunctionCall),
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        #[serde(default)]
        encrypted_content: Option<String>,
        #[serde(default)]
        status: Option<ToolStatus>,
    },
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
    pub id: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
    pub call_id: String,
    pub name: String,
    pub status: ToolStatus,
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

impl<T> completion::CompletionModel for ResponsesCompletionModel<T>
where
    T: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = super::Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.provider.name", "openai");
        span.record("gen_ai.request.model", &self.model);
        let request = self.create_completion_request(completion_request)?;
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
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
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
        ResponsesCompletionModel::stream(self, request).await
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        if response.output.is_empty() {
            return Err(CompletionError::ResponseError(
                "Response contained no parts".to_owned(),
            ));
        }

        // Extract the msg_ ID from the first Output::Message item
        let message_id = response.output.iter().find_map(|item| match item {
            Output::Message(msg) => Some(msg.id.clone()),
            _ => None,
        });

        let content: Vec<completion::AssistantContent> = response
            .output
            .iter()
            .cloned()
            .flat_map(<Vec<completion::AssistantContent>>::from)
            .collect();

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage
                    .input_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
            })
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
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        output: String,
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
                completion::AssistantContent::Text(Text { text: refusal })
            }
            AssistantContent::OutputText(Text { text }) => {
                completion::AssistantContent::Text(Text { text })
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

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                // If there are messages with both tool results and user content, openai will only
                //  handle tool results. It's unlikely that there will be both.
                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::ToolResult(message::ToolResult {
                                call_id,
                                content,
                                ..
                            }) => Ok::<_, message::MessageError>(Message::ToolResult {
                                tool_call_id: call_id.ok_or_else(|| {
                                    MessageError::ConversionError(
                                        "Tool result `call_id` is required for OpenAI Responses API"
                                            .into(),
                                    )
                                })?,
                                output: {
                                    let res = content.first();
                                    match res {
                                        completion::message::ToolResultContent::Text(Text {
                                            text,
                                        }) => text,
                                        _ => return  Err(MessageError::ConversionError("This API only currently supports text tool results".into()))
                                    }
                                },
                            }),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    let other_content = other_content
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::Text(message::Text { text }) => {
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
                                        let media_type = if let Some(media_type) = media_type {
                                            media_type.to_mime_type().to_string()
                                        } else {
                                            String::new()
                                        };
                                        format!("data:{media_type};base64,{data}")
                                    }
                                    DocumentSourceKind::Url(url) => url,
                                    DocumentSourceKind::Raw(_) => {
                                        return Err(MessageError::ConversionError(
                                            "Raw files not supported, encode as base64 first"
                                                .into(),
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
                                            "Raw files not supported, encode as base64 first"
                                                .into(),
                                        ));
                                    }
                                    doc => {
                                        return Err(MessageError::ConversionError(format!(
                                            "Unsupported document type: {doc}"
                                        )));
                                    }
                                };

                                Ok(UserContent::InputFile {
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
                                    format: match media_type {
                                        Some(media_type) => media_type,
                                        None => AudioMediaType::MP3,
                                    },
                                },
                            }),
                            message::UserContent::Audio(_) => Err(MessageError::ConversionError(
                                "Audio must be base64 encoded data".into(),
                            )),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let other_content = OneOrMany::many(other_content).map_err(|_| {
                        MessageError::ConversionError(
                            "User message did not contain OpenAI Responses-compatible content"
                                .to_string(),
                        )
                    })?;

                    Ok(vec![Message::User {
                        content: other_content,
                        name: None,
                    }])
                }
            }
            message::Message::Assistant { content, id } => {
                let assistant_message_id = id.ok_or_else(|| {
                    MessageError::ConversionError(
                        "Assistant message ID is required for OpenAI Responses API".into(),
                    )
                })?;

                match content.first() {
                    crate::message::AssistantContent::Text(Text { text }) => {
                        Ok(vec![Message::Assistant {
                            id: assistant_message_id.clone(),
                            status: ToolStatus::Completed,
                            content: OneOrMany::one(AssistantContentType::Text(
                                AssistantContent::OutputText(Text { text }),
                            )),
                            name: None,
                        }])
                    }
                    crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
                        id,
                        call_id,
                        function,
                        ..
                    }) => Ok(vec![Message::Assistant {
                        content: OneOrMany::one(AssistantContentType::ToolCall(
                            OutputFunctionCall {
                                call_id: call_id.ok_or_else(|| {
                                    MessageError::ConversionError(
                                        "Tool call `call_id` is required for OpenAI Responses API"
                                            .into(),
                                    )
                                })?,
                                arguments: function.arguments,
                                id,
                                name: function.name,
                                status: ToolStatus::Completed,
                            },
                        )),
                        id: assistant_message_id.clone(),
                        name: None,
                        status: ToolStatus::Completed,
                    }]),
                    crate::message::AssistantContent::Reasoning(reasoning) => {
                        let openai_reasoning = openai_reasoning_from_core(&reasoning)?;
                        Ok(vec![Message::Assistant {
                            content: OneOrMany::one(AssistantContentType::Reasoning(
                                openai_reasoning,
                            )),
                            id: assistant_message_id,
                            name: None,
                            status: ToolStatus::Completed,
                        }])
                    }
                    crate::message::AssistantContent::Image(_) => {
                        Err(MessageError::ConversionError(
                            "Assistant image content is not supported in OpenAI Responses API"
                                .into(),
                        ))
                    }
                }
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
