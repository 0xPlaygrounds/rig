//! The OpenAI Responses API.
//!
//! By default when creating a completion client, this is the API that gets used.
//!
//! If you'd like to switch back to the regular Completions API, you can do so by using the `.completions_api()` function - see below for an example:
//! ```rust
//! let openai_client = rig::providers::openai::Client::from_env();
//! let model = openai_client.completion_model.completion_model("gpt-4o").completions_api();
//! ```
use super::{Client, responses_api::streaming::StreamingCompletionResponse};
use super::{ImageUrl, InputAudio, SystemContent};
use crate::completion::CompletionError;
use crate::json_utils;
use crate::message::{AudioMediaType, MessageError, Text};
use crate::one_or_many::string_or_one_or_many;

use crate::{OneOrMany, completion, message};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
    // TODO: Fix this before opening a PR!
    // tool_choice: Option<T>,
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
#[derive(Debug, Deserialize, Serialize, Clone)]
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

/// Message roles. Used by OpenAI Responses API to determine who created a given message.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// The type of content used in an [`InputItem`]. Addtionally holds data for each type of input content.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputContent {
    Message(Message),
    OutputMessage(Message),
    FunctionCall(OutputFunctionCall),
    FunctionCallOutput(ToolResult),
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
            Message::Assistant { .. } => Self {
                role: Some(Role::Assistant),
                input: InputContent::OutputMessage(value),
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
                                        call_id: call_id
                                            .clone()
                                            .expect("The call ID of this tool should exist!"),
                                        output: text,
                                        status: ToolStatus::Completed,
                                    }),
                                });
                            }
                        }
                        _ => {
                            return Err(CompletionError::ProviderError(
                                "This API only supports text and tool results at the moment"
                                    .to_string(),
                            ));
                        }
                    }
                }

                Ok(items)
            }
            crate::completion::Message::Assistant { id, content } => {
                let mut items = Vec::new();

                for assistant_content in content {
                    match assistant_content {
                        crate::message::AssistantContent::Text(Text { text }) => {
                            let id = id.as_ref().unwrap_or(&String::default()).clone();
                            items.push(InputItem {
                                role: Some(Role::Assistant),
                                input: InputContent::OutputMessage(Message::Assistant {
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
                        }) => {
                            items.push(InputItem {
                                role: None,
                                input: InputContent::FunctionCall(OutputFunctionCall {
                                    arguments: function.arguments,
                                    call_id: call_id.expect("The tool call ID should exist!"),
                                    id: tool_id,
                                    name: function.name,
                                    status: ToolStatus::Completed,
                                }),
                            });
                        }
                    }
                }

                Ok(items)
            }
        }
    }
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

        let parameters = parameters
            .as_object_mut()
            .expect("parameters should be a JSON object");
        parameters.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );

        let parameters = serde_json::Value::Object(parameters.clone());

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
        let input = {
            let mut partial_history = vec![];
            if let Some(docs) = req.normalized_documents() {
                partial_history.push(docs);
            }
            partial_history.extend(req.chat_history);

            // Initialize full history with preamble (or empty if non-existent)
            let mut full_history: Vec<InputItem> = Vec::new();

            // Convert and extend the rest of the history
            full_history.extend(
                partial_history
                    .into_iter()
                    .map(|x| <Vec<InputItem>>::try_from(x).unwrap())
                    .collect::<Vec<Vec<InputItem>>>()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<InputItem>>(),
            );

            full_history
        };

        let input = OneOrMany::many(input)
            .expect("This should never panic - if it does, please file a bug report");

        let stream = req
            .additional_params
            .clone()
            .unwrap_or(Value::Null)
            .as_bool();

        let additional_parameters = if let Some(map) = req.additional_params {
            serde_json::from_value::<AdditionalParameters>(map).expect("Converting additional parameters to AdditionalParameters should never fail as every field is an Option")
        } else {
            // If there's no additional parameters, initialise an empty object
            AdditionalParameters::default()
        };

        Ok(Self {
            input,
            model,
            instructions: req.preamble,
            max_output_tokens: req.max_tokens,
            stream,
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
pub struct ResponsesCompletionModel {
    /// The OpenAI client
    pub(crate) client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl ResponsesCompletionModel {
    /// Creates a new [`ResponsesCompletionModel`].
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    /// Use the Completions API instead of Responses.
    pub fn completions_api(self) -> crate::providers::openai::completion::CompletionModel {
        crate::providers::openai::completion::CompletionModel::new(self.client, &self.model)
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
    #[serde(skip_serializing_if = "Map::is_empty")]
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
    Low,
    #[default]
    Medium,
    High,
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
        };

        res
    }
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

impl completion::CompletionModel for ResponsesCompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;
        let request = serde_json::to_value(request)?;

        tracing::debug!("Input: {}", serde_json::to_string_pretty(&request)?);

        let response = self.client.post("/responses").json(&request).send().await?;

        if response.status().is_success() {
            let t = response.text().await?;
            tracing::debug!(target: "rig", "OpenAI response: {}", t);

            let response = serde_json::from_str::<Self::Response>(&t)?;
            response.try_into()
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        Self::stream(self, request).await
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

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
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
}

/// Different types of user content.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    InputText {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
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
                                tool_call_id: call_id.expect("The tool call ID should exist"),
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
                    let other_content = OneOrMany::many(other_content).expect(
                        "There must be other content here if there were no tool result content",
                    );

                    Ok(vec![Message::User {
                        content: other_content.map(|content| match content {
                            message::UserContent::Text(message::Text { text }) => {
                                UserContent::InputText { text }
                            }
                            message::UserContent::Image(message::Image {
                                data, detail, ..
                            }) => UserContent::Image {
                                image_url: ImageUrl {
                                    url: data,
                                    detail: detail.unwrap_or_default(),
                                },
                            },
                            message::UserContent::Document(message::Document { data, .. }) => {
                                UserContent::InputText { text: data }
                            }
                            message::UserContent::Audio(message::Audio {
                                data,
                                media_type,
                                ..
                            }) => UserContent::Audio {
                                input_audio: InputAudio {
                                    data,
                                    format: match media_type {
                                        Some(media_type) => media_type,
                                        None => AudioMediaType::MP3,
                                    },
                                },
                            },
                            _ => unreachable!(),
                        }),
                        name: None,
                    }])
                }
            }
            message::Message::Assistant { content, id } => {
                let assistant_message_id = id;

                match content.first() {
                    crate::message::AssistantContent::Text(Text { text }) => {
                        Ok(vec![Message::Assistant {
                            id: assistant_message_id
                                .expect("The assistant message ID should exist"),
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
                    }) => Ok(vec![Message::Assistant {
                        content: OneOrMany::one(AssistantContentType::ToolCall(
                            OutputFunctionCall {
                                call_id: call_id.expect("The call ID should exist"),
                                arguments: function.arguments,
                                id,
                                name: function.name,
                                status: ToolStatus::Completed,
                            },
                        )),
                        id: assistant_message_id.expect("The assistant message ID should exist!"),
                        name: None,
                        status: ToolStatus::Completed,
                    }]),
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
