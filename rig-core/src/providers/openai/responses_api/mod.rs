/// The OpenAI Responses API.
use super::{responses_api::streaming::StreamingCompletionResponse, Client};
use super::{ImageUrl, InputAudio, SystemContent};
use crate::completion::CompletionError;
use crate::json_utils;
use crate::message::{AudioMediaType, MessageError, Text};
use crate::one_or_many::string_or_one_or_many;

use crate::{completion, message, OneOrMany};
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
    pub input: OneOrMany<InputItem>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    // TODO: Fix this before opening a PR!
    // tool_choice: Option<T>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesToolDefinition>,
    /// Additional parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub addtl_params: Option<AddtlParams>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InputItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<Role>,
    #[serde(flatten)]
    input: InputContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputContent {
    Message(Message),
    OutputMessage(Message),
    FunctionCall(OutputFunctionCall),
    FunctionCallOutput(ToolResult),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolResult {
    call_id: String,
    output: String,
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

// impl TryFrom<crate::completion::Message> for Vec<InputItem> {
//     type Error = CompletionError;
//     fn try_from(value: crate::completion::Message) -> Result<Self, Self::Error> {
//         match value {
//             crate::completion::Message::User { content } => match content.first() {
//                 crate::message::UserContent::Text(Text { text }) => Ok(InputItem {
//                     role: Some(Role::User),
//                     input: InputContent::Message(Message::User {
//                         content: OneOrMany::one(UserContent::InputText { text }),
//                         name: None,
//                     }),
//                 }),
//                 crate::message::UserContent::ToolResult(
//                     crate::completion::message::ToolResult {
//                         id,
//                         call_id,
//                         content,
//                     },
//                 ) => {
//                     let thing = content.first();
//                     let crate::completion::message::ToolResultContent::Text(Text { text }) = thing
//                     else {
//                         panic!("This thing only supports text!");
//                     };

//                     let output = serde_json::from_str(&text)?;

//                     Ok(InputItem {
//                         role: None,
//                         input: InputContent::FunctionCallOutput(ToolResult {
//                             call_id: call_id.expect("The call ID of this tool should exist!"),
//                             output,
//                             status: ToolStatus::Completed,
//                         }),
//                     })
//                 }
//                 _ => Err(CompletionError::ProviderError(
//                     "This API only supports text and tool results at the moment".to_string(),
//                 )),
//             },
//             crate::completion::Message::Assistant { id, content } => match content.first() {
//                 crate::message::AssistantContent::Text(Text { text }) => {
//                     let id = id.unwrap_or_default();
//                     Ok(InputItem {
//                         role: Some(Role::Assistant),
//                         input: InputContent::OutputMessage(Message::Assistant {
//                             content: OneOrMany::one(AssistantContentType::Text(
//                                 AssistantContent::OutputText(Text { text }),
//                             )),
//                             id,
//                             name: None,
//                             status: ToolStatus::Completed,
//                         }),
//                     })
//                 }
//                 crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
//                     id,
//                     call_id,
//                     function,
//                 }) => Ok(InputItem {
//                     role: None,
//                     input: InputContent::FunctionCall(OutputFunctionCall {
//                         arguments: function.arguments,
//                         call_id: call_id.expect("The tool call ID should exist!"),
//                         id,
//                         name: function.name,
//                         status: ToolStatus::Completed,
//                     }),
//                 }),
//             },
//         }
//     }
// }
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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResponsesToolDefinition {
    pub name: String,
    pub parameters: serde_json::Value,
    pub strict: bool,
    #[serde(rename = "type")]
    pub kind: String,
    pub description: String,
}

impl From<completion::ToolDefinition> for ResponsesToolDefinition {
    fn from(value: completion::ToolDefinition) -> Self {
        let completion::ToolDefinition {
            name,
            parameters,
            description,
        } = value;
        Self {
            name,
            parameters,
            description,
            kind: "function".to_string(),
            strict: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens: u64,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: u64,
}

impl ResponsesUsage {
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
                InputTokensDetails {
                    cached_tokens: lhs.cached_tokens + tokens.cached_tokens,
                }
            } else {
                InputTokensDetails {
                    cached_tokens: lhs.cached_tokens,
                }
            }
        });
        let output_tokens = self.output_tokens + rhs.output_tokens;
        let output_tokens_details = OutputTokensDetails {
            reasoning_tokens: self.output_tokens_details.reasoning_tokens
                + rhs.output_tokens_details.reasoning_tokens,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

impl InputTokensDetails {
    pub(crate) fn new() -> Self {
        Self { cached_tokens: 0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

impl OutputTokensDetails {
    pub(crate) fn new() -> Self {
        Self {
            reasoning_tokens: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IncompleteDetailsReason {
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseObject {
    Response,
}

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

            println!("{partial_history:?}");

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

            println!("{full_history:?}");
            full_history
        };

        let input = OneOrMany::many(input)
            .expect("This should never panic - if it does, please file a bug report");

        let stream = req
            .additional_params
            .clone()
            .unwrap_or(Value::Null)
            .as_bool();

        let addtl_params = if let Some(map) = req.additional_params {
            serde_json::from_value::<AddtlParams>(map).ok()
        } else {
            None
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
            addtl_params,
        })
    }
}

#[derive(Clone)]
pub struct ResponsesCompletionModel {
    pub(crate) client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl ResponsesCompletionModel {
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

    pub(crate) fn create_completion_request(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<CompletionRequest, CompletionError> {
        let req = CompletionRequest::try_from((self.model.clone(), completion_request))?;

        Ok(req)
    }
}

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
    pub error: Option<ResponseError>,
    pub incomplete_details: Option<IncompleteDetailsReason>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u64>,
    pub model: String,
    pub usage: Option<ResponsesUsage>,
    pub output: Vec<Output>,
    pub tools: Vec<ResponsesToolDefinition>,
    #[serde(flatten)]
    pub addtl_params: AddtlParams,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub background: Option<bool>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub text: Option<TextConfig>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub include: Option<Vec<Include>>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub top_p: Option<f64>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub truncation: Option<TruncationStrategy>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub user: Option<String>,
    // #[serde(skip_serializing_if = "Map::is_empty")]
    // pub metadata: serde_json::Map<String, serde_json::Value>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub parallel_tool_calls: Option<bool>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub previous_response_id: Option<String>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub reasoning: Option<Reasoning>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub service_tier: Option<OpenAIServiceTier>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub store: Option<bool>,
}

/// Additional parameters for the completion request type for OpenAI's Response API: <https://platform.openai.com/docs/api-reference/responses/create>
/// Intended to be derived from [`crate::completion::request::CompletionRequest`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AddtlParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<Include>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    Auto,
    #[default]
    Disabled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextConfig {
    pub format: TextFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    JsonSchema(StructuredOutputsInput),
    Text,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredOutputsInput {
    /// The name of your schema.
    pub name: String,
    /// Your required output schema. It is recommended that you use the JsonSchema macro, which you can check out at <https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html>.
    pub schema: serde_json::Value,
    /// Enable strict output. If you are using your AI agent in a data pipeline or another scenario that requires the data to be absolutely fixed to a given schema, it is recommended to set this to true.
    pub strict: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reasoning {
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryLevel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummaryLevel {
    #[default]
    Auto,
    Concise,
    Detailed,
}

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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputFunctionCall {
    pub id: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
    pub call_id: String,
    pub name: String,
    pub status: ToolStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputMessage {
    pub id: String,
    pub role: OutputRole,
    pub status: ResponseStatus,
    pub content: Vec<AssistantContent>,
}

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

        println!("Input: {}", serde_json::to_string_pretty(&request)?);

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

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum AssistantContentType {
    Text(AssistantContent),
    ToolCall(OutputFunctionCall),
}

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
