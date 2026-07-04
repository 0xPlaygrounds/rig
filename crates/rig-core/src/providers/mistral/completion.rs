use http::Request;
use serde::{Deserialize, Deserializer, Serialize};
use std::{convert::Infallible, str::FromStr};
use tracing::{Instrument, Level, enabled, info_span};

use super::client::{Client, Usage};
use crate::completion::GetTokenUsage;
use crate::http_client::{self, HttpClientExt};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleToolCallChunk,
};
use crate::streaming::StreamingCompletionResponse;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
    providers::mistral::client::ApiResponse,
    telemetry::SpanCombinator,
};

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `pixtral-large` Mistral multimodal model
pub const PIXTRAL_LARGE: &str = "pixtral-large-latest";
/// The latest version of the `mistral` Mistral multimodal model, trained on datasets from the Middle East & South Asia
pub const MISTRAL_SABA: &str = "mistral-saba-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";
/// The `24-09` version of the `pixtral-small` Mistral multimodal model
pub const PIXTRAL_SMALL: &str = "pixtral-12b-2409";
/// The `open-mistral-nemo` model
pub const MISTRAL_NEMO: &str = "open-mistral-nemo";
/// The `open-mistral-mamba` model
pub const CODESTRAL_MAMBA: &str = "open-codestral-mamba";

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub struct AssistantContent {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
}

fn mistral_content_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => parts
            .into_iter()
            .filter_map(|part| {
                (part.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                    .then(|| part.get("text").and_then(serde_json::Value::as_str))
                    .flatten()
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn deserialize_mistral_content_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<serde_json::Value>::deserialize(deserializer)?
        .map(mistral_content_value_to_text)
        .unwrap_or_default())
}

fn deserialize_optional_mistral_content_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<serde_json::Value>::deserialize(deserializer)?
        .map(mistral_content_value_to_text)
        .filter(|text| !text.is_empty()))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        #[serde(default, deserialize_with = "deserialize_mistral_content_string")]
        content: String,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        prefix: bool,
    },
    System {
        content: String,
    },
    Tool {
        /// The name of the tool that was called
        #[serde(skip_serializing_if = "String::is_empty")]
        name: String,
        /// The content of the tool call
        content: String,
        /// The id of the tool call
        tool_call_id: String,
    },
}

impl Message {
    pub fn user(content: String) -> Self {
        Message::User { content }
    }

    pub fn assistant(content: String, tool_calls: Vec<ToolCall>, prefix: bool) -> Self {
        Message::Assistant {
            content,
            tool_calls,
            prefix,
        }
    }

    pub fn system(content: String) -> Self {
        Message::System { content }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::System { content }]),
            message::Message::User { content } => {
                let mut tool_result_messages = Vec::new();
                let mut other_messages = Vec::new();

                for content_item in content {
                    match content_item {
                        message::UserContent::ToolResult(message::ToolResult {
                            id,
                            call_id,
                            content: tool_content,
                        }) => {
                            let tool_call_id = call_id.unwrap_or(id);
                            let content_text = tool_content
                                .into_iter()
                                .find_map(|content_item| match content_item {
                                    message::ToolResultContent::Text(text) => Some(text.text),
                                    message::ToolResultContent::Image(_) => None,
                                })
                                .unwrap_or_default();
                            tool_result_messages.push(Message::Tool {
                                name: String::new(),
                                content: content_text,
                                tool_call_id,
                            });
                        }
                        message::UserContent::Text(message::Text { text, .. }) => {
                            other_messages.push(Message::User { content: text });
                        }
                        _ => {}
                    }
                }

                tool_result_messages.append(&mut other_messages);
                Ok(tool_result_messages)
            }
            message::Message::Assistant { content, .. } => {
                let mut text_content = Vec::new();
                let mut tool_calls = Vec::new();

                for content in content {
                    match content {
                        message::AssistantContent::Text(text) => text_content.push(text),
                        message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call)
                        }
                        message::AssistantContent::Reasoning(_) => {
                            // Mistral conversion path currently does not support assistant-history
                            // reasoning items. Silently skip to avoid crashing the process.
                        }
                        message::AssistantContent::Image(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Mistral assistant messages do not support image content".into(),
                            ));
                        }
                    }
                }

                if text_content.is_empty() && tool_calls.is_empty() {
                    return Ok(vec![]);
                }

                Ok(vec![Message::Assistant {
                    content: text_content
                        .into_iter()
                        .next()
                        .map(|content| content.text)
                        .unwrap_or_default(),
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                    prefix: false,
                }])
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            id: tool_call.id,
            r#type: ToolType::default(),
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    #[serde(default)]
    r#type: ToolResultContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultContentType {
    #[default]
    Text,
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent {
            r#type: ToolResultContentType::default(),
            text: s,
        }
    }
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text { text: s }
    }
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text {
            text: s.to_string(),
        })
    }
}

impl From<String> for AssistantContent {
    fn from(s: String) -> Self {
        AssistantContent { text: s }
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent {
            text: s.to_string(),
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(ToolChoiceMode),
    Function {
        r#type: ToolChoiceFunctionKind,
        function: ToolChoiceFunction,
    },
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
enum ToolChoiceMode {
    #[default]
    Auto,
    None,
    Any,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ToolChoiceFunctionKind {
    Function,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolChoiceFunction {
    name: String,
}

impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Auto => Self::Mode(ToolChoiceMode::Auto),
            message::ToolChoice::None => Self::Mode(ToolChoiceMode::None),
            message::ToolChoice::Required => Self::Mode(ToolChoiceMode::Any),
            message::ToolChoice::Specific { function_names } => {
                specific_tool_choice(function_names)?
            }
        };

        Ok(res)
    }
}

fn specific_tool_choice(function_names: Vec<String>) -> Result<ToolChoice, CompletionError> {
    let mut names = function_names.into_iter();
    let Some(name) = names.next() else {
        return Err(CompletionError::RequestError(
            "ToolChoice::Specific requires at least one function name".into(),
        ));
    };
    if names.next().is_some() {
        return Err(CompletionError::RequestError(
            "Mistral only supports forcing one specific tool".into(),
        ));
    }

    Ok(ToolChoice::Function {
        r#type: ToolChoiceFunctionKind::Function,
        function: ToolChoiceFunction { name },
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MistralCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for MistralCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble.clone())],
            None => vec![],
        };
        let chat_history: Vec<Message> = chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        if full_history.is_empty() {
            return Err(CompletionError::RequestError(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Mistral request has no provider-compatible messages after conversion",
                )
                .into(),
            ));
        }

        let tool_choice = req
            .tool_choice
            .clone()
            .map(ToolChoice::try_from)
            .transpose()?;

        let additional_params = if let Some(schema) = req.output_schema {
            let name = schema
                .as_object()
                .and_then(|object| object.get("title"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("response_schema")
                .to_string();
            let mut schema_value = schema.to_value();
            crate::providers::openai::sanitize_schema(&mut schema_value);
            let response_format = serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": true,
                        "schema": schema_value,
                    }
                }
            });
            Some(match req.additional_params {
                Some(existing) => json_utils::merge(existing, response_format),
                None => response_format,
            })
        } else {
            req.additional_params
        };

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params,
        })
    }
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type OutputMessage = Choice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.clone())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        let res = self
            .choices
            .iter()
            .filter_map(|choice| match choice.message {
                Message::Assistant { ref content, .. } => {
                    if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    }
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("\n");

        if res.is_empty() { None } else { Some(res) }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = self.completion_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage.cached_input_tokens = self.cached_tokens();
        usage
    }
}

impl GetTokenUsage for CompletionResponse {
    fn token_usage(&self) -> crate::completion::Usage {
        self.usage
            .as_ref()
            .map(GetTokenUsage::token_usage)
            .unwrap_or_default()
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = if content.is_empty() {
                    vec![]
                } else {
                    vec![completion::AssistantContent::text(content.clone())]
                };

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: usage.completion_tokens as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage.cached_tokens(),
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();

        let message_id = response.id.clone();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: Some(message_id),
        })
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Send + Clone + std::fmt::Debug + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = MistralStreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into())
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request =
            MistralCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Mistral completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "mistral",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post("v1/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(request).await?;

            let status = response.status();
            if status.is_success() {
                let text = http_client::text(response).await?;
                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response);
                        span.record_response_metadata(&response);
                        response.try_into()
                    }
                    ApiResponse::Err(err) => {
                        tracing::warn!(message = %err.message, "provider returned an error response");
                        Err(CompletionError::from_http_response(status, text))
                    }
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::from_http_response(status, text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let mut request =
            MistralCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        request.additional_params = Some(json_utils::merge(
            request
                .additional_params
                .unwrap_or_else(|| serde_json::json!({})),
            serde_json::json!({ "stream": true }),
        ));

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Mistral streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("v1/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "mistral",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(
            send_mistral_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

#[derive(Default, Deserialize, Debug)]
struct StreamingFunction {
    name: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_utils::deserialize_json_string_or_value"
    )]
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StreamingToolCall {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    function: StreamingFunction,
}

impl From<&StreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &StreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
    }
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_mistral_content_string"
    )]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum FinishReason {
    ToolCalls,
    Stop,
    Length,
    ModelLength,
    Error,
    #[serde(untagged)]
    Other(String),
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<FinishReason>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct MistralStreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for MistralStreamingCompletionResponse {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        usage.cached_input_tokens = self.usage.cached_tokens();
        usage
    }
}

#[derive(Clone, Copy)]
struct MistralCompatibleProfile;

impl CompatibleStreamProfile for MistralCompatibleProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = MistralStreamingCompletionResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let data = match serde_json::from_str::<StreamingCompletionChunk>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::debug!("Couldn't parse Mistral SSE payload: {:?}", error);
                return Ok(None);
            }
        };

        Ok(Some(
            openai_chat_completions_compatible::normalize_first_choice_chunk(
                data.id,
                data.model,
                data.usage,
                &data.choices,
                |choice| CompatibleChoiceData {
                    finish_reason: match choice.finish_reason {
                        Some(FinishReason::ToolCalls) => CompatibleFinishReason::ToolCalls,
                        _ => CompatibleFinishReason::Other,
                    },
                    text: choice.delta.content.clone(),
                    reasoning: None,
                    tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                        &choice.delta.tool_calls,
                    ),
                    details: Vec::new(),
                },
            ),
        ))
    }

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse {
        MistralStreamingCompletionResponse { usage }
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        true
    }
}

async fn send_mistral_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<StreamingCompletionResponse<MistralStreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_streaming_request(
        http_client,
        req,
        MistralCompatibleProfile,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionRequestBuilder, ToolDefinition as RigToolDefinition};
    use crate::message::ToolChoice as RigToolChoice;
    use crate::test_utils::MockCompletionModel;

    #[test]
    fn test_response_deserialization() {
        //https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
        let json_data = r#"
        {
            "id": "cmpl-e5cc70bb28c444948073e77776eb30ef",
            "object": "chat.completion",
            "model": "mistral-small-latest",
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 34,
                "total_tokens": 50
            },
            "created": 1702256327,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "string",
                        "tool_calls": [
                            {
                                "id": "null",
                                "type": "function",
                                "function": {
                                    "name": "string",
                                    "arguments": "{ }"
                                },
                                "index": 0
                            }
                        ],
                        "prefix": false,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        "#;
        let completion_response = serde_json::from_str::<CompletionResponse>(json_data).unwrap();
        assert_eq!(completion_response.model, MISTRAL_SMALL);

        let CompletionResponse {
            id,
            object,
            created,
            choices,
            usage,
            ..
        } = completion_response;

        assert_eq!(id, "cmpl-e5cc70bb28c444948073e77776eb30ef");

        let usage = usage.unwrap();
        assert_eq!(usage.prompt_tokens, 16);
        assert_eq!(usage.completion_tokens, 34);
        assert_eq!(usage.total_tokens, 50);
        assert_eq!(usage.cached_tokens(), 0);
        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.num_cached_tokens.is_none());
        assert_eq!(object, "chat.completion".to_string());
        assert_eq!(created, 1702256327);
        assert_eq!(choices.len(), 1);
    }

    #[test]
    fn test_usage_deserializes_prompt_tokens_details_cached_tokens() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": { "cached_tokens": 42 }
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            42
        );
        assert_eq!(usage.cached_tokens(), 42);
    }

    #[test]
    fn test_usage_accepts_singular_prompt_token_details_alias() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_token_details": { "cached_tokens": 7 }
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            7
        );
        assert_eq!(usage.cached_tokens(), 7);
    }

    #[test]
    fn test_usage_falls_back_to_num_cached_tokens() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "num_cached_tokens": 13
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.num_cached_tokens, Some(13));
        assert!(usage.prompt_tokens_details.is_none());
        assert_eq!(usage.cached_tokens(), 13);
    }

    #[test]
    fn test_usage_prefers_prompt_tokens_details_over_num_cached_tokens() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "num_cached_tokens": 1,
            "prompt_tokens_details": { "cached_tokens": 99 }
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.cached_tokens(), 99);
    }

    #[test]
    fn test_token_usage_threads_cached_tokens_into_completion_usage() {
        let json = r#"{
            "id": "cmpl-x",
            "object": "chat.completion",
            "model": "mistral-small-latest",
            "created": 1700000000,
            "choices": [{
                "index": 0,
                "message": { "content": "hi", "role": "assistant", "prefix": false },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": { "cached_tokens": 42 }
            }
        }"#;
        let response: CompletionResponse = serde_json::from_str(json).unwrap();
        let usage = response.token_usage();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.cached_input_tokens, 42);
    }

    #[test]
    fn test_assistant_reasoning_is_skipped_in_message_conversion() {
        let assistant = message::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::reasoning("hidden")),
        };

        let converted: Vec<Message> = assistant.try_into().expect("conversion should work");
        assert!(converted.is_empty());
    }

    #[test]
    fn test_tool_result_with_call_id_omits_unstable_name() {
        let message = message::Message::tool_result_with_call_id(
            "runtime_generated_name",
            Some("call_123".to_string()),
            "tool output",
        );

        let converted: Vec<Message> = message.try_into().expect("conversion should work");
        assert_eq!(converted.len(), 1);

        let serialized = serde_json::to_value(&converted[0]).expect("message should serialize");
        assert_eq!(serialized["role"], "tool");
        assert_eq!(serialized["content"], "tool output");
        assert_eq!(serialized["tool_call_id"], "call_123");
        assert!(
            serialized.get("name").is_none(),
            "Mistral tool result names are omitted because Rig does not store a stable tool name separately from tool result ids"
        );
    }

    #[test]
    fn test_assistant_text_and_tool_call_are_preserved_when_reasoning_present() {
        let assistant = message::Message::Assistant {
            id: None,
            content: OneOrMany::many(vec![
                message::AssistantContent::reasoning("hidden"),
                message::AssistantContent::text("visible"),
                message::AssistantContent::tool_call(
                    "call_1",
                    "subtract",
                    serde_json::json!({"x": 2, "y": 1}),
                ),
            ])
            .expect("non-empty assistant content"),
        };

        let converted: Vec<Message> = assistant.try_into().expect("conversion should work");
        assert_eq!(converted.len(), 1);

        match &converted[0] {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content, "visible");
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_1");
                assert_eq!(tool_calls[0].function.name, "subtract");
                assert_eq!(
                    tool_calls[0].function.arguments,
                    serde_json::json!({"x": 2, "y": 1})
                );
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn test_request_serializes_mistral_tool_choice_and_max_tokens() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
            .max_tokens(123)
            .tool(RigToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool_choice(RigToolChoice::Required)
            .build();

        let mistral_request = MistralCompletionRequest::try_from((MISTRAL_SMALL, request))
            .expect("Mistral request should serialize");
        let serialized = serde_json::to_value(mistral_request).expect("request should serialize");

        assert_eq!(serialized["tool_choice"], serde_json::json!("any"));
        assert_eq!(serialized["max_tokens"], 123);
    }

    #[test]
    fn test_request_serializes_specific_tool_choice_as_function_object() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use beta.")
            .tool(RigToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            })
            .tool(RigToolDefinition {
                name: "beta".to_string(),
                description: "Beta tool".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            })
            .tool_choice(RigToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            })
            .build();

        let mistral_request = MistralCompletionRequest::try_from((MISTRAL_SMALL, request))
            .expect("Mistral request should serialize");
        let serialized = serde_json::to_value(mistral_request).expect("request should serialize");

        assert_eq!(
            serialized["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "beta"}})
        );
    }

    #[test]
    fn test_assistant_response_accepts_null_and_array_content() {
        let null_content = r#"{
            "id": "cmpl-null",
            "object": "chat.completion",
            "model": "mistral-small-latest",
            "created": 1700000000,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "tool_123456789",
                        "type": "function",
                        "function": {"name": "alpha", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        }"#;
        let response: CompletionResponse = serde_json::from_str(null_content).unwrap();
        match &response.choices[0].message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content, "");
                assert_eq!(tool_calls.len(), 1);
            }
            _ => panic!("expected assistant message"),
        }

        let array_content = r#"{
            "id": "cmpl-array",
            "object": "chat.completion",
            "model": "mistral-small-latest",
            "created": 1700000000,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": [{"type": "text", "text": "hidden"}]},
                        {"type": "text", "text": "visible"}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        }"#;
        let response: CompletionResponse = serde_json::from_str(array_content).unwrap();
        match &response.choices[0].message {
            Message::Assistant { content, .. } => assert_eq!(content, "visible"),
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn test_request_conversion_errors_when_all_messages_are_filtered() {
        let request = CompletionRequest {
            preamble: None,
            chat_history: OneOrMany::one(message::Message::Assistant {
                id: None,
                content: OneOrMany::one(message::AssistantContent::reasoning("hidden")),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            model: None,
            output_schema: None,
        };

        let result = MistralCompletionRequest::try_from((MISTRAL_SMALL, request));
        assert!(matches!(result, Err(CompletionError::RequestError(_))));
    }
}
