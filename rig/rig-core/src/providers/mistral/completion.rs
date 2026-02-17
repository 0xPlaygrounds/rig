use async_stream::stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};
use tracing::{Instrument, Level, enabled, info_span};

use super::client::{Client, Usage};
use crate::completion::GetTokenUsage;
use crate::http_client::{self, HttpClientExt};
use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse};
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
                            let call_id_key = call_id.unwrap_or_else(|| id.clone());
                            let content_text = tool_content
                                .into_iter()
                                .find_map(|content_item| match content_item {
                                    message::ToolResultContent::Text(text) => Some(text.text),
                                    message::ToolResultContent::Image(_) => None,
                                })
                                .unwrap_or_default();
                            tool_result_messages.push(Message::Tool {
                                name: id,
                                content: content_text,
                                tool_call_id: call_id_key,
                            });
                        }
                        message::UserContent::Text(message::Text { text }) => {
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
                            panic!("Image content is not currently supported on Mistral via Rig");
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

#[derive(Debug, Default, Serialize, Deserialize)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Any,
}

impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Auto => Self::Auto,
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Required => Self::Any,
            message::ToolChoice::Specific { .. } => {
                return Err(CompletionError::ProviderError(
                    "Mistral doesn't support requiring specific tools to be called".to_string(),
                ));
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MistralCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for MistralCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for Mistral");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble.clone())],
            None => vec![],
        };
        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<Message> = req
            .chat_history
            .clone()
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
            .map(crate::providers::openai::completion::ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
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

impl GetTokenUsage for CompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let api_usage = self.usage.clone()?;

        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = api_usage.prompt_tokens as u64;
        usage.output_tokens = api_usage.completion_tokens as u64;
        usage.total_tokens = api_usage.total_tokens as u64;

        Some(usage)
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
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

fn assistant_content_to_streaming_choice(
    content: message::AssistantContent,
) -> Option<RawStreamingChoice<CompletionResponse>> {
    match content {
        message::AssistantContent::Text(t) => Some(RawStreamingChoice::Message(t.text)),
        message::AssistantContent::ToolCall(tc) => Some(RawStreamingChoice::ToolCall(
            RawStreamingToolCall::new(tc.id, tc.function.name, tc.function.arguments),
        )),
        message::AssistantContent::Reasoning(_) => None,
        message::AssistantContent::Image(_) => {
            panic!("Image content is not supported on Mistral via Rig")
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Send + Clone + std::fmt::Debug + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = CompletionResponse;

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

            if response.status().is_success() {
                let text = http_client::text(response).await?;
                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response);
                        span.record_response_metadata(&response);
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
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
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let resp = self.completion(request).await?;

        let stream = stream! {
            for c in resp.choice.clone() {
                if let Some(choice) = assistant_content_to_streaming_choice(c) {
                    yield Ok(choice);
                }
            }

            yield Ok(RawStreamingChoice::FinalResponse(resp.raw_response.clone()));
        };

        Ok(StreamingCompletionResponse::stream(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let Usage {
            completion_tokens,
            prompt_tokens,
            total_tokens,
        } = usage.unwrap();

        assert_eq!(prompt_tokens, 16);
        assert_eq!(completion_tokens, 34);
        assert_eq!(total_tokens, 50);
        assert_eq!(object, "chat.completion".to_string());
        assert_eq!(created, 1702256327);
        assert_eq!(choices.len(), 1);
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
    fn test_streaming_choice_mapping_skips_reasoning_and_preserves_other_content() {
        assert!(
            assistant_content_to_streaming_choice(message::AssistantContent::reasoning("hidden"))
                .is_none()
        );

        let text_choice =
            assistant_content_to_streaming_choice(message::AssistantContent::text("visible"))
                .expect("text should be preserved");
        match text_choice {
            RawStreamingChoice::Message(text) => assert_eq!(text, "visible"),
            _ => panic!("expected text streaming choice"),
        }

        let tool_choice =
            assistant_content_to_streaming_choice(message::AssistantContent::tool_call(
                "call_2",
                "add",
                serde_json::json!({"x": 2, "y": 3}),
            ))
            .expect("tool call should be preserved");
        match tool_choice {
            RawStreamingChoice::ToolCall(call) => {
                assert_eq!(call.id, "call_2");
                assert_eq!(call.name, "add");
                assert_eq!(call.arguments, serde_json::json!({"x": 2, "y": 3}));
            }
            _ => panic!("expected tool-call streaming choice"),
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
