use async_stream::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{convert::Infallible, str::FromStr};

use super::client::{Client, Usage};
use crate::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
    providers::mistral::client::ApiResponse,
    OneOrMany,
};

pub const CODESTRAL: &str = "codestral-latest";
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
pub const PIXTRAL_LARGE: &str = "pixtral-large-latest";
pub const MISTRAL_SABA: &str = "mistral-saba-latest";
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

//Free models
pub const MISTRAL_SMALL: &str = "mistral-small-latest";
pub const PIXTRAL_SMALL: &str = "pixtral-12b-2409";
pub const MISTRAL_NEMO: &str = "open-mistral-nemo";
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
                let (_, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                let messages = other_content
                    .into_iter()
                    .filter_map(|content| match content {
                        message::UserContent::Text(message::Text { text }) => {
                            Some(Message::User { content: text })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                Ok(messages)
            }
            message::Message::Assistant { content } => {
                let (text_content, tool_calls) = content.into_iter().fold(
                    (Vec::new(), Vec::new()),
                    |(mut texts, mut tools), content| {
                        match content {
                            message::AssistantContent::Text(text) => texts.push(text),
                            message::AssistantContent::ToolCall(tool_call) => tools.push(tool_call),
                        }
                        (texts, tools)
                    },
                );

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
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }

        partial_history.extend(completion_request.chat_history);

        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble.clone())],
            None => vec![],
        };

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,

            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        let request = if let Some(temperature) = completion_request.temperature {
            json_utils::merge(
                request,
                json!({
                    "temperature": temperature,
                }),
            )
        } else {
            request
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
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

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("v1/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let text = response.text().await?;
            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                ApiResponse::Ok(response) => {
                    tracing::debug!(target: "rig",
                        "Mistral completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let resp = self.completion(request).await?;

        let stream = Box::pin(stream! {
            for c in resp.choice.clone() {
                match c {
                    message::AssistantContent::Text(t) => {
                        yield Ok(RawStreamingChoice::Message(t.text.clone()))
                    }
                    message::AssistantContent::ToolCall(tc) => {
                        yield Ok(RawStreamingChoice::ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        })
                    }
                }
            }

            yield Ok(RawStreamingChoice::FinalResponse(resp.raw_response.clone()));
        });

        Ok(StreamingCompletionResponse::stream(stream))
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
}
