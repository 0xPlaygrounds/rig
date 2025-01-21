// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/api/endpoints#chat-completions)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    json_utils, message, providers::openai::Message,
};

use serde_json::json;
use xai_api_types::{CompletionResponse, ToolDefinition};

use super::client::{xai_api_types::ApiResponse, Client};

/// `grok-beta` completion model
pub const GROK_BETA: &str = "grok-beta";

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    async fn completion(
        &self,
        mut completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let mut messages: Vec<message::Message> = match &completion_request.preamble {
            Some(preamble) => vec![preamble.as_str().into()],
            None => vec![],
        };
        messages.append(&mut completion_request.chat_history);
        messages.push(completion_request.prompt_with_context());

        // Convert history to open ai message format
        let messages = messages.into_iter().map(Message::from).collect::<Vec<_>>();

        let mut request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": messages,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": messages,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        let response = self
            .client
            .post("/v1/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(completion) => completion.try_into(),
                ApiResponse::Error(error) => Err(CompletionError::ProviderError(error.message())),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}

pub mod xai_api_types {
    use serde::{Deserialize, Serialize};

    use crate::completion::{self, CompletionError};
    use crate::providers::openai::{AssistantContent, AssistantMessage, Message};

    impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
        type Error = CompletionError;

        fn try_from(value: CompletionResponse) -> std::result::Result<Self, Self::Error> {
            match value.choices.as_slice() {
                [Choice {
                    message: Message::Assistant(AssistantMessage::Content { content, .. }),
                    ..
                }, ..] => {
                    let content_str = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text { text } => text.clone(),
                            AssistantContent::Refusal { refusal } => refusal.clone(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    Ok(completion::CompletionResponse {
                        choice: completion::ModelChoice::Message(content_str),
                        raw_response: value,
                    })
                }
                [Choice {
                    message: Message::Assistant(AssistantMessage::ToolCalls { tool_calls, .. }),
                    ..
                }, ..] => {
                    let call = tool_calls.first();
                    Ok(completion::CompletionResponse {
                        choice: completion::ModelChoice::ToolCall(
                            call.function.name.clone(),
                            serde_json::from_str(&call.function.arguments)?,
                        ),
                        raw_response: value,
                    })
                }
                _ => Err(CompletionError::ResponseError(
                    "Response did not contain a valid message or tool call".into(),
                )),
            }
        }
    }

    impl From<completion::ToolDefinition> for ToolDefinition {
        fn from(tool: completion::ToolDefinition) -> Self {
            Self {
                r#type: "function".into(),
                function: tool,
            }
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ToolDefinition {
        pub r#type: String,
        pub function: completion::ToolDefinition,
    }

    #[derive(Debug, Deserialize)]
    pub struct Function {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct CompletionResponse {
        pub id: String,
        pub model: String,
        pub choices: Vec<Choice>,
        pub created: i64,
        pub object: String,
        pub system_fingerprint: String,
        pub usage: Usage,
    }

    #[derive(Debug, Deserialize)]
    pub struct Choice {
        pub finish_reason: String,
        pub index: i32,
        pub message: Message,
    }

    #[derive(Debug, Deserialize)]
    pub struct Usage {
        pub completion_tokens: i32,
        pub prompt_tokens: i32,
        pub total_tokens: i32,
    }
}
