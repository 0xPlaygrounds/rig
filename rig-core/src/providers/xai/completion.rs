// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/api/endpoints#chat-completions)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    json_utils,
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

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        mut completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let mut messages = if let Some(preamble) = &completion_request.preamble {
            vec![completion::Message {
                role: "system".into(),
                content: preamble.clone(),
            }]
        } else {
            vec![]
        };
        messages.append(&mut completion_request.chat_history);

        let prompt_with_context = completion_request.prompt_with_context();

        messages.push(completion::Message {
            role: "user".into(),
            content: prompt_with_context,
        });

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

    impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
        type Error = CompletionError;

        fn try_from(value: CompletionResponse) -> Result<Self, Self::Error> {
            match value.choices.as_slice() {
                [Choice {
                    message:
                        Message {
                            content: Some(content),
                            ..
                        },
                    ..
                }, ..] => Ok(completion::CompletionResponse {
                    choice: completion::ModelChoice::Message(content.to_string()),
                    raw_response: value,
                }),
                [Choice {
                    message:
                        Message {
                            tool_calls: Some(calls),
                            ..
                        },
                    ..
                }, ..] => {
                    let call = calls.first().ok_or(CompletionError::ResponseError(
                        "Tool selection is empty".into(),
                    ))?;

                    Ok(completion::CompletionResponse {
                        choice: completion::ModelChoice::ToolCall(
                            call.function.name.clone(),
                            "".to_owned(),
                            serde_json::from_str(&call.function.arguments)?,
                        ),
                        raw_response: value,
                    })
                }
                _ => Err(CompletionError::ResponseError(
                    "Response did not contain a message or tool call".into(),
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

    #[derive(Debug, Deserialize)]
    pub struct ToolCall {
        pub id: String,
        pub r#type: String,
        pub function: Function,
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
    pub struct Message {
        pub role: String,
        pub content: Option<String>,
        pub tool_calls: Option<Vec<ToolCall>>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Usage {
        pub completion_tokens: i32,
        pub prompt_tokens: i32,
        pub total_tokens: i32,
    }
}
