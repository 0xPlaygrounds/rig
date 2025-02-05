// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/api/endpoints#chat-completions)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    json_utils,
    providers::openai::Message,
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
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        // Convert prompt to user message
        let prompt: Vec<Message> = completion_request.prompt_with_context().try_into()?;

        // Convert existing chat history
        let chat_history: Vec<Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        // Combine all messages into a single history
        full_history.extend(chat_history);
        full_history.extend(prompt);

        let mut request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
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
    use crate::providers::openai::{AssistantContent, Message};
    use crate::OneOrMany;

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
                    let mut content = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text { text } => {
                                completion::AssistantContent::text(text)
                            }
                            AssistantContent::Refusal { refusal } => {
                                completion::AssistantContent::text(refusal)
                            }
                        })
                        .collect::<Vec<_>>();

                    content.extend(
                        tool_calls
                            .iter()
                            .map(|call| {
                                completion::AssistantContent::tool_call(
                                    &call.function.name,
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
