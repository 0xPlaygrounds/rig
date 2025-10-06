// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/docs/api-reference#chat-completions)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    http_client, json_utils,
    providers::openai::Message,
};

use super::client::{Client, xai_api_types::ApiResponse};
use crate::completion::CompletionRequest;
use crate::providers::openai;
use crate::streaming::StreamingCompletionResponse;
use serde_json::{Value, json};
use tracing::{Instrument, info_span};
use xai_api_types::{CompletionResponse, ToolDefinition};

/// xAI completion models as of 2025-06-04
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";
pub const GROK_4: &str = "grok-4-0709";

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub(crate) fn create_completion_request(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Convert documents into user message
        let docs: Option<Vec<Message>> = completion_request
            .normalized_documents()
            .map(|docs| docs.try_into())
            .transpose()?;

        // Convert existing chat history
        let chat_history: Vec<Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        // Init full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        // Docs appear right after preamble, if they exist
        if let Some(docs) = docs {
            full_history.extend(docs)
        }

        // Chat history and prompt appear in the order they were provided
        full_history.extend(chat_history);

        let tool_choice = completion_request
            .tool_choice
            .map(crate::providers::openrouter::ToolChoice::try_from)
            .transpose()?;

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
                "tool_choice": tool_choice,
            })
        };

        request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }

    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;
        let request_messages_json_str =
            serde_json::to_string(&request.get("messages").unwrap()).unwrap();

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = &request_messages_json_str,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::debug!("xAI completion request: {request_messages_json_str}");

        async move {
            let response = self
                .client
                .reqwest_post("/v1/chat/completions")
                .json(&request)
                .send()
                .await
                .map_err(|e| CompletionError::HttpError(http_client::Error::Instance(e.into())))?;

            if response.status().is_success() {
                match response
                    .json::<ApiResponse<CompletionResponse>>()
                    .await
                    .map_err(|e| {
                        CompletionError::HttpError(http_client::Error::Instance(e.into()))
                    })? {
                    ApiResponse::Ok(completion) => completion.try_into(),
                    ApiResponse::Error(error) => {
                        Err(CompletionError::ProviderError(error.message()))
                    }
                }
            } else {
                Err(CompletionError::ProviderError(
                    response.text().await.map_err(|e| {
                        CompletionError::HttpError(http_client::Error::Instance(e.into()))
                    })?,
                ))
            }
        }
        .instrument(span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}

pub mod xai_api_types {
    use serde::{Deserialize, Serialize};

    use crate::OneOrMany;
    use crate::completion::{self, CompletionError};
    use crate::providers::openai::{AssistantContent, Message};

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

            let usage = completion::Usage {
                input_tokens: response.usage.prompt_tokens as u64,
                output_tokens: response.usage.completion_tokens as u64,
                total_tokens: response.usage.total_tokens as u64,
            };

            Ok(completion::CompletionResponse {
                choice,
                usage,
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

    #[derive(Debug, Deserialize, Serialize)]
    pub struct CompletionResponse {
        pub id: String,
        pub model: String,
        pub choices: Vec<Choice>,
        pub created: i64,
        pub object: String,
        pub system_fingerprint: String,
        pub usage: Usage,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Choice {
        pub finish_reason: String,
        pub index: i32,
        pub message: Message,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Usage {
        pub completion_tokens: i32,
        pub prompt_tokens: i32,
        pub total_tokens: i32,
    }
}
