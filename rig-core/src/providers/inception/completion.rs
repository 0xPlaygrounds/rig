use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::{self, CompletionError},
    message::{self, MessageError},
    OneOrMany,
};

use super::client::Client;

// ================================================================
// Inception Completion API
// ================================================================
/// `mercury-coder-small` completion model
pub const MERCURY_CODER_SMALL: &str = "mercury-coder-small";

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

impl From<Choice> for completion::AssistantContent {
    fn from(choice: Choice) -> Self {
        completion::AssistantContent::from(&choice)
    }
}

impl From<&Choice> for completion::AssistantContent {
    fn from(choice: &Choice) -> Self {
        completion::AssistantContent::Text(completion::message::Text {
            text: choice.message.content.clone(),
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {}\nCompletion tokens: {}\nTotal tokens: {}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

impl TryFrom<message::Message> for Message {
    type Error = MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        Ok(match message {
            message::Message::User { content } => Message {
                role: Role::User,
                content: match content.first() {
                    message::UserContent::Text(message::Text { text }) => text.clone(),
                    _ => {
                        return Err(MessageError::ConversionError(
                            "User message content must be a text message".to_string(),
                        ))
                    }
                },
            },
            message::Message::Assistant { content } => Message {
                role: Role::Assistant,
                content: match content.first() {
                    message::AssistantContent::Text(message::Text { text }) => text.clone(),
                    _ => {
                        return Err(MessageError::ConversionError(
                            "Assistant message content must be a text message".to_string(),
                        ))
                    }
                },
            },
        })
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let content = response.choices.iter().map(Into::into).collect::<Vec<_>>();

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

const MAX_TOKENS: u64 = 8192;

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
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let max_tokens = completion_request.max_tokens.unwrap_or(MAX_TOKENS);

        let prompt_message: Message = completion_request
            .prompt_with_context()
            .try_into()
            .map_err(|e: MessageError| CompletionError::RequestError(e.into()))?;

        let mut messages = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                message
                    .try_into()
                    .map_err(|e: MessageError| CompletionError::RequestError(e.into()))
            })
            .collect::<Result<Vec<Message>, _>>()?;

        messages.push(prompt_message);

        let request = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        });

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let response = response.json::<CompletionResponse>().await?;
            tracing::info!(target: "rig",
                "Inception completion token usage: {}",
                response.usage
            );
            Ok(response.try_into()?)
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }
}
