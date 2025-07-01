use serde::Deserialize;

use super::client::{ApiErrorResponse, ApiResponse, Client, Usage};

use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    providers::openai::Message,
};
use serde_json::{Value, json};

use crate::providers::openai::AssistantContent;
use crate::providers::openrouter::streaming::FinalCompletionResponse;
use crate::streaming::StreamingCompletionResponse;

// ================================================================
// OpenRouter Completion API
// ================================================================
/// The `qwen/qwq-32b` model. Find more models at <https://openrouter.ai/models>.
pub const QWEN_QWQ_32B: &str = "qwen/qwq-32b";
/// The `anthropic/claude-3.7-sonnet` model. Find more models at <https://openrouter.ai/models>.
pub const CLAUDE_3_7_SONNET: &str = "anthropic/claude-3.7-sonnet";
/// The `perplexity/sonar-pro` model. Find more models at <https://openrouter.ai/models>.
pub const PERPLEXITY_SONAR_PRO: &str = "perplexity/sonar-pro";
/// The `google/gemini-2.0-flash-001` model. Find more models at <https://openrouter.ai/models>.
pub const GEMINI_FLASH_2_0: &str = "google/gemini-2.0-flash-001";

/// A openrouter completion object.
///
/// For more information, see this link: <https://docs.openrouter.xyz/reference/create_chat_completion_v1_chat_completions_post>
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub system_fingerprint: Option<String>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
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
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Text { text } => completion::AssistantContent::text(text),
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

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    /// Name of the model (e.g.: deepseek-ai/DeepSeek-R1)
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
        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        // Gather docs
        if let Some(docs) = completion_request.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

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

        let request = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(crate::providers::openai::completion::ToolDefinition::from).collect::<Vec<_>>()
        });

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = FinalCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenRouter completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    tracing::debug!(target: "rig",
                        "OpenRouter response: {response:?}");
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
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, completion_request).await
    }
}
