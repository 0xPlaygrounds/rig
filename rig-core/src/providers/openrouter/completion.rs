use serde::{Deserialize, Serialize};
use tracing::{Instrument, info_span};

use super::client::{ApiErrorResponse, ApiResponse, Client, Usage};

use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    http_client, json_utils,
    providers::openai::Message,
};
use serde_json::{Value, json};

use crate::providers::openai::AssistantContent;
use crate::providers::openrouter::streaming::FinalCompletionResponse;
use crate::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;

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
#[derive(Debug, Serialize, Deserialize)]
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

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(Vec<ToolChoiceFunctionKind>),
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            crate::message::ToolChoice::None => Self::None,
            crate::message::ToolChoice::Auto => Self::Auto,
            crate::message::ToolChoice::Required => Self::Required,
            crate::message::ToolChoice::Specific { function_names } => {
                let vec: Vec<ToolChoiceFunctionKind> = function_names
                    .into_iter()
                    .map(|name| ToolChoiceFunctionKind::Function { name })
                    .collect();

                Self::Function(vec)
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "function")]
pub enum ToolChoiceFunctionKind {
    Function { name: String },
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    /// Name of the model (e.g.: deepseek-ai/DeepSeek-R1)
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
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

        let tool_choice = completion_request
            .tool_choice
            .map(ToolChoice::try_from)
            .transpose()?;

        let request = json!({
            "model": self.model,
            "messages": full_history,
            "temperature": completion_request.temperature,
            "tools": completion_request.tools.into_iter().map(crate::providers::openai::completion::ToolDefinition::from).collect::<Vec<_>>(),
            "tool_choice": tool_choice,
        });

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = FinalCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completion",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        async move {
            let response = self
                .client
                .reqwest_post("/chat/completions")
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
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response.usage);
                        span.record_model_output(&response.choices);
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model_name", &response.model);

                        tracing::debug!(target: "rig::completion",
                            "OpenRouter response: {response:?}");
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
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
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, completion_request).await
    }
}
