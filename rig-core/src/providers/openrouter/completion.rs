use serde::{Deserialize, Serialize};

use super::client::{ApiErrorResponse, ApiResponse, Client, Usage};
use crate::message::AudioMediaType;
use crate::one_or_many::string_or_one_or_many;

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
    providers::openai::{AudioAssistant, SystemContent, ToolCall, ToolResultContent, UserContent},
    OneOrMany,
};
use serde_json::{json, Value};

use crate::providers::openai::{AssistantContent, ImageUrl, InputAudio};
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
            "tool_calls": completion_request.tools
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

/// A re-implementation of the OpenAI router for OpenRouter.
/// Differences:
/// - includes `reasoning` field in Assistant message
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: OneOrMany<ToolResultContent>,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) = content
                    .into_iter()
                    .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

                // If there are messages with both tool results and user content, openai will only
                //  handle tool results. It's unlikely that there will be both.
                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            message::UserContent::ToolResult(message::ToolResult {
                                id,
                                content,
                            }) => Ok::<_, message::MessageError>(Message::ToolResult {
                                tool_call_id: id,
                                content: content.try_map(|content| match content {
                                    message::ToolResultContent::Text(message::Text { text }) => {
                                        Ok(text.into())
                                    }
                                    _ => Err(message::MessageError::ConversionError(
                                        "Tool result content does not support non-text".into(),
                                    )),
                                })?,
                            }),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    let other_content = OneOrMany::many(other_content).expect(
                        "There must be other content here if there were no tool result content",
                    );

                    Ok(vec![Message::User {
                        content: other_content.map(|content| match content {
                            message::UserContent::Text(message::Text { text }) => {
                                UserContent::Text { text }
                            }
                            message::UserContent::Image(message::Image {
                                data, detail, ..
                            }) => UserContent::Image {
                                image_url: ImageUrl {
                                    url: data,
                                    detail: detail.unwrap_or_default(),
                                },
                            },
                            message::UserContent::Document(message::Document { data, .. }) => {
                                UserContent::Text { text: data }
                            }
                            message::UserContent::Audio(message::Audio {
                                data,
                                media_type,
                                ..
                            }) => UserContent::Audio {
                                input_audio: InputAudio {
                                    data,
                                    format: match media_type {
                                        Some(media_type) => media_type,
                                        None => AudioMediaType::MP3,
                                    },
                                },
                            },
                            _ => unreachable!(),
                        }),
                        name: None,
                    }])
                }
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

                // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
                //  so either `content` or `tool_calls` will have some content.
                Ok(vec![Message::Assistant {
                    content: text_content
                        .into_iter()
                        .map(|content| content.text.into())
                        .collect::<Vec<_>>(),
                    refusal: None,
                    reasoning: None,
                    audio: None,
                    name: None,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
}
