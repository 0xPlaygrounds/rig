// ================================================================
// OpenAI Completion API
// ================================================================

use super::{ApiErrorResponse, ApiResponse, Client, streaming::StreamingCompletionResponse};
use crate::completion::{CompletionError, CompletionRequest};
use crate::message::{AudioMediaType, ImageDetail};
use crate::one_or_many::string_or_one_or_many;
use crate::{OneOrMany, completion, json_utils, message};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::fmt;

use std::str::FromStr;

pub mod streaming;

/// `o4-mini-2025-04-16` completion model
pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";
/// `o4-mini` completion model
pub const O4_MINI: &str = "o4-mini";
/// `o3` completion model
pub const O3: &str = "o3";
/// `o3-mini` completion model
pub const O3_MINI: &str = "o3-mini";
/// `o3-mini-2025-01-31` completion model
pub const O3_MINI_2025_01_31: &str = "o3-mini-2025-01-31";
/// `o1-pro` completion model
pub const O1_PRO: &str = "o1-pro";
/// `o1`` completion model
pub const O1: &str = "o1";
/// `o1-2024-12-17` completion model
pub const O1_2024_12_17: &str = "o1-2024-12-17";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";

/// `gpt-4.1-mini` completion model
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano` completion model
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-4.1-2025-04-14` completion model
pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
/// `gpt-4.1` completion model
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.5-preview` completion model
pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
/// `gpt-4.5-preview-2025-02-27` completion model
pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
/// `gpt-4o-2024-11-20` completion model (this is newer than 4o)
pub const GPT_4O_2024_11_20: &str = "gpt-4o-2024-11-20";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini` completion model
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

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

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioAssistant {
    id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    r#type: SystemContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    Refusal { refusal: String },
}

impl From<AssistantContent> for completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        match value {
            AssistantContent::Text { text } => completion::AssistantContent::text(text),
            AssistantContent::Refusal { refusal } => completion::AssistantContent::text(refusal),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    Image {
        image_url: ImageUrl,
    },
    Audio {
        input_audio: InputAudio,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolResultContent {
    #[serde(default)]
    r#type: ToolResultContentType,
    pub text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultContentType {
    #[default]
    Text,
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent {
            r#type: ToolResultContentType::default(),
            text: s,
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
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
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
                                ..
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
            message::Message::Assistant { content, .. } => {
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

impl From<ToolCall> for message::ToolCall {
    fn from(tool_call: ToolCall) -> Self {
        Self {
            id: tool_call.id,
            call_id: None,
            function: message::ToolFunction {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message {
            Message::User { content, .. } => message::Message::User {
                content: content.map(|content| content.into()),
            },
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .into_iter()
                    .map(|content| match content {
                        AssistantContent::Text { text } => message::AssistantContent::text(text),

                        // TODO: Currently, refusals are converted into text, but should be
                        //  investigated for generalization.
                        AssistantContent::Refusal { refusal } => {
                            message::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(content).map_err(|_| {
                        message::MessageError::ConversionError(
                            "Neither `content` nor `tool_calls` was provided to the Message"
                                .to_owned(),
                        )
                    })?,
                }
            }

            Message::ToolResult {
                tool_call_id,
                content,
            } => message::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    tool_call_id,
                    content.map(|content| message::ToolResultContent::text(content.text)),
                )),
            },

            // System messages should get stripped out when converting messages, this is just a
            // stop gap to avoid obnoxious error handling or panic occurring.
            Message::System { content, .. } => message::Message::User {
                content: content.map(|content| message::UserContent::text(content.text)),
            },
        })
    }
}

impl From<UserContent> for message::UserContent {
    fn from(content: UserContent) -> Self {
        match content {
            UserContent::Text { text } => message::UserContent::text(text),
            UserContent::Image { image_url } => message::UserContent::image(
                image_url.url,
                Some(message::ContentFormat::default()),
                None,
                Some(image_url.detail),
            ),
            UserContent::Audio { input_audio } => message::UserContent::audio(
                input_audio.data,
                Some(message::ContentFormat::default()),
                Some(input_audio.format),
            ),
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
        AssistantContent::Text { text: s }
    }
}

impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string(),
        })
    }
}
impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Deserialize)]
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
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text } => text,
                            AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Usage {
            prompt_tokens,
            total_tokens,
        } = self;
        write!(
            f,
            "Prompt tokens: {prompt_tokens} Total tokens: {total_tokens}"
        )
    }
}

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    pub fn into_agent_builder(self) -> crate::agent::AgentBuilder<Self> {
        crate::agent::AgentBuilder::new(self)
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // Convert and extend the rest of the history
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
            serde_json::json!({
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

        // only include temperature if it exists
        // because some models don't support temperature
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

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

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
            let t = response.text().await?;
            tracing::debug!(target: "rig", "OpenAI completion error: {}", t);

            match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "OpenAI completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{}", usage.total_tokens)).unwrap_or("N/A".to_string())
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
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        CompletionModel::stream(self, request).await
    }
}
