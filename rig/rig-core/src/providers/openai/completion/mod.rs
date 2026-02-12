// ================================================================
// OpenAI Completion API
// ================================================================

use super::{
    CompletionsClient as Client,
    client::{ApiErrorResponse, ApiResponse},
    streaming::StreamingCompletionResponse,
};
use crate::completion::{
    CompletionError, CompletionRequest as CoreCompletionRequest, GetTokenUsage,
};
use crate::http_client::{self, HttpClientExt};
use crate::message::{AudioMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::one_or_many::string_or_one_or_many;
use crate::telemetry::{ProviderResponseExt, SpanCombinator};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, completion, json_utils, message};
use serde::{Deserialize, Serialize, Serializer};
use std::convert::Infallible;
use std::fmt;
use tracing::{Instrument, Level, enabled, info_span};

use std::str::FromStr;

pub mod streaming;

/// Serializes user content as a plain string when there's a single text item,
/// otherwise as an array of content parts.
fn serialize_user_content<S>(
    content: &OneOrMany<UserContent>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if content.len() == 1
        && let UserContent::Text { text } = content.first_ref()
    {
        return serializer.serialize_str(text);
    }
    content.serialize(serializer)
}

/// `gpt-5.1` completion model
pub const GPT_5_1: &str = "gpt-5.1";

/// `gpt-5` completion model
pub const GPT_5: &str = "gpt-5";
/// `gpt-5` completion model
pub const GPT_5_MINI: &str = "gpt-5-mini";
/// `gpt-5` completion model
pub const GPT_5_NANO: &str = "gpt-5-nano";

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
        #[serde(
            deserialize_with = "string_or_one_or_many",
            serialize_with = "serialize_user_content"
        )]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(
            default,
            deserialize_with = "json_utils::string_or_vec",
            skip_serializing_if = "Vec::is_empty",
            serialize_with = "serialize_assistant_content_vec"
        )]
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
        content: ToolResultContentValue,
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
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    pub r#type: SystemContentType,
    pub text: String,
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContentValue {
    Array(Vec<ToolResultContent>),
    String(String),
}

impl ToolResultContentValue {
    pub fn from_string(s: String, use_array_format: bool) -> Self {
        if use_array_format {
            ToolResultContentValue::Array(vec![ToolResultContent::from(s)])
        } else {
            ToolResultContentValue::String(s)
        }
    }

    pub fn as_text(&self) -> String {
        match self {
            ToolResultContentValue::Array(arr) => arr
                .iter()
                .map(|c| c.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
            ToolResultContentValue::String(s) => s.clone(),
        }
    }

    pub fn to_array(&self) -> Self {
        match self {
            ToolResultContentValue::Array(_) => self.clone(),
            ToolResultContentValue::String(s) => {
                ToolResultContentValue::Array(vec![ToolResultContent::from(s.clone())])
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

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

/// Function definition for a tool, with optional strict mode
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
                strict: None,
            },
        }
    }
}

impl ToolDefinition {
    /// Apply strict mode to this tool definition.
    /// This sets `strict: true` and sanitizes the schema to meet OpenAI requirements.
    pub fn with_strict(mut self) -> Self {
        self.function.strict = Some(true);
        super::sanitize_schema(&mut self.function.parameters);
        self
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;
    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Specific { .. } => {
                return Err(CompletionError::ProviderError(
                    "Provider doesn't support only using specific tools".to_string(),
                ));
            }
            message::ToolChoice::Auto => Self::Auto,
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Required => Self::Required,
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

impl TryFrom<message::ToolResult> for Message {
    type Error = message::MessageError;

    fn try_from(value: message::ToolResult) -> Result<Self, Self::Error> {
        let text = value
            .content
            .into_iter()
            .map(|content| {
                match content {
                message::ToolResultContent::Text(message::Text { text }) => Ok(text),
                message::ToolResultContent::Image(_) => Err(message::MessageError::ConversionError(
                    "OpenAI does not support images in tool results. Tool results must be text."
                        .into(),
                )),
            }
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");

        Ok(Message::ToolResult {
            tool_call_id: value.id,
            content: ToolResultContentValue::String(text),
        })
    }
}

impl TryFrom<message::UserContent> for UserContent {
    type Error = message::MessageError;

    fn try_from(value: message::UserContent) -> Result<Self, Self::Error> {
        match value {
            message::UserContent::Text(message::Text { text }) => Ok(UserContent::Text { text }),
            message::UserContent::Image(message::Image {
                data,
                detail,
                media_type,
                ..
            }) => match data {
                DocumentSourceKind::Url(url) => Ok(UserContent::Image {
                    image_url: ImageUrl {
                        url,
                        detail: detail.unwrap_or_default(),
                    },
                }),
                DocumentSourceKind::Base64(data) => {
                    let url = format!(
                        "data:{};base64,{}",
                        media_type.map(|i| i.to_mime_type()).ok_or(
                            message::MessageError::ConversionError(
                                "OpenAI Image URI must have media type".into()
                            )
                        )?,
                        data
                    );

                    let detail = detail.ok_or(message::MessageError::ConversionError(
                        "OpenAI image URI must have image detail".into(),
                    ))?;

                    Ok(UserContent::Image {
                        image_url: ImageUrl { url, detail },
                    })
                }
                DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                    "Raw files not supported, encode as base64 first".into(),
                )),
                DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                    "Document has no body".into(),
                )),
                doc => Err(message::MessageError::ConversionError(format!(
                    "Unsupported document type: {doc:?}"
                ))),
            },
            message::UserContent::Document(message::Document { data, .. }) => {
                if let DocumentSourceKind::Base64(text) | DocumentSourceKind::String(text) = data {
                    Ok(UserContent::Text { text })
                } else {
                    Err(message::MessageError::ConversionError(
                        "Documents must be base64 or a string".into(),
                    ))
                }
            }
            message::UserContent::Audio(message::Audio {
                data, media_type, ..
            }) => match data {
                DocumentSourceKind::Base64(data) => Ok(UserContent::Audio {
                    input_audio: InputAudio {
                        data,
                        format: match media_type {
                            Some(media_type) => media_type,
                            None => AudioMediaType::MP3,
                        },
                    },
                }),
                DocumentSourceKind::Url(_) => Err(message::MessageError::ConversionError(
                    "URLs are not supported for audio".into(),
                )),
                DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                    "Raw files are not supported for audio".into(),
                )),
                DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                    "Audio has no body".into(),
                )),
                audio => Err(message::MessageError::ConversionError(format!(
                    "Unsupported audio type: {audio:?}"
                ))),
            },
            message::UserContent::ToolResult(_) => Err(message::MessageError::ConversionError(
                "Tool result is in unsupported format".into(),
            )),
            message::UserContent::Video(_) => Err(message::MessageError::ConversionError(
                "Video is in unsupported format".into(),
            )),
        }
    }
}

impl TryFrom<OneOrMany<message::UserContent>> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(value: OneOrMany<message::UserContent>) -> Result<Self, Self::Error> {
        let (tool_results, other_content): (Vec<_>, Vec<_>) = value
            .into_iter()
            .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

        // If there are messages with both tool results and user content, openai will only
        //  handle tool results. It's unlikely that there will be both.
        if !tool_results.is_empty() {
            tool_results
                .into_iter()
                .map(|content| match content {
                    message::UserContent::ToolResult(tool_result) => tool_result.try_into(),
                    _ => unreachable!(),
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            let other_content: Vec<UserContent> = other_content
                .into_iter()
                .map(|content| content.try_into())
                .collect::<Result<Vec<_>, _>>()?;

            let other_content = OneOrMany::many(other_content)
                .expect("There must be other content here if there were no tool result content");

            Ok(vec![Message::User {
                content: other_content,
                name: None,
            }])
        }
    }
}

impl TryFrom<OneOrMany<message::AssistantContent>> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(value: OneOrMany<message::AssistantContent>) -> Result<Self, Self::Error> {
        let mut text_content = Vec::new();
        let mut tool_calls = Vec::new();

        for content in value {
            match content {
                message::AssistantContent::Text(text) => text_content.push(text),
                message::AssistantContent::ToolCall(tool_call) => tool_calls.push(tool_call),
                message::AssistantContent::Reasoning(_) => {
                    // OpenAI Chat Completions does not support assistant-history reasoning items.
                    // Silently skip unsupported reasoning content.
                }
                message::AssistantContent::Image(_) => {
                    panic!(
                        "The OpenAI Completions API doesn't support image content in assistant messages!"
                    );
                }
            }
        }

        if text_content.is_empty() && tool_calls.is_empty() {
            return Ok(vec![]);
        }

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

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => content.try_into(),
            message::Message::Assistant { content, .. } => content.try_into(),
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
            signature: None,
            additional_params: None,
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
                    OneOrMany::one(message::ToolResultContent::text(content.as_text())),
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
            UserContent::Image { image_url } => {
                message::UserContent::image_url(image_url.url, None, Some(image_url.detail))
            }
            UserContent::Audio { input_audio } => {
                message::UserContent::audio(input_audio.data, Some(input_audio.format))
            }
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

#[derive(Debug, Deserialize, Serialize)]
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

        let usage = response
            .usage
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens as u64)
                    .unwrap_or(0),
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

impl ProviderResponseExt for CompletionResponse {
    type OutputMessage = Choice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.to_owned())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.to_owned())
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        let Message::User { ref content, .. } = self.choices.last()?.message.clone() else {
            return None;
        };

        let UserContent::Text { text } = content.first() else {
            return None;
        };

        Some(text)
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Cached tokens from prompt caching
    #[serde(default)]
    pub cached_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
        }
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Usage {
            prompt_tokens,
            total_tokens,
            ..
        } = self;
        write!(
            f,
            "Prompt tokens: {prompt_tokens} Total tokens: {total_tokens}"
        )
    }
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = (self.total_tokens - self.prompt_tokens) as u64;
        usage.total_tokens = self.total_tokens as u64;
        usage.cached_input_tokens = self
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens as u64)
            .unwrap_or(0);

        Some(usage)
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
}

impl<T> CompletionModel<T>
where
    T: Default + std::fmt::Debug + Clone + 'static,
{
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
        }
    }

    /// Enable strict mode for tool schemas.
    ///
    /// When enabled, tool schemas are automatically sanitized to meet OpenAI's strict mode requirements:
    /// - `additionalProperties: false` is added to all objects
    /// - All properties are marked as required
    /// - `strict: true` is set on each function definition
    ///
    /// This allows OpenAI to guarantee that the model's tool calls will match the schema exactly.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    pub fn with_tool_result_array_content(mut self) -> Self {
        self.tool_result_array_content = true;
        self
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(flatten)]
    additional_params: Option<serde_json::Value>,
}

pub struct OpenAIRequestParams {
    pub model: String,
    pub request: CoreCompletionRequest,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
}

impl TryFrom<OpenAIRequestParams> for CompletionRequest {
    type Error = CompletionError;

    fn try_from(params: OpenAIRequestParams) -> Result<Self, Self::Error> {
        let OpenAIRequestParams {
            model,
            request: req,
            strict_tools,
            tool_result_array_content,
        } = params;

        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        let CoreCompletionRequest {
            model: request_model,
            preamble,
            chat_history,
            tools,
            temperature,
            additional_params,
            tool_choice,
            output_schema,
            ..
        } = req;

        partial_history.extend(chat_history);

        let mut full_history: Vec<Message> =
            preamble.map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        if full_history.is_empty() {
            return Err(CompletionError::RequestError(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "OpenAI Chat Completions request has no provider-compatible messages after conversion",
                )
                .into(),
            ));
        }

        if tool_result_array_content {
            for msg in &mut full_history {
                if let Message::ToolResult { content, .. } = msg {
                    *content = content.to_array();
                }
            }
        }

        let tool_choice = tool_choice.map(ToolChoice::try_from).transpose()?;

        let tools: Vec<ToolDefinition> = tools
            .into_iter()
            .map(|tool| {
                let def = ToolDefinition::from(tool);
                if strict_tools { def.with_strict() } else { def }
            })
            .collect();

        // Map output_schema to OpenAI's response_format and merge into additional_params
        let additional_params = if let Some(schema) = output_schema {
            let name = schema
                .as_object()
                .and_then(|o| o.get("title"))
                .and_then(|v| v.as_str())
                .unwrap_or("response_schema")
                .to_string();
            let mut schema_value = schema.to_value();
            super::sanitize_schema(&mut schema_value);
            let response_format = serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": true,
                        "schema": schema_value
                    }
                }
            });
            Some(match additional_params {
                Some(existing) => json_utils::merge(existing, response_format),
                None => response_format,
            })
        } else {
            additional_params
        };

        let res = Self {
            model: request_model.unwrap_or(model),
            messages: full_history,
            tools,
            tool_choice,
            temperature,
            additional_params,
        };

        Ok(res)
    }
}

impl TryFrom<(String, CoreCompletionRequest)> for CompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (String, CoreCompletionRequest)) -> Result<Self, Self::Error> {
        CompletionRequest::try_from(OpenAIRequestParams {
            model,
            request: req,
            strict_tools: false,
            tool_result_array_content: false,
        })
    }
}

impl crate::telemetry::ProviderRequestExt for CompletionRequest {
    type InputMessage = Message;

    fn get_input_messages(&self) -> Vec<Self::InputMessage> {
        self.messages.clone()
    }

    fn get_system_prompt(&self) -> Option<String> {
        let first_message = self.messages.first()?;

        let Message::System { ref content, .. } = first_message.clone() else {
            return None;
        };

        let SystemContent { text, .. } = content.first();

        Some(text)
    }

    fn get_prompt(&self) -> Option<String> {
        let last_message = self.messages.last()?;

        let Message::User { ref content, .. } = last_message.clone() else {
            return None;
        };

        let UserContent::Text { text } = content.first() else {
            return None;
        };

        Some(text)
    }

    fn get_model_name(&self) -> String {
        self.model.clone()
    }
}

impl CompletionModel<reqwest::Client> {
    pub fn into_agent_builder(self) -> crate::agent::AgentBuilder<Self> {
        crate::agent::AgentBuilder::new(self)
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt
        + Default
        + std::fmt::Debug
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = super::CompletionsClient<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = CompletionRequest::try_from(OpenAIRequestParams {
            model: self.model.to_owned(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Chat Completions completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let text = http_client::text(response).await?;

                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_response_metadata(&response);
                        span.record_token_usage(&response.usage);

                        if enabled!(Level::TRACE) {
                            tracing::trace!(
                                target: "rig::completions",
                                "OpenAI Chat Completions completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        Self::stream(self, request).await
    }
}

fn serialize_assistant_content_vec<S>(
    value: &Vec<AssistantContent>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_empty() {
        serializer.serialize_str("")
    } else {
        value.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_request_uses_request_model_override() {
        let request = crate::completion::CompletionRequest {
            model: Some("gpt-4.1".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
        })
        .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert_eq!(serialized["model"], "gpt-4.1");
    }

    #[test]
    fn test_openai_request_uses_default_model_when_override_unset() {
        let request = crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
        })
        .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert_eq!(serialized["model"], "gpt-4o-mini");
    }

    #[test]
    fn assistant_reasoning_is_silently_skipped() {
        let assistant_content = OneOrMany::one(message::AssistantContent::reasoning("hidden"));

        let converted: Vec<Message> = assistant_content
            .try_into()
            .expect("conversion should work");

        assert!(converted.is_empty());
    }

    #[test]
    fn assistant_text_and_tool_call_are_preserved_when_reasoning_is_present() {
        let assistant_content = OneOrMany::many(vec![
            message::AssistantContent::reasoning("hidden"),
            message::AssistantContent::text("visible"),
            message::AssistantContent::tool_call(
                "call_1",
                "subtract",
                serde_json::json!({"x": 2, "y": 1}),
            ),
        ])
        .expect("non-empty assistant content");

        let converted: Vec<Message> = assistant_content
            .try_into()
            .expect("conversion should work");
        assert_eq!(converted.len(), 1);

        match &converted[0] {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(
                    content,
                    &vec![AssistantContent::Text {
                        text: "visible".to_string()
                    }]
                );
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_1");
                assert_eq!(tool_calls[0].function.name, "subtract");
                assert_eq!(
                    tool_calls[0].function.arguments,
                    serde_json::json!({"x": 2, "y": 1})
                );
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn request_conversion_errors_when_all_messages_are_filtered() {
        let request = CoreCompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(message::Message::Assistant {
                id: None,
                content: OneOrMany::one(message::AssistantContent::reasoning("hidden")),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let result = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
        });

        assert!(matches!(result, Err(CompletionError::RequestError(_))));
    }
}
