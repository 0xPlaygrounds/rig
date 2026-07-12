// ================================================================
// OpenAI Completion API
// ================================================================

use super::{client::ApiResponse, streaming::StreamingCompletionResponse};
use crate::completion::{
    CompletionError, CompletionRequest as CoreCompletionRequest, GetTokenUsage,
};
use crate::http_client::{self, HttpClientExt};
use crate::message::{AudioMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::one_or_many::string_or_one_or_many;
use crate::telemetry::{
    CompletionOperation, CompletionSpanBuilder, ProviderResponseExt, SpanCombinator,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, completion, json_utils, message};
use serde::{Deserialize, Serialize, Serializer};
use std::convert::Infallible;
use std::fmt;
use tracing::{Instrument, Level, enabled};

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
        && let UserContent::Text { text, .. } = content.first_ref()
    {
        return serializer.serialize_str(text);
    }
    content.serialize(serializer)
}

/// `gpt-5.6` completion model (alias that routes to GPT-5.6 Sol)
pub const GPT_5_6: &str = "gpt-5.6";

/// `gpt-5.6-sol` completion model
pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";

/// `gpt-5.6-terra` completion model
pub const GPT_5_6_TERRA: &str = "gpt-5.6-terra";

/// `gpt-5.6-luna` completion model
pub const GPT_5_6_LUNA: &str = "gpt-5.6-luna";

/// `gpt-5.5` completion model
pub const GPT_5_5: &str = "gpt-5.5";

/// `gpt-5.2` completion model
pub const GPT_5_2: &str = "gpt-5.2";

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
    // Gemini-backed OpenAI-compatible gateways (e.g. OpenRouter) can answer
    // with `role: "model"`; accept it on deserialization.
    #[serde(alias = "model")]
    Assistant {
        #[serde(
            default,
            deserialize_with = "json_utils::string_or_vec",
            skip_serializing_if = "Vec::is_empty",
            serialize_with = "serialize_assistant_content_vec"
        )]
        content: Vec<AssistantContent>,
        // OpenAI-compatible providers expose hidden reasoning on this non-standard
        // field, and some require it to be echoed back on assistant tool-call turns.
        // Serialized as `reasoning_content` (llama.cpp/DeepSeek dialect); the
        // `reasoning` alias accepts OpenRouter responses.
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "reasoning_content",
            alias = "reasoning"
        )]
        reasoning: Option<String>,
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
        /// Structured reasoning blocks used by OpenAI-compatible providers
        /// such as OpenRouter. Empty (and omitted from the wire) for
        /// providers that do not emit or accept them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning_details: Vec<ReasoningDetails>,
        /// Generated images returned by image-generation models (OpenRouter's
        /// sibling `images` array). Inbound only — never serialized back into
        /// a request.
        #[serde(default, skip_serializing)]
        images: Vec<ResponseImage>,
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

fn history_contains_tool_result(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|message| matches!(message, Message::ToolResult { .. }))
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AudioAssistant {
    pub id: String,
}

/// Structured reasoning blocks attached to assistant messages by
/// OpenAI-compatible providers such as OpenRouter (`reasoning_details`).
///
/// The `Option` fields are intentionally serialized even when `None`
/// (`"format":null,"id":null`) to match the provider wire format.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningDetails {
    #[serde(rename = "reasoning.summary")]
    Summary {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        summary: String,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        data: String,
    },
    #[serde(rename = "reasoning.text")]
    Text {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        text: Option<String>,
        signature: Option<String>,
    },
}

/// An image emitted by an image-generation model. OpenRouter returns generated
/// images out-of-band from `content`, as a sibling `images` array on the
/// assistant message. Each entry mirrors the request-side `image_url` content
/// part structure.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ResponseImage {
    pub image_url: ImageUrl,
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
            AssistantContent::Text { text, .. } => completion::AssistantContent::text(text),
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
    /// Audio content part. Serialized with OpenAI's `input_audio` wire tag;
    /// the legacy `audio` tag is still accepted on deserialization.
    #[serde(rename = "input_audio", alias = "audio")]
    Audio {
        input_audio: InputAudio,
    },
    /// File content part for documents such as PDFs.
    ///
    /// Maps to OpenAI's `{"type":"file","file":{...}}` content type. Either
    /// `file_data` (a base64 data URI like `data:application/pdf;base64,...`)
    /// or `file_id` (a previously uploaded file reference) must be set.
    File {
        file: FileData,
    },
    /// Video content part (URL or base64 data URI), used by OpenAI-compatible
    /// providers such as OpenRouter. Wire tag: `video_url`.
    #[serde(rename = "video_url")]
    Video {
        video_url: VideoUrl,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    /// Image detail level. Optional so that providers whose wire format omits
    /// it (e.g. OpenRouter) can leave the key out entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Video payload for [`UserContent::Video`].
///
/// `url` is either a publicly accessible URL or a base64 data URI
/// (e.g. `data:video/mp4;base64,...`).
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioMediaType,
}

/// File payload for [`UserContent::File`].
///
/// At least one of `file_data` or `file_id` must be set for the content part
/// to be accepted by OpenAI's chat completions API. `filename` is optional
/// but recommended.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct FileData {
    /// Inline file data as a base64 data URI, e.g.
    /// `data:application/pdf;base64,JVBERi0xLjQK...`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    /// Identifier of a previously uploaded file (OpenAI Files API).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// Display name of the file. Recommended for inline `file_data`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
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

/// Native function tools and provider-hosted tool declarations share one wire array.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ChatToolDefinition {
    Function(ToolDefinition),
    Hosted(serde_json::Value),
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

#[derive(Default, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    /// Force the model to call one specific function:
    /// `{"type": "function", "function": {"name": "..."}}`.
    Function {
        name: String,
    },
}

#[derive(Deserialize, Serialize)]
struct ToolChoiceFunctionName {
    name: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolChoiceFunctionRepr {
    Function { function: ToolChoiceFunctionName },
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => ToolChoiceFunctionRepr::Function {
                function: ToolChoiceFunctionName { name: name.clone() },
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Mode(String),
            Function(ToolChoiceFunctionRepr),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Mode(mode) => match mode.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(serde::de::Error::custom(format!(
                    "unknown tool_choice mode {other:?}"
                ))),
            },
            Repr::Function(ToolChoiceFunctionRepr::Function {
                function: ToolChoiceFunctionName { name },
            }) => Ok(Self::Function { name }),
        }
    }
}

impl ToolChoice {
    /// Force a call to the named function.
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function { name: name.into() }
    }
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;
    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Specific { function_names } => {
                let [name] = function_names.as_slice() else {
                    return Err(CompletionError::ProviderError(
                        "Provider only supports forcing exactly one specific tool".to_string(),
                    ));
                };
                Self::function(name)
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
    #[serde(
        serialize_with = "json_utils::stringified_json::serialize",
        deserialize_with = "json_utils::stringified_json::deserialize_maybe_stringified"
    )]
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
                message::ToolResultContent::Text(message::Text { text, .. }) => Ok(text),
                message::ToolResultContent::Image(_) => Err(message::MessageError::ConversionError(
                    "OpenAI does not support images in tool results. Tool results must be text."
                        .into(),
                )),
            }
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");

        Ok(Message::ToolResult {
            // `call_id` carries the provider-issued call id when it differs
            // from the rig-level tool-result id (e.g. Mistral, llama.cpp).
            tool_call_id: value.call_id.unwrap_or(value.id),
            content: ToolResultContentValue::String(text),
        })
    }
}

impl TryFrom<message::UserContent> for UserContent {
    type Error = message::MessageError;

    fn try_from(value: message::UserContent) -> Result<Self, Self::Error> {
        match value {
            message::UserContent::Text(message::Text { text, .. }) => Ok(UserContent::Text { text }),
            message::UserContent::Image(message::Image {
                data,
                detail,
                media_type,
                ..
            }) => match data {
                DocumentSourceKind::Url(url) => Ok(UserContent::Image {
                    image_url: ImageUrl {
                        url,
                        // OpenAI's wire format always carries a detail level;
                        // absent rig-level detail maps to the default (auto).
                        detail: Some(detail.unwrap_or_default()),
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

                    let detail = Some(detail.unwrap_or_default());

                    Ok(UserContent::Image {
                        image_url: ImageUrl { url, detail },
                    })
                }
                DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                    "Raw files not supported, encode as base64 first".into(),
                )),
                DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
                    "File IDs are not supported for images".into(),
                )),
                DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                    "Document has no body".into(),
                )),
                doc => Err(message::MessageError::ConversionError(format!(
                    "Unsupported document type: {doc:?}"
                ))),
            },
            message::UserContent::Document(message::Document {
                data: DocumentSourceKind::FileId(file_id),
                ..
            }) => Ok(UserContent::File {
                file: FileData {
                    file_data: None,
                    file_id: Some(file_id),
                    filename: None,
                },
            }),
            message::UserContent::Document(message::Document {
                data,
                media_type: Some(message::DocumentMediaType::PDF),
                ..
            }) => match data {
                DocumentSourceKind::Base64(b64) => Ok(UserContent::File {
                    file: FileData {
                        file_data: Some(format!("data:application/pdf;base64,{b64}")),
                        file_id: None,
                        filename: Some("document.pdf".to_string()),
                    },
                }),
                DocumentSourceKind::Url(_) => Err(message::MessageError::ConversionError(
                    "OpenAI chat completions does not accept URL files; use the Responses API or pass base64-encoded bytes".into(),
                )),
                DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                    "Raw files not supported, encode as base64 first".into(),
                )),
                DocumentSourceKind::String(_) => Err(message::MessageError::ConversionError(
                    "PDF documents must be base64-encoded, not raw strings".into(),
                )),
                DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
                    "File ID documents should be converted without media type constraints".into(),
                )),
                DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                    "Document has no body".into(),
                )),
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
                DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
                    "File IDs are not supported for audio".into(),
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
            message::UserContent::Video(message::Video {
                data, media_type, ..
            }) => {
                let url = match data {
                    DocumentSourceKind::Url(url) => url,
                    DocumentSourceKind::Base64(data) => {
                        let mime = media_type
                            .ok_or_else(|| {
                                message::MessageError::ConversionError(
                                    "Video media type required for base64 encoding".into(),
                                )
                            })?
                            .to_mime_type();
                        format!("data:{mime};base64,{data}")
                    }
                    DocumentSourceKind::Raw(_) => {
                        return Err(message::MessageError::ConversionError(
                            "Raw bytes not supported for video, encode as base64 first".into(),
                        ));
                    }
                    DocumentSourceKind::FileId(_) => {
                        return Err(message::MessageError::ConversionError(
                            "File IDs are not supported for video".into(),
                        ));
                    }
                    DocumentSourceKind::String(_) => {
                        return Err(message::MessageError::ConversionError(
                            "String source not supported for video".into(),
                        ));
                    }
                    DocumentSourceKind::Unknown => {
                        return Err(message::MessageError::ConversionError(
                            "Video has no data".into(),
                        ));
                    }
                };
                Ok(UserContent::Video {
                    video_url: VideoUrl { url },
                })
            }
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
                    _ => Err(message::MessageError::ConversionError(
                        "expected tool result content while converting OpenAI input".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            let other_content: Vec<UserContent> = other_content
                .into_iter()
                .map(|content| content.try_into())
                .collect::<Result<Vec<_>, _>>()?;

            let other_content = OneOrMany::many(other_content).map_err(|_| {
                message::MessageError::ConversionError(
                    "OpenAI user message did not contain any non-tool content".into(),
                )
            })?;

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
        // Distinct reasoning blocks are joined with a newline (matching
        // `display_text()`'s own inter-block separator) rather than glued
        // together, so replayed multi-block reasoning keeps its boundaries.
        let mut reasoning_parts: Vec<String> = Vec::new();

        for content in value {
            match content {
                message::AssistantContent::Text(text) => text_content.push(text),
                message::AssistantContent::ToolCall(tool_call) => tool_calls.push(tool_call),
                message::AssistantContent::Reasoning(reasoning) => {
                    let display = reasoning.display_text();
                    if !display.is_empty() {
                        reasoning_parts.push(display);
                    }
                }
                message::AssistantContent::Image(_) => {
                    return Err(message::MessageError::ConversionError(
                        "OpenAI assistant messages do not support image content in chat completions"
                            .into(),
                    ));
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
            reasoning: if reasoning_parts.is_empty() {
                None
            } else {
                Some(reasoning_parts.join("\n"))
            },
            refusal: None,
            audio: None,
            name: None,
            tool_calls: tool_calls
                .into_iter()
                .map(|tool_call| tool_call.into())
                .collect::<Vec<_>>(),
            reasoning_details: Vec::new(),
            images: Vec::new(),
        }])
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::system(&content)]),
            message::Message::User { content } => content.try_into(),
            message::Message::Assistant { content, .. } => content.try_into(),
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            // Keep the assistant echo consistent with the tool-result side,
            // which prefers the provider-issued `call_id` over the rig-level
            // id (e.g. Responses-API history replayed via chat completions).
            id: tool_call.call_id.unwrap_or(tool_call.id),
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
            internal_call_id: None,
            parent_internal_call_id: None,
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
                reasoning,
                ..
            } => {
                let mut assistant_content = Vec::new();

                if let Some(reasoning) = reasoning
                    && !reasoning.is_empty()
                {
                    assistant_content.push(message::AssistantContent::reasoning(reasoning));
                }

                assistant_content.extend(content.into_iter().map(|content| match content {
                    AssistantContent::Text { text, .. } => message::AssistantContent::text(text),
                    AssistantContent::Refusal { refusal } => {
                        message::AssistantContent::text(refusal)
                    }
                }));

                assistant_content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(assistant_content).map_err(|_| {
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
            UserContent::Text { text, .. } => message::UserContent::text(text),
            UserContent::Image { image_url } => {
                message::UserContent::image_url(image_url.url, None, image_url.detail)
            }
            UserContent::Audio { input_audio } => {
                message::UserContent::audio(input_audio.data, Some(input_audio.format))
            }
            UserContent::File {
                file: FileData {
                    file_data, file_id, ..
                },
            } => match file_data {
                Some(data_url) => {
                    let kind = match data_url.strip_prefix("data:application/pdf;base64,") {
                        Some(b64) => DocumentSourceKind::Base64(b64.to_string()),
                        None => DocumentSourceKind::String(data_url),
                    };
                    message::UserContent::Document(message::Document {
                        data: kind,
                        media_type: Some(message::DocumentMediaType::PDF),
                        additional_params: None,
                    })
                }
                None => match file_id {
                    Some(id) => message::UserContent::Document(message::Document {
                        data: DocumentSourceKind::FileId(id),
                        media_type: None,
                        additional_params: None,
                    }),
                    None => message::UserContent::text(String::new()),
                },
            },
            UserContent::Video { video_url } => {
                let decomposed = video_url
                    .url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                    .and_then(|(mime, data)| {
                        // Only decompose data URIs whose media type survives
                        // the round trip; unrecognized MIMEs (e.g.
                        // video/quicktime, parameterized types) stay as URLs
                        // so re-serialization reproduces the original URI.
                        crate::message::VideoMediaType::from_mime_type(mime)
                            .map(|media_type| (media_type, data))
                    });
                match decomposed {
                    Some((media_type, data)) => message::UserContent::video(data, Some(media_type)),
                    None => message::UserContent::video_url(video_url.url, None),
                }
            }
        }
    }
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text { text: s }
    }
}

impl From<&str> for UserContent {
    fn from(s: &str) -> Self {
        UserContent::Text {
            text: s.to_string(),
        }
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
    // Defaulted on deserialization: some OpenAI-compatible gateways
    // (HuggingFace router sub-providers, TGI variants) omit them.
    #[serde(default)]
    pub object: String,
    #[serde(default)]
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
                reasoning,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text, .. } => text,
                            AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
                        }
                    })
                    .collect::<Vec<_>>();

                if let Some(reasoning) = reasoning {
                    // llama.cpp exposes hidden reasoning on a separate non-standard field.
                    // Keep it structured here so the non-streaming path matches streaming
                    // behavior and does not pollute plain-text response surfaces.
                    content.push(completion::AssistantContent::reasoning(reasoning));
                }

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
            .map(GetTokenUsage::token_usage)
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            finish_reason: None,
            raw_finish_reason: None,
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
        let response = self
            .choices
            .iter()
            .filter_map(|choice| assistant_message_text_response(&choice.message))
            .collect::<Vec<_>>()
            .join("\n");

        if response.is_empty() {
            None
        } else {
            Some(response)
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

fn assistant_message_text_response(message: &Message) -> Option<String> {
    let Message::Assistant {
        content, refusal, ..
    } = message
    else {
        return None;
    };

    let mut segments = content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text { text, .. } => (!text.is_empty()).then(|| text.clone()),
            AssistantContent::Refusal { refusal } => (!refusal.is_empty()).then(|| refusal.clone()),
        })
        .collect::<Vec<_>>();

    if segments.is_empty()
        && let Some(refusal) = refusal.as_ref().filter(|refusal| !refusal.is_empty())
    {
        segments.push(refusal.clone());
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments.join("\n"))
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

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens reported by reasoning-capable providers.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f64>,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: None,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            queue_time: None,
            prompt_time: None,
            completion_time: None,
            total_time: None,
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
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::providers::internal::completion_usage(
            self.prompt_tokens as u64,
            self.completion_tokens
                .unwrap_or_else(|| self.total_tokens.saturating_sub(self.prompt_tokens))
                as u64,
            self.total_tokens as u64,
            self.prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens as u64)
                .unwrap_or(0),
        );
        usage.reasoning_tokens = self
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens as u64)
            .unwrap_or(0);
        usage
    }
}

/// Per-model options that affect request conversion/finalization for the shared
/// OpenAI-compatible chat-completions path.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompletionModelOptions {
    /// Whether tool schemas should be sanitized for strict-mode validation.
    pub strict_tools: bool,
    /// Whether tool-result messages should serialize their content as arrays.
    pub tool_result_array_content: bool,
    /// Whether the model requested provider-specific prompt caching markers.
    pub prompt_caching: bool,
}

/// Contract for provider extensions that speak the OpenAI Chat Completions wire
/// format through [`GenericCompletionModel`]. Mirrors
/// [`AnthropicCompatibleProvider`](crate::providers::anthropic::completion::AnthropicCompatibleProvider)
/// on the Anthropic-compatible side.
///
/// Request construction runs the hooks in a fixed order:
/// [`prepare_request`](Self::prepare_request) on the typed request, then
/// serialization, then (for streaming) the `stream`/`stream_options` merge,
/// and finally
/// [`finalize_request_body_with_options`](Self::finalize_request_body_with_options)
/// on the serialized body — so the finalize hook always sees the streaming
/// parameters and model-level options.
pub trait OpenAICompatibleProvider: crate::client::Provider {
    /// Provider name recorded on `gen_ai.provider.name` telemetry spans.
    const PROVIDER_NAME: &'static str;

    /// Whether the backend can emit a whole tool call (id, name, and complete
    /// arguments) in a single streaming chunk, as llama.cpp-based servers do.
    /// When true, the shared streaming layer emits such calls as soon as they
    /// arrive instead of holding them until the stream ends.
    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = false;

    /// Whether the provider supports tool calling. When false, `tools` and
    /// `tool_choice` are dropped with a warning during request conversion —
    /// before tool-choice validation, so unsupported tool configurations
    /// never error client-side on a provider that ignores tools anyway.
    const SUPPORTS_TOOLS: bool = true;

    /// Whether `output_schema` maps to OpenAI's `response_format`. Providers
    /// whose APIs reject `json_schema` response formats set this to false;
    /// the schema is then dropped with a warning instead of being sent.
    const SUPPORTS_RESPONSE_FORMAT: bool = true;

    /// Whether streaming requests include
    /// `"stream_options": {"include_usage": true}`. Providers that reject
    /// unknown parameters and already report usage on the final chunk set
    /// this to false.
    const STREAM_INCLUDE_USAGE: bool = true;

    /// The usage payload parsed from streaming chunks and carried on the
    /// final streaming response. OpenAI's [`Usage`] for most providers;
    /// providers with richer usage accounting (e.g. Mistral's cached-token
    /// fallbacks, DeepSeek's cache hit/miss counters) substitute their own.
    type StreamingUsage: Clone
        + Default
        + GetTokenUsage
        + Serialize
        + serde::de::DeserializeOwned
        + Unpin
        + WasmCompatSend
        + WasmCompatSync
        + 'static;

    /// The chat-completions payload this provider returns.
    type Response: serde::de::DeserializeOwned
        + Serialize
        + crate::telemetry::ProviderResponseExt<Usage: GetTokenUsage>
        + TryInto<completion::CompletionResponse<Self::Response>, Error = CompletionError>
        + WasmCompatSend
        + WasmCompatSync;

    /// The request path for chat completions, resolved against the client
    /// base URL by [`Provider::build_uri`](crate::client::Provider::build_uri).
    /// Providers that route the model through the URL (e.g. Azure deployment
    /// paths) or keep other capabilities on differently-versioned paths
    /// override this. `model` is the identifier the completion model handle
    /// was created with; per-request model overrides only affect the body.
    fn completion_path(&self, model: &str) -> String {
        let _ = model;
        "/chat/completions".to_string()
    }

    /// Build the typed chat-completions request. Providers that share the
    /// OpenAI transport but need provider-specific message conversion can
    /// override this while still using [`GenericCompletionModel`] for sending,
    /// streaming, error handling, and telemetry.
    fn build_completion_request(
        &self,
        model: String,
        request: CoreCompletionRequest,
        options: CompletionModelOptions,
    ) -> Result<CompletionRequest, CompletionError> {
        let mut request = CompletionRequest::try_from(OpenAIRequestParams {
            model,
            request,
            strict_tools: options.strict_tools,
            tool_result_array_content: options.tool_result_array_content,
            supports_response_format: Self::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: Self::SUPPORTS_TOOLS,
        })?;
        if Self::PROVIDER_NAME == "openai" {
            request.merge_provider_tools();
        }
        Ok(request)
    }

    /// Adjust the typed request before serialization (e.g. rewrite the model
    /// identifier or fold provider-native tool definitions out of
    /// `additional_params`).
    fn prepare_request(&self, request: &mut CompletionRequest) -> Result<(), CompletionError> {
        let _ = request;
        Ok(())
    }

    /// Adjust the fully serialized request body — after any streaming
    /// parameters are merged — immediately before it is sent. This is where
    /// wire-level dialect differences live (e.g. Mistral's `"any"` tool
    /// choice, DeepSeek's string-flattened message content).
    fn finalize_request_body(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
        let _ = body;
        Ok(())
    }

    /// Adjust the fully serialized request body with model-level options.
    /// Providers that do not need model-instance options should override
    /// [`finalize_request_body`](Self::finalize_request_body) instead.
    fn finalize_request_body_with_options(
        &self,
        body: &mut serde_json::Value,
        options: CompletionModelOptions,
    ) -> Result<(), CompletionError> {
        let _ = options;
        self.finalize_request_body(body)
    }

    /// Decorate streamed tool calls from provider-specific streaming detail
    /// payloads. Most OpenAI-compatible providers do not emit such details.
    fn decorate_streaming_tool_call(
        &self,
        detail: &serde_json::Value,
        tool_calls: &mut std::collections::HashMap<usize, crate::streaming::RawStreamingToolCall>,
    ) {
        let _ = (detail, tool_calls);
    }
}

impl OpenAICompatibleProvider for super::OpenAICompletionsExt {
    const PROVIDER_NAME: &'static str = "openai";

    type StreamingUsage = Usage;
    type Response = CompletionResponse;
}

/// A chat-completions model over any [`OpenAICompatibleProvider`] extension.
/// This is the advertised path for OpenAI-compatible providers; see the
/// provider checklist in [`crate::providers`].
#[derive(Clone)]
pub struct GenericCompletionModel<Ext = super::OpenAICompletionsExt, H = reqwest::Client> {
    pub(crate) client: crate::client::Client<Ext, H>,
    pub model: String,
    pub(crate) strict_tools: bool,
    pub(crate) tool_result_array_content: bool,
    pub(crate) prompt_caching: bool,
}

/// The completion model struct for OpenAI's Chat Completions API.
///
/// This preserves the historical public generic shape where the first generic
/// parameter is the HTTP client type.
pub type CompletionModel<H = reqwest::Client> =
    GenericCompletionModel<super::OpenAICompletionsExt, H>;

impl<Ext, H> GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>: std::fmt::Debug + Clone + 'static,
    Ext: crate::client::Provider + Clone + 'static,
{
    pub fn new(client: crate::client::Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
            prompt_caching: false,
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
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(flatten)]
    pub additional_params: Option<serde_json::Value>,
}

impl CompletionRequest {
    fn merge_provider_tools(&mut self) {
        if let Some(params) = self.additional_params.as_mut()
            && let Some(object) = params.as_object_mut()
            && let Some(hosted) = object.remove("tools")
        {
            let hosted = hosted.as_array().cloned().unwrap_or_else(|| vec![hosted]);
            self.tools
                .extend(hosted.into_iter().map(ChatToolDefinition::Hosted));
        }
    }
}

/// Shared helper for provider `finalize_request_body` hooks whose APIs take
/// message `content` as a plain string: flattens a content-part array to the
/// concatenation of its text parts. When `only_if_all_text` is set, arrays
/// containing non-text parts are left untouched (for APIs with their own
/// multimodal handling); otherwise non-text parts are dropped.
pub(crate) fn flatten_text_content_parts(
    content: &mut serde_json::Value,
    separator: &str,
    only_if_all_text: bool,
) {
    // Refusals are textual content too; flatten them alongside text parts.
    // Checked per key so a null-padded `text` next to a string `refusal`
    // still counts as textual.
    fn part_text(part: &serde_json::Value) -> Option<&str> {
        part.get("text")
            .and_then(serde_json::Value::as_str)
            .or_else(|| part.get("refusal").and_then(serde_json::Value::as_str))
    }

    let Some(parts) = content.as_array() else {
        return;
    };
    if only_if_all_text && !parts.iter().all(|part| part_text(part).is_some()) {
        return;
    }
    let mut flattened = String::new();
    for text in parts.iter().filter_map(part_text) {
        if !flattened.is_empty() {
            flattened.push_str(separator);
        }
        flattened.push_str(text);
    }
    *content = serde_json::Value::String(flattened);
}

/// Joins the `text` fields of `type == "text"` content parts, in order.
pub(crate) fn joined_text_parts(parts: &[serde_json::Value]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            (part.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                .then(|| part.get("text").and_then(serde_json::Value::as_str))
                .flatten()
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Shared helper for provider `finalize_request_body` hooks whose APIs only
/// accept plain `{role, content}` chat messages: removes tool-exchange
/// remnants left in shared histories (role `tool` messages, assistant
/// `tool_calls`/`reasoning_content`), optionally flattens content-part arrays
/// to strings, and drops assistant turns left without content (pure
/// tool-call scaffolding). With `merge_same_role`, consecutive same-role
/// string-content messages are additionally merged — the removals can leave
/// user/user as well as assistant/assistant adjacency, and alternation-strict
/// APIs (Perplexity) reject both; providers without that constraint keep
/// their turns separate.
pub(crate) fn sanitize_plain_text_history(
    messages: &mut Vec<serde_json::Value>,
    flatten: Option<(&str, bool)>,
    strip_names: bool,
    merge_same_role: bool,
) {
    messages
        .retain(|message| message.get("role").and_then(serde_json::Value::as_str) != Some("tool"));

    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(serde_json::Value::as_str) == Some("assistant") {
            object.remove("tool_calls");
            object.remove("reasoning_content");
        }
        if strip_names {
            object.remove("name");
        }
        if let Some((separator, only_if_all_text)) = flatten
            && let Some(content) = object.get_mut("content")
        {
            flatten_text_content_parts(content, separator, only_if_all_text);
        }
    }

    messages.retain(|message| {
        if message.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
            return true;
        }
        match message.get("content") {
            Some(serde_json::Value::String(text)) => !text.is_empty(),
            Some(serde_json::Value::Null) | None => false,
            Some(_) => true,
        }
    });

    if !merge_same_role {
        return;
    }

    let mut merged: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    for message in std::mem::take(messages) {
        let merged_text = if let Some(role) = message
            .get("role")
            .and_then(serde_json::Value::as_str)
            .filter(|role| matches!(*role, "assistant" | "user"))
            && let Some(previous) = merged.last()
            && previous.get("role").and_then(serde_json::Value::as_str) == Some(role)
            && let Some(previous_text) = previous.get("content").and_then(serde_json::Value::as_str)
            && let Some(text) = message.get("content").and_then(serde_json::Value::as_str)
        {
            Some(format!("{previous_text}\n{text}"))
        } else {
            None
        };

        if let Some(text) = merged_text
            && let Some(previous) = merged.last_mut().and_then(serde_json::Value::as_object_mut)
        {
            previous.insert("content".to_string(), serde_json::Value::String(text));
            continue;
        }
        merged.push(message);
    }
    *messages = merged;
}

pub struct OpenAIRequestParams {
    pub model: String,
    pub request: CoreCompletionRequest,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
    /// Maps `output_schema` to `response_format` when true; drops it with a
    /// warning when false (providers whose APIs reject `json_schema`).
    pub supports_response_format: bool,
    /// Serializes `tools`/`tool_choice` when true; drops them with a warning
    /// when false (providers without tool-calling support).
    pub supports_tools: bool,
}

impl TryFrom<OpenAIRequestParams> for CompletionRequest {
    type Error = CompletionError;

    fn try_from(params: OpenAIRequestParams) -> Result<Self, Self::Error> {
        let OpenAIRequestParams {
            model,
            request: req,
            strict_tools,
            tool_result_array_content,
            supports_response_format,
            supports_tools,
        } = params;
        let chat_history = req.chat_history_with_documents();

        let CoreCompletionRequest {
            model: request_model,
            preamble,
            chat_history: _,
            tools,
            temperature,
            max_tokens,
            additional_params,
            tool_choice,
            output_schema,
            ..
        } = req;

        let mut partial_history = Vec::new();
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

        let history_has_tool_result = history_contains_tool_result(&full_history);

        let (tools, tool_choice) = if supports_tools {
            let tool_choice = tool_choice.map(ToolChoice::try_from).transpose()?;
            let tools: Vec<ChatToolDefinition> = tools
                .into_iter()
                .map(|tool| {
                    let def = ToolDefinition::from(tool);
                    ChatToolDefinition::Function(if strict_tools { def.with_strict() } else { def })
                })
                .collect();
            (tools, tool_choice)
        } else {
            if !tools.is_empty() {
                tracing::warn!("Tool use is not supported by this provider; tools will be ignored");
            }
            if tool_choice.is_some() {
                tracing::warn!("Tool choice is not supported by this provider and will be ignored");
            }
            (Vec::new(), None)
        };

        if output_schema.is_some() && !supports_response_format {
            tracing::warn!(
                "Structured outputs are not supported by this provider; ignoring output_schema"
            );
        }

        // Some OpenAI-compatible backends such as llama.cpp will skip tool execution
        // if `response_format` is sent on the first turn alongside tools. Delay the
        // schema until after the conversation contains a tool result.
        let should_apply_response_format = output_schema.is_some()
            && supports_response_format
            && (tools.is_empty() || history_has_tool_result);

        // Map output_schema to OpenAI's response_format and merge into additional_params
        let additional_params = if let Some(schema) = output_schema
            && should_apply_response_format
        {
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
            max_tokens,
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
            supports_response_format: true,
            supports_tools: true,
        })
    }
}

impl GenericCompletionModel<super::OpenAICompletionsExt, reqwest::Client> {
    pub fn into_agent_builder(self) -> crate::agent::AgentBuilder<Self> {
        crate::agent::AgentBuilder::new(self)
    }
}

impl<Ext, H> completion::CompletionModel for GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: crate::client::Provider
        + OpenAICompatibleProvider
        + crate::client::DebugExt
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    H: Clone + Default + std::fmt::Debug + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = Ext::Response;
    type StreamingResponse = StreamingCompletionResponse<Ext::StreamingUsage>;

    type Client = crate::client::Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    // OpenAI Chat Completions *defers* `response_format` while tools are present
    // and no tool result exists yet (see `should_apply_response_format`), then
    // applies it once a tool result is in the history. So the native constraint
    // does not suppress tool calls — they compose — which is what this flag
    // governs. (Caveat: a turn-1 answer with no tool call is therefore not
    // schema-constrained; `Native` is "guaranteed" only once tools have run.)
    // See issue #1928.
    fn composes_native_output_with_tools(&self) -> bool {
        // Providers that drop `output_schema` (SUPPORTS_RESPONSE_FORMAT =
        // false) cannot compose native structured output with tools; the
        // agent then falls back to tool-mode enforcement as their
        // pre-migration hand-rolled models did.
        Ext::SUPPORTS_RESPONSE_FORMAT
    }

    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse<Ext::Response>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let options = CompletionModelOptions {
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            prompt_caching: self.prompt_caching,
        };
        let mut request = self.client.ext().build_completion_request(
            self.model.to_owned(),
            completion_request,
            options,
        )?;
        self.client.ext().prepare_request(&mut request)?;
        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &request.model,
            CompletionOperation::Chat,
        )
        .system_instructions(system_instructions.as_deref())
        .build();

        let mut request_body = serde_json::to_value(&request)?;
        self.client
            .ext()
            .finalize_request_body_with_options(&mut request_body, options)?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Chat Completions completion request: {}",
                serde_json::to_string_pretty(&request_body)?
            );
        }

        let body = serde_json::to_vec(&request_body)?;
        // Deliberately the configured model, not the per-request override:
        // Azure's deployment URL is pinned to the model handle.
        let path = self.client.ext().completion_path(&self.model);

        let req = self
            .client
            .post(&path)?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(req).await?;

            let status = response.status();
            if status.is_success() {
                let text = http_client::text(response).await?;

                match serde_json::from_str::<ApiResponse<Ext::Response>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_response_metadata(&response);
                        span.record_token_usage(&response.get_usage());

                        if enabled!(Level::TRACE) {
                            tracing::trace!(
                                target: "rig::completions",
                                "OpenAI Chat Completions completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        response.try_into()
                    }
                    ApiResponse::Err(err) => {
                        tracing::warn!(message = %err.message, "provider returned an error response");
                        Err(CompletionError::from_http_response(status, text))
                    }
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::from_http_response(status, text))
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
        GenericCompletionModel::stream(self, request).await
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
    use crate::completion::CompletionRequestBuilder;
    use crate::telemetry::ProviderResponseExt;
    use crate::test_utils::MockCompletionModel;
    use std::collections::HashMap;

    fn test_document(id: &str, text: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: HashMap::new(),
        }
    }

    #[test]
    fn video_data_uri_with_unrecognized_mime_round_trips_as_url() {
        let original = "data:video/quicktime;base64,AAAA";
        let openai_content = UserContent::Video {
            video_url: VideoUrl {
                url: original.to_string(),
            },
        };

        let rig_content: message::UserContent = openai_content.into();
        // Unrecognized MIME: kept as a URL source, not decomposed.
        assert!(matches!(
            &rig_content,
            message::UserContent::Video(video)
                if matches!(&video.data, message::DocumentSourceKind::Url(url) if url == original)
        ));

        let back = UserContent::try_from(rig_content).expect("video should convert back");
        assert!(matches!(
            back,
            UserContent::Video { video_url } if video_url.url == original
        ));
    }

    #[test]
    fn video_data_uri_with_known_mime_decomposes_to_base64() {
        let openai_content = UserContent::Video {
            video_url: VideoUrl {
                url: "data:video/mp4;base64,AAAA".to_string(),
            },
        };

        let rig_content: message::UserContent = openai_content.into();
        assert!(matches!(
            &rig_content,
            message::UserContent::Video(video)
                if video.media_type == Some(crate::message::VideoMediaType::MP4)
                    && matches!(&video.data, message::DocumentSourceKind::Base64(data) if data == "AAAA")
        ));
    }

    #[test]
    fn sanitize_plain_text_history_strips_tool_exchange_and_keeps_alternation() {
        let mut messages = vec![
            serde_json::json!({"role": "user", "content": "Look up the label."}),
            serde_json::json!({"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]}),
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "crimson"}),
            serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": "The label is crimson."}],
                "reasoning_content": "thinking"
            }),
            serde_json::json!({"role": "user", "content": "Thanks!"}),
        ];

        sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

        let roles = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect::<Vec<_>>();
        // tool message removed, tool-call-only assistant dropped, no
        // consecutive assistants left.
        assert_eq!(roles, ["user", "assistant", "user"]);
        assert_eq!(messages[1]["content"], "The label is crimson.");
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[1].get("tool_calls").is_none());
    }

    #[test]
    fn sanitize_plain_text_history_merges_consecutive_user_messages() {
        // Dropping a tool exchange whose final assistant answer never made it
        // into history leaves user/user adjacency, which alternation-strict
        // APIs reject.
        let mut messages = vec![
            serde_json::json!({"role": "user", "content": "Look it up."}),
            serde_json::json!({"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]}),
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "crimson"}),
            serde_json::json!({"role": "user", "content": "Ask again."}),
        ];

        sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Look it up.\nAsk again.");
    }

    #[test]
    fn flatten_text_content_parts_treats_refusals_as_text() {
        let mut content = serde_json::json!([
            {"type": "text", "text": "Partly:"},
            {"type": "refusal", "refusal": "I cannot help with that."}
        ]);

        flatten_text_content_parts(&mut content, "\n", true);

        assert_eq!(content, "Partly:\nI cannot help with that.");
    }

    #[test]
    fn sanitize_plain_text_history_merges_consecutive_assistant_messages() {
        let mut messages = vec![
            serde_json::json!({"role": "assistant", "content": "First."}),
            serde_json::json!({"role": "tool", "tool_call_id": "c", "content": "x"}),
            serde_json::json!({"role": "assistant", "content": "Second."}),
        ];

        sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "First.\nSecond.");
    }

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
            supports_response_format: true,
            supports_tools: true,
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
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert_eq!(serialized["model"], "gpt-4o-mini");
    }

    #[test]
    fn openai_chat_request_keeps_documents_after_system_messages() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Prompt")
            .message(crate::completion::Message::system("System prompt"))
            .message(crate::completion::Message::user("Earlier user turn"))
            .message(crate::completion::Message::assistant(
                "Earlier assistant turn",
            ))
            .document(test_document("doc1", "Document text."))
            .build();

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(&openai_request.messages).expect("messages should serialize");
        let messages = serialized.as_array().expect("messages should be an array");

        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert!(
            messages[1].to_string().contains("<file id: doc1>"),
            "document message should follow system message: {messages:?}"
        );
        assert_eq!(messages[2]["role"], "user");
        assert!(
            messages[2].to_string().contains("Earlier user turn"),
            "prior user history should follow document message: {messages:?}"
        );
        assert_eq!(messages[3]["role"], "assistant");
        assert!(
            messages[3].to_string().contains("Earlier assistant turn"),
            "prior assistant history should follow prior user history: {messages:?}"
        );
        assert_eq!(messages[4]["role"], "user");
        assert!(
            messages[4].to_string().contains("Prompt"),
            "prompt should remain last: {messages:?}"
        );
    }

    #[test]
    fn openai_chat_direct_request_keeps_documents_after_system_messages() {
        let request = CoreCompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::many(vec![
                crate::completion::Message::system("System prompt"),
                crate::completion::Message::assistant("Earlier assistant turn"),
                crate::completion::Message::system("Mid-conversation instruction"),
                crate::completion::Message::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![test_document("doc1", "Document text.")],
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
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(&openai_request.messages).expect("messages should serialize");
        let messages = serialized.as_array().expect("messages should be an array");

        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert!(
            messages[1].to_string().contains("<file id: doc1>"),
            "document message should follow leading system messages: {messages:?}"
        );
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[3]["role"], "system");
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc1>"))
                .count(),
            1,
            "document message should appear exactly once: {messages:?}"
        );
    }

    #[test]
    fn assistant_reasoning_alone_is_dropped() {
        let assistant_content = OneOrMany::one(message::AssistantContent::reasoning("hidden"));

        let converted: Vec<Message> = assistant_content
            .try_into()
            .expect("conversion should work");

        assert!(converted.is_empty());
    }

    // Regression test: providers that serve thinking models over the OpenAI
    // Chat Completions schema (DeepSeek-R1, GLM-4.6, Qwen3-Thinking) return
    // 400 "thinking is enabled but reasoning_content is missing" on the next
    // turn if the prior assistant tool-call message didn't echo the reasoning.
    #[test]
    fn assistant_reasoning_is_attached_to_tool_call_message() {
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
                reasoning,
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
                assert_eq!(reasoning.as_deref(), Some("hidden"));
            }
            _ => panic!("expected assistant message"),
        }

        let json = serde_json::to_value(&converted[0]).expect("serialize");
        assert_eq!(json["reasoning_content"], "hidden");
    }

    #[test]
    fn assistant_reasoning_roundtrips_back_to_rig_message() {
        let assistant = Message::Assistant {
            content: vec![AssistantContent::Text {
                text: "visible".to_string(),
            }],
            reasoning: Some("hidden".to_string()),
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
            reasoning_details: vec![],
            images: vec![],
        };

        let rig_msg: message::Message = assistant.try_into().expect("convert back");

        let message::Message::Assistant { content, .. } = rig_msg else {
            panic!("expected assistant");
        };

        let items: Vec<_> = content.into_iter().collect();
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], message::AssistantContent::Reasoning(_)));
        assert!(matches!(items[1], message::AssistantContent::Text(_)));
    }

    #[test]
    fn provider_response_text_response_reads_assistant_multipart_output() {
        let response = CompletionResponse {
            id: "resp_123".to_owned(),
            object: "chat.completion".to_owned(),
            created: 0,
            model: GPT_4O.to_owned(),
            system_fingerprint: None,
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: vec![
                        AssistantContent::Text {
                            text: "first".to_owned(),
                        },
                        AssistantContent::Refusal {
                            refusal: "second".to_owned(),
                        },
                        AssistantContent::Text {
                            text: "third".to_owned(),
                        },
                    ],
                    reasoning: Some("hidden".to_owned()),
                    refusal: None,
                    audio: None,
                    name: None,
                    tool_calls: vec![],
                    reasoning_details: vec![],
                    images: vec![],
                },
                logprobs: None,
                finish_reason: "stop".to_owned(),
            }],
            usage: None,
        };

        assert_eq!(
            response.get_text_response(),
            Some("first\nsecond\nthird".to_owned())
        );
    }

    #[test]
    fn provider_response_text_response_falls_back_to_assistant_refusal_field() {
        let response = CompletionResponse {
            id: "resp_123".to_owned(),
            object: "chat.completion".to_owned(),
            created: 0,
            model: GPT_4O.to_owned(),
            system_fingerprint: None,
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: vec![],
                    reasoning: None,
                    refusal: Some("blocked".to_owned()),
                    audio: None,
                    name: None,
                    tool_calls: vec![],
                    reasoning_details: vec![],
                    images: vec![],
                },
                logprobs: None,
                finish_reason: "stop".to_owned(),
            }],
            usage: None,
        };

        assert_eq!(response.get_text_response(), Some("blocked".to_owned()));
    }

    #[test]
    fn test_max_tokens_is_forwarded_to_request() {
        let request = crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: Some(4096),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert_eq!(serialized["max_tokens"], 4096);
    }

    #[test]
    fn test_max_tokens_omitted_when_none() {
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
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert!(serialized.get("max_tokens").is_none());
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
            supports_response_format: true,
            supports_tools: true,
        });

        assert!(matches!(result, Err(CompletionError::RequestError(_))));
    }

    #[test]
    fn request_conversion_omits_response_format_on_initial_tool_turn() {
        let request = CoreCompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(message::Message::user(
                "Hello, whats the weather in London?",
            )),
            documents: vec![],
            tools: vec![completion::ToolDefinition {
                output_schema: None,
                name: "weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: Some(
                serde_json::from_value(serde_json::json!({
                    "title": "WeatherResponse",
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "weather": { "type": "string" }
                    },
                    "required": ["city", "weather"]
                }))
                .expect("schema should deserialize"),
            ),
        };

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert!(
            serialized.get("response_format").is_none(),
            "initial tool turn should omit response_format: {serialized:?}"
        );
    }

    #[test]
    fn request_conversion_restores_response_format_after_tool_result() {
        let request = CoreCompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                message::Message::user("Hello, whats the weather in London?"),
                message::Message::Assistant {
                    id: None,
                    content: OneOrMany::one(message::AssistantContent::tool_call(
                        "call_1",
                        "weather",
                        serde_json::json!({ "city": "London" }),
                    )),
                },
                message::Message::tool_result(
                    "call_1",
                    "The weather in London is all fire and brimstone",
                ),
            ])
            .expect("history should be non-empty"),
            documents: vec![],
            tools: vec![completion::ToolDefinition {
                output_schema: None,
                name: "weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: Some(
                serde_json::from_value(serde_json::json!({
                    "title": "WeatherResponse",
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "weather": { "type": "string" }
                    },
                    "required": ["city", "weather"]
                }))
                .expect("schema should deserialize"),
            ),
        };

        let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request conversion should succeed");

        let serialized =
            serde_json::to_value(openai_request).expect("serialization should succeed");

        assert!(
            serialized.get("response_format").is_some(),
            "follow-up turn should restore response_format: {serialized:?}"
        );
    }

    #[test]
    fn deserialize_llama_cpp_tool_call() {
        let request = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{ "type": "function", "function": { "name": "hello_world", "arguments": { "city": "Paris" } }, "id": "xxx" }]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": { "completion_tokens": 13, "prompt_tokens": 255, "total_tokens": 268 },
            "id": "xxx"
        }
        "#;
        let response = serde_json::from_str::<ApiResponse<CompletionResponse>>(request).unwrap();

        let ApiResponse::Ok(response) = response else {
            panic!("expected successful completion response");
        };
        assert_eq!(response.choices.len(), 1);

        let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
            panic!("expected assistant message");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "xxx");
        assert_eq!(tool_calls[0].function.name, "hello_world");
        assert_eq!(
            tool_calls[0].function.arguments,
            serde_json::json!({"city": "Paris"})
        );
    }

    #[test]
    fn deserialize_openai_stringified_tool_call() {
        let request = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{ "type": "function", "function": { "name": "hello_world", "arguments": "{\"city\":\"Paris\"}" }, "id": "xxx" }]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": { "completion_tokens": 13, "prompt_tokens": 255, "total_tokens": 268 },
            "id": "xxx"
        }
        "#;
        let response = serde_json::from_str::<ApiResponse<CompletionResponse>>(request).unwrap();

        let ApiResponse::Ok(response) = response else {
            panic!("expected successful completion response");
        };
        assert_eq!(response.choices.len(), 1);

        let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
            panic!("expected assistant message");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "xxx");
        assert_eq!(tool_calls[0].function.name, "hello_world");
        assert_eq!(
            tool_calls[0].function.arguments,
            serde_json::json!({"city": "Paris"})
        );
    }

    #[test]
    fn deserialize_llama_cpp_response_with_reasoning_content() {
        let request = r#"
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Now I understand the structure better. I need to: ..."
                    }
                }
            ],
            "created": 1776750378,
            "model": "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 920,
                "prompt_tokens": 27806,
                "total_tokens": 28726,
                "prompt_tokens_details": { "cached_tokens": 18698 }
            },
            "id": "chatcmpl-xxxx",
            "timings": {
                "cache_n": 18698,
                "prompt_n": 9108,
                "prompt_ms": 226645.81,
                "prompt_per_token_ms": 24.884256697408873,
                "prompt_per_second": 40.186050648807495,
                "predicted_n": 920,
                "predicted_ms": 177167.955,
                "predicted_per_token_ms": 192.57386413043477,
                "predicted_per_second": 5.192812661860888
            }
        }
        "#;
        let response = serde_json::from_str::<ApiResponse<CompletionResponse>>(request).unwrap();
        let ApiResponse::Ok(response) = response else {
            panic!("expected successful completion response");
        };

        let response: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();

        assert_eq!(response.choice.len(), 1);

        let completion::message::AssistantContent::Reasoning(reasoning) = response.choice.first()
        else {
            panic!("expected assistant content to be reasoning");
        };
        assert_eq!(
            reasoning.first_text(),
            Some("Now I understand the structure better. I need to: ...")
        );
    }

    #[test]
    fn pdf_base64_document_serializes_as_file_content_part() {
        let doc = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Base64("JVBERi0xLjQK".into()),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        });
        let converted: UserContent = doc.try_into().expect("conversion should succeed");
        let json = serde_json::to_value(&converted).expect("serialize");

        assert_eq!(json["type"], "file");
        assert_eq!(
            json["file"]["file_data"],
            "data:application/pdf;base64,JVBERi0xLjQK"
        );
        assert_eq!(json["file"]["filename"], "document.pdf");
        assert!(json["file"].get("file_id").is_none());
    }

    #[test]
    fn file_id_document_serializes_as_file_content_part() {
        let doc = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId("file_abc".into()),
            media_type: None,
            additional_params: None,
        });
        let converted: UserContent = doc.try_into().expect("conversion should succeed");
        let json = serde_json::to_value(&converted).expect("serialize");

        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["file_id"], "file_abc");
        assert!(json["file"].get("file_data").is_none());
    }

    #[test]
    fn base64_image_without_detail_defaults_to_auto() {
        let image = message::UserContent::Image(message::Image {
            data: DocumentSourceKind::Base64("iVBORw0KGgo=".into()),
            media_type: Some(message::ImageMediaType::PNG),
            detail: None,
            additional_params: None,
        });
        let converted: UserContent = image.try_into().expect("conversion should succeed");
        let UserContent::Image { image_url } = converted else {
            panic!("expected image content");
        };

        assert_eq!(image_url.url, "data:image/png;base64,iVBORw0KGgo=");
        assert_eq!(image_url.detail, Some(ImageDetail::Auto));
    }

    // Regression guard: callers passing markdown/plain text wrapped in
    // `UserContent::Document` should keep getting flattened to `text`.
    #[test]
    fn non_pdf_document_still_serializes_as_text() {
        let doc = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::String("# Markdown".into()),
            media_type: None,
            additional_params: None,
        });
        let converted: UserContent = doc.try_into().expect("conversion should succeed");
        let json = serde_json::to_value(&converted).expect("serialize");

        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "# Markdown");
    }

    #[test]
    fn pdf_url_document_returns_conversion_error() {
        let doc = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Url("https://example.com/x.pdf".into()),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        });
        let res: Result<UserContent, _> = doc.try_into();
        assert!(matches!(
            res,
            Err(message::MessageError::ConversionError(_))
        ));
    }

    #[test]
    fn pdf_raw_document_returns_conversion_error() {
        let doc = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Raw(b"%PDF-1.4\n".to_vec()),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        });
        let res: Result<UserContent, _> = doc.try_into();
        assert!(matches!(
            res,
            Err(message::MessageError::ConversionError(_))
        ));
    }

    #[test]
    fn file_user_content_deserializes_from_wire_json() {
        let raw = r#"{"type":"file","file":{"file_data":"data:application/pdf;base64,AAAA","filename":"x.pdf"}}"#;
        let parsed: UserContent = serde_json::from_str(raw).expect("deserialize");
        let UserContent::File { file } = parsed else {
            panic!("expected File variant");
        };
        assert_eq!(
            file.file_data.as_deref(),
            Some("data:application/pdf;base64,AAAA")
        );
        assert_eq!(file.filename.as_deref(), Some("x.pdf"));
        assert!(file.file_id.is_none());
    }

    #[test]
    fn file_variant_round_trips_back_to_pdf_document() {
        let wire = UserContent::File {
            file: FileData {
                file_data: Some("data:application/pdf;base64,QUJD".to_string()),
                file_id: None,
                filename: Some("document.pdf".to_string()),
            },
        };
        let rig: message::UserContent = wire.into();
        let message::UserContent::Document(doc) = rig else {
            panic!("expected Document");
        };
        assert_eq!(doc.media_type, Some(message::DocumentMediaType::PDF));
        assert!(matches!(doc.data, DocumentSourceKind::Base64(ref b) if b == "QUJD"));
    }

    #[test]
    fn file_variant_with_file_id_only_round_trips_to_document_file_id() {
        let wire = UserContent::File {
            file: FileData {
                file_data: None,
                file_id: Some("file_abc".to_string()),
                filename: None,
            },
        };
        let rig: message::UserContent = wire.into();
        let message::UserContent::Document(doc) = rig else {
            panic!("expected Document");
        };
        assert_eq!(doc.media_type, None);
        assert!(matches!(doc.data, DocumentSourceKind::FileId(ref id) if id == "file_abc"));

        let converted: UserContent = message::UserContent::Document(doc)
            .try_into()
            .expect("conversion should succeed");
        let json = serde_json::to_value(&converted).expect("serialize");

        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["file_id"], "file_abc");
        assert!(json["file"].get("file_data").is_none());
    }

    // Guards against `OneOrMany::many` flattening at the User content site:
    // a mixed text + PDF message must produce one User message with both parts.
    #[test]
    fn mixed_text_and_pdf_user_message_produces_two_content_parts() {
        let user = message::Message::User {
            content: OneOrMany::many(vec![
                message::UserContent::text("What is in this PDF?"),
                message::UserContent::Document(message::Document {
                    data: DocumentSourceKind::Base64("JVBERi0K".into()),
                    media_type: Some(message::DocumentMediaType::PDF),
                    additional_params: None,
                }),
            ])
            .expect("non-empty content"),
        };
        let converted: Vec<Message> = user.try_into().expect("conversion should succeed");
        assert_eq!(converted.len(), 1);
        let Message::User { content, .. } = &converted[0] else {
            panic!("expected user message");
        };
        let parts: Vec<&UserContent> = content.iter().collect();
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], UserContent::Text { .. }));
        assert!(matches!(parts[1], UserContent::File { .. }));
    }

    #[tokio::test]
    async fn completion_preserves_raw_provider_error_json_on_api_error_envelope() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::providers::openai::CompletionsClient;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"slow down","type":"rate_limit","code":"rate_limit_exceeded"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o-mini");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
                assert_eq!(error.provider_response_body(), Some(body));
                assert_eq!(
                    error.provider_response_status(),
                    Some(http::StatusCode::ACCEPTED)
                );
                let json = error
                    .provider_response_json()
                    .expect("raw body should be valid JSON")
                    .expect("parsed JSON should be present");
                assert_eq!(json["code"], "rate_limit_exceeded");
                assert_eq!(json["type"], "rate_limit");
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[test]
    fn mixed_native_and_hosted_chat_tools_use_one_wire_key() {
        let mut request = CompletionRequest {
            model: "gpt".into(),
            messages: vec![],
            tools: vec![ChatToolDefinition::Function(ToolDefinition {
                r#type: "function".into(),
                function: FunctionDefinition {
                    name: "native".into(),
                    description: "native".into(),
                    parameters: serde_json::json!({"type":"object"}),
                    strict: None,
                },
            })],
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            additional_params: Some(serde_json::json!({"tools":[{"type":"web_search"}]})),
        };
        request.merge_provider_tools();
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["tools"].as_array().unwrap().len(), 2);
        assert_eq!(
            json.as_object()
                .unwrap()
                .keys()
                .filter(|key| *key == "tools")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::providers::openai::CompletionsClient;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
        let client = CompletionsClient::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("gpt-4o-mini");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(error.provider_response_body(), Some(body));
        let json = error
            .provider_response_json()
            .expect("raw body should be valid JSON")
            .expect("parsed JSON should be present");
        assert_eq!(json["error"]["type"], "rate_limit_error");
    }
}
