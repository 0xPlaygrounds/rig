//! Chat Completions message types, model identifiers, and request conversion.
//!
//! ```
//! use rig_core::providers::openai::completion::Message;
//! let message = Message::system("Answer briefly.");
//! ```

use crate::completion::CompletionRequest as CoreCompletionRequest;
use crate::error::EncodeError;
use crate::json_utils::string_or_vec;
use crate::message::{AudioMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::{completion, json_utils, message};
use serde::{Deserialize, Serialize, Serializer};
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

/// Serializes user content as a plain string when there's a single text item,
/// otherwise as an array of content parts.
fn serialize_user_content<S>(content: &[UserContent], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if content.len() == 1
        && let Some(UserContent::Text { text, .. }) = content.first()
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
/// `gpt-5-mini` completion model.
pub const GPT_5_MINI: &str = "gpt-5-mini";
/// `gpt-5-nano` completion model.
pub const GPT_5_NANO: &str = "gpt-5-nano";

/// `gpt-4.5-preview` completion model
pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
/// `gpt-4.5-preview-2025-02-27` completion model
pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
/// `gpt-4o-2024-11-20` completion model.
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
/// `o1` completion model.
pub const O1: &str = "o1";
/// `o1-2024-12-17` completion model
pub const O1_2024_12_17: &str = "o1-2024-12-17";
/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini` completion model.
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
        #[serde(deserialize_with = "string_or_vec")]
        content: Vec<SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(
            deserialize_with = "string_or_vec",
            serialize_with = "serialize_user_content"
        )]
        content: Vec<UserContent>,
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
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// Structured reasoning blocks used by OpenAI-compatible providers
        /// such as OpenRouter. Empty (and omitted from the wire) for
        /// providers that do not emit or accept them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning_details: Vec<ReasoningDetails>,
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
            content: vec![content.to_owned().into()],
            name: None,
        }
    }
}

fn history_contains_tool_result(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|message| matches!(message, Message::ToolResult { .. }))
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
    /// Audio content part, OpenAI's `input_audio` wire tag.
    #[serde(rename = "input_audio")]
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

/// Text or image content in a tool-result message.
/// Image emission requires
/// [`Quirks::supports_image_tool_results`](crate::providers::openai::wire::Quirks::supports_image_tool_results).
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum ToolResultContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
}

impl ToolResultContent {
    /// The text of this part, or `None` for a non-text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            Self::Image { .. } => None,
        }
    }
}

impl FromStr for ToolResultContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent::Text { text: s }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContentValue {
    Array(Vec<ToolResultContent>),
    String(String),
}

impl ToolResultContentValue {
    /// Join text parts with newlines, discarding images. String values are cloned.
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContentValue::Array(arr) => arr
                .iter()
                .filter_map(ToolResultContent::as_text)
                .collect::<Vec<_>>()
                .join("\n"),
            ToolResultContentValue::String(s) => s.clone(),
        }
    }

    /// Whether any part of this result is an image.
    pub fn has_image(&self) -> bool {
        matches!(self, ToolResultContentValue::Array(arr)
            if arr.iter().any(|c| matches!(c, ToolResultContent::Image { .. })))
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

#[derive(Default, Clone, Debug, PartialEq)]
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
    type Error = EncodeError;
    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::Specific { function_names } => {
                let [name] = function_names.as_slice() else {
                    return Err(EncodeError::request(
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
        // Single-item conversion supplies a candidate. The full-request
        // builder applies occurrence-scoped IDs to both calls and results.
        let tool_call_id = value.wire_call_id().into_owned();
        let parts = value
            .content
            .into_iter()
            .map(|content| match content {
                message::ToolResultContent::Text(message::Text { text, .. }) => {
                    Ok(ToolResultContent::from(text))
                }
                message::ToolResultContent::Json { value } => {
                    Ok(ToolResultContent::from(value.to_string()))
                }
                // The request builder checks image support using the selected dialect.
                message::ToolResultContent::Image(message::Image {
                    data,
                    media_type,
                    detail,
                    ..
                }) => {
                    let url = match data {
                        DocumentSourceKind::Url(url) => url,
                        DocumentSourceKind::Base64(data) => {
                            let media_type = media_type.ok_or_else(|| {
                                message::MessageError::ConversionError(
                                    "a base64 image in a tool result needs a media type to build \
                                     its data URI"
                                        .into(),
                                )
                            })?;
                            format!("data:{};base64,{}", media_type.to_mime_type(), data)
                        }
                        // Error messages must not expose raw image bytes.
                        DocumentSourceKind::Raw(_) => {
                            return Err(message::MessageError::ConversionError(
                                "raw image bytes are not supported in a tool result; encode as \
                                 base64 first"
                                    .into(),
                            ));
                        }
                        // Source values may contain private caller data.
                        DocumentSourceKind::FileId(_) => {
                            return Err(message::MessageError::ConversionError(
                                "a provider-side file id is not supported in a tool result on \
                                 this surface; use a URL or base64"
                                    .into(),
                            ));
                        }
                        DocumentSourceKind::String(_) | DocumentSourceKind::Unknown => {
                            return Err(message::MessageError::ConversionError(
                                "this image carries no usable source; use a URL or base64".into(),
                            ));
                        }
                    };
                    Ok(ToolResultContent::Image {
                        image_url: ImageUrl { url, detail },
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Only a lone *text* part flattens to a bare string; an image has no
        // string form, so flattening it would silently discard it.
        let content = match parts.as_slice() {
            [ToolResultContent::Text { text }] => ToolResultContentValue::String(text.clone()),
            _ => ToolResultContentValue::Array(parts),
        };

        Ok(Message::ToolResult {
            tool_call_id,
            content,
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
                        format: media_type.unwrap_or(AudioMediaType::MP3),
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

/// Convert user content into ordered user and tool-result messages.
/// Group adjacent non-tool content. Return conversion errors for unsupported media.
pub fn user_content_to_messages(
    value: Vec<message::UserContent>,
) -> Result<Vec<Message>, message::MessageError> {
    fn flush_user_content(messages: &mut Vec<Message>, pending: &mut Vec<UserContent>) {
        // Consecutive tool results must not introduce empty user messages.
        if pending.is_empty() {
            return;
        }

        messages.push(Message::User {
            content: std::mem::take(pending),
            name: None,
        });
    }

    let mut messages = Vec::new();
    let mut pending = Vec::new();

    for content in value {
        match content {
            message::UserContent::ToolResult(tool_result) => {
                flush_user_content(&mut messages, &mut pending);
                messages.push(tool_result.try_into()?);
            }
            content => pending.push(content.try_into()?),
        }
    }

    flush_user_content(&mut messages, &mut pending);
    Ok(messages)
}

/// Convert assistant content into at most one message, rejecting images.
/// When `reasoning_details` is true, preserve structured reasoning parts and
/// signatures; otherwise use display text. Return no message when text, calls,
/// and structured details are all empty.
pub fn assistant_content_to_messages(
    value: Vec<message::AssistantContent>,
    reasoning_details: bool,
) -> Result<Vec<Message>, message::MessageError> {
    let mut text_content = Vec::new();
    let mut tool_calls = Vec::new();
    // Distinct reasoning blocks are joined with a newline (matching
    // `display_text()`'s own inter-block separator) rather than glued
    // together, so replayed multi-block reasoning keeps its boundaries.
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut details: Vec<ReasoningDetails> = Vec::new();

    for content in value {
        match content {
            message::AssistantContent::Text(text) => text_content.push(text),
            message::AssistantContent::ToolCall(tool_call) => tool_calls.push(tool_call),
            // Structured replay preserves signatures and encrypted payloads.
            message::AssistantContent::Reasoning(reasoning)
                if reasoning_details && !reasoning.content.is_empty() =>
            {
                // A block the stream aggregated without a wire id carries the
                // accumulator's shared "" identity; it replays as a null id,
                // the shape the provider's own unary body uses.
                let id = reasoning.id.filter(|id| !id.is_empty());
                // `index` numbers the entries across the whole message, the
                // way the provider numbers the array it sent.
                let base = details.len();
                let entries = reasoning.content.iter().enumerate().map(|(offset, part)| {
                    let id = id.clone();
                    let index = Some(base + offset);
                    match part {
                        message::ReasoningContent::Text { text, signature } => {
                            ReasoningDetails::Text {
                                id,
                                format: None,
                                index,
                                text: Some(text.clone()),
                                signature: signature.clone(),
                            }
                        }
                        message::ReasoningContent::Summary(summary) => ReasoningDetails::Summary {
                            id,
                            format: None,
                            index,
                            summary: summary.clone(),
                        },
                        message::ReasoningContent::Encrypted(data)
                        | message::ReasoningContent::Redacted { data } => {
                            ReasoningDetails::Encrypted {
                                id,
                                format: None,
                                index,
                                data: data.clone(),
                            }
                        }
                    }
                });
                details.extend(entries);
            }
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

    // A details-only assistant message is not an empty turn: it is exactly
    // the signed-reasoning echo the dialect requires before the tool call it
    // precedes, and dropping it loses the signature.
    if text_content.is_empty() && tool_calls.is_empty() && details.is_empty() {
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
        name: None,
        tool_calls: tool_calls
            .into_iter()
            .map(std::convert::Into::into)
            .collect::<Vec<_>>(),
        reasoning_details: details,
    }])
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    /// The dialect-agnostic conversion. Structured reasoning replay is a
    /// per-dialect capability this impl cannot see, so reasoning takes the
    /// plain `reasoning_content` path; the wire's own conversion
    /// ([`OpenAIRequestParams`]) passes the dialect's answer.
    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::system(&content)]),
            message::Message::User { content } => user_content_to_messages(content),
            message::Message::Assistant { content, .. } => {
                assistant_content_to_messages(content, false)
            }
        }
    }
}

fn message_with_tool_ids(
    source: message::Message,
    position: usize,
    ids: &crate::providers::internal::tool_call_ids::ToolCallIds,
    reasoning_details: bool,
) -> Result<Vec<Message>, message::MessageError> {
    let content_positions: Vec<_> = match &source {
        message::Message::Assistant { content, .. } => content
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                matches!(part, message::AssistantContent::ToolCall(_)).then_some(index)
            })
            .collect(),
        message::Message::User { content } => content
            .iter()
            .enumerate()
            .filter_map(|(index, part)| {
                matches!(part, message::UserContent::ToolResult(_)).then_some(index)
            })
            .collect(),
        message::Message::System { .. } => Vec::new(),
    };
    let mut converted: Vec<Message> = match source {
        message::Message::Assistant { content, .. } => {
            assistant_content_to_messages(content, reasoning_details)?
        }
        source => source.try_into()?,
    };
    // Conversion can split text into separate messages, but retains every tool
    // call/result in source order. Assign only wire fields, never core provenance.
    let slots: Vec<&mut String> = converted
        .iter_mut()
        .flat_map(|message| match message {
            Message::Assistant { tool_calls, .. } => {
                tool_calls.iter_mut().map(|call| &mut call.id).collect()
            }
            Message::ToolResult { tool_call_id, .. } => vec![tool_call_id],
            _ => Vec::new(),
        })
        .collect();
    if slots.len() != content_positions.len() {
        return Err(message::MessageError::ConversionError(
            "tool identity mapping lost a content occurrence during OpenAI conversion".into(),
        ));
    }
    for (slot, content) in slots.into_iter().zip(content_positions) {
        *slot = ids
            .get(position, content)
            .ok_or_else(|| {
                message::MessageError::ConversionError(
                    "missing planned OpenAI tool identity".into(),
                )
            })?
            .to_owned();
    }
    Ok(converted)
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            // Use the same wire-handle selection as tool-result conversion.
            id: tool_call.wire_call_id().into_owned(),
            r#type: ToolType::default(),
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
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
        s.to_owned().into()
    }
}

impl FromStr for UserContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.to_owned().into())
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
        Ok(s.to_owned().into())
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
        Ok(s.to_owned().into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    // Null-or-missing tolerated on deserialization: some OpenAI-compatible
    // gateways (HuggingFace router sub-providers, TGI variants, Copilot's
    // multi-vendor chat route) omit them or send explicit `null`.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub object: String,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    /// Service tier that processed the request, when OpenAI reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// Return a nonempty top-level refusal only when every content part is empty.
/// Applies to whole messages; streaming fallback is evaluated per delta.
pub(crate) fn assistant_refusal_fallback<'a>(
    content: &[AssistantContent],
    refusal: Option<&'a str>,
) -> Option<&'a str> {
    let has_text = content.iter().any(|part| {
        !match part {
            AssistantContent::Text { text } => text,
            AssistantContent::Refusal { refusal } => refusal,
        }
        .is_empty()
    });

    refusal.filter(|refusal| !has_text && !refusal.is_empty())
}

/// The whole-message text view: every non-empty part in arrival order, with
/// the sibling `refusal` appended only when [`assistant_refusal_fallback`]
/// says it is the turn's text.
///
/// No wire path reads text this way — the driver records off the folded
/// response — so this survives for the OpenAI-compatible providers' unary
/// decode tests, which read a decoded message's text through it.
#[cfg(test)]
pub(crate) fn assistant_message_text_response(message: &Message) -> Option<String> {
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

    if let Some(refusal) = assistant_refusal_fallback(content, refusal.as_deref()) {
        segments.push(refusal.to_owned());
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments.join("\n"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    // Null-or-missing tolerated on deserialization: Copilot's chat route
    // (fronting non-OpenAI vendors) can omit either field or send explicit
    // `null`; normalization treats "" as absent.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub finish_reason: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Cached tokens from prompt caching
    #[serde(default)]
    pub cached_tokens: usize,
    /// Audio input tokens, defaulting null or missing values to zero.
    /// Zero is omitted from serialization. [`Usage::to_normalized`] uses the
    /// reported total to determine whether audio is additional to prompt tokens.
    #[serde(
        default,
        deserialize_with = "json_utils::null_or_default",
        skip_serializing_if = "is_zero"
    )]
    pub audio_tokens: usize,
    /// Tokens written to cache on this call. `None` means unreported, not zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
}

/// Whether a counter is absent-as-zero, for `skip_serializing_if`.
fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens reported by reasoning-capable providers.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    pub total_tokens: usize,
    // Not aliased to Mistral's singular `prompt_token_details`: Mistral's
    // embeddings reply carries *both* keys (the singular always `null`), and
    // an alias makes serde reject the document as a duplicate field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Mistral's top-level cached-prompt count, reported beside (or instead
    /// of) `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
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
            num_cached_tokens: None,
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

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl Usage {
    /// Return prompt tokens plus audio only when that sum and output match the total.
    /// Missing output counts are treated as zero for this comparison.
    fn input_tokens(&self) -> usize {
        let audio = self
            .prompt_tokens_details
            .map_or(0, |details| details.audio_tokens);
        let beside = self.prompt_tokens.saturating_add(audio);
        let accounted = beside.saturating_add(self.completion_tokens.unwrap_or(0));
        if audio != 0 && accounted == self.total_tokens {
            beside
        } else {
            self.prompt_tokens
        }
    }

    /// Normalize token accounting, deriving absent output counts from the total.
    /// Cached input prefers prompt details and falls back to `num_cached_tokens`.
    pub fn to_normalized(&self) -> crate::completion::Usage {
        let input_tokens = self.input_tokens();
        let details = self.prompt_tokens_details.as_ref();
        crate::completion::Usage {
            input_tokens: Some(input_tokens as u64),
            // Gateways that omit `completion_tokens` still send the total, so
            // the completion count is the remainder.
            output_tokens: Some(
                self.completion_tokens
                    .unwrap_or_else(|| self.total_tokens.saturating_sub(input_tokens))
                    as u64,
            ),
            total_tokens: Some(self.total_tokens as u64),
            cached_input_tokens: details
                .map(|d| d.cached_tokens as u64)
                .or(self.num_cached_tokens),
            cache_creation_input_tokens: details
                .and_then(|d| d.cache_write_tokens)
                .map(|tokens| tokens as u64),
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens as u64),
            ..Default::default()
        }
    }
}

/// Whether the model matches the GPT-5 through GPT-9 or numeric o-series rules
/// used to select `max_completion_tokens`.
pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    /// Match a single-digit GPT major version at least `lowest`, allowing dot
    /// and hyphen suffixes. Multi-digit major versions do not match.
    fn is_numbered_gpt_family(model: &str, lowest: u32) -> bool {
        model
            .strip_prefix("gpt-")
            .and_then(|rest| rest.split(['.', '-']).next())
            .filter(|major| major.len() == 1)
            .and_then(|major| major.parse::<u32>().ok())
            .is_some_and(|major| major >= lowest)
    }

    /// Match `o` followed by a digit and then an end, hyphen, or another digit.
    fn is_o_series(model: &str) -> bool {
        let mut chars = model.chars();
        chars.next() == Some('o')
            && chars.next().is_some_and(|digit| digit.is_ascii_digit())
            && chars
                .next()
                .is_none_or(|next| next == '-' || next.is_ascii_digit())
    }

    is_numbered_gpt_family(model, 5) || is_o_series(model)
}

/// Serialize a chat-completions request into the body the target endpoint
/// expects, applying the spellings that depend on the endpoint rather than on
/// the request.
///
/// Both the unary and the streaming path build their body through here so the
/// two cannot disagree about what rig sends.
pub(crate) fn request_body(
    request: &CompletionRequest,
    modern_output_cap: bool,
) -> Result<serde_json::Value, EncodeError> {
    let mut body = serde_json::to_value(request)?;

    if modern_output_cap
        && let Some(object) = body.as_object_mut()
        && let Some(max_tokens) = object.remove("max_tokens")
    {
        // Preserve explicit modern caps while removing the rejected legacy key.
        object.entry("max_completion_tokens").or_insert(max_tokens);
    }

    Ok(body)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(flatten)]
    pub additional_params: Option<serde_json::Value>,
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

/// Remove tool messages, assistant tool calls and reasoning, and empty assistant turns.
/// Optionally strip names and flatten content using the supplied separator and
/// text-only guard. With `merge_same_role`, join adjacent user or assistant text
/// messages of the same role with newlines.
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
    /// Whether the endpoint honours an image inside a `role:"tool"` message;
    /// see
    /// [`Quirks::supports_image_tool_results`](crate::providers::openai::wire::Quirks::supports_image_tool_results).
    pub supports_image_tool_results: bool,
    /// Maps `output_schema` to `response_format` when true; drops it with a
    /// warning when false (providers whose APIs reject `json_schema`).
    pub supports_response_format: bool,
    /// Whether `response_format` rides beside advertised tools before the
    /// first tool result; see
    /// [`Quirks::response_format_with_tools`](crate::providers::openai::wire::Quirks::response_format_with_tools).
    pub response_format_with_tools: bool,
    /// Serializes `tools`/`tool_choice` when true; drops them with a warning
    /// when false (providers without tool-calling support).
    pub supports_tools: bool,
    /// Whether the dialect accepts structured reasoning replay on assistant
    /// messages; see
    /// [`Quirks::reasoning_details`](crate::providers::openai::wire::Quirks::reasoning_details).
    ///
    /// When set, a reasoning block replays as a `reasoning_details` entry
    /// carrying its signature, encrypted blob or summary; when clear it
    /// replays as the plain `reasoning_content` string, because a dialect
    /// that never sent the array does not accept it either.
    pub reasoning_details: bool,
}

impl TryFrom<OpenAIRequestParams> for CompletionRequest {
    type Error = EncodeError;

    fn try_from(params: OpenAIRequestParams) -> Result<Self, Self::Error> {
        let OpenAIRequestParams {
            model,
            request: req,
            strict_tools,
            tool_result_array_content,
            supports_image_tool_results,
            supports_response_format,
            response_format_with_tools,
            supports_tools,
            reasoning_details,
        } = params;
        let chat_history = req.chat_history_with_documents();

        let CoreCompletionRequest {
            model: request_model,
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

        let tool_ids =
            crate::providers::internal::tool_call_ids::ToolCallIds::new(&partial_history)
                .map_err(EncodeError::request)?;

        let mut full_history: Vec<Message> = Vec::new();
        full_history.extend(
            partial_history
                .into_iter()
                .enumerate()
                .map(|(position, message)| {
                    message_with_tool_ids(message, position, &tool_ids, reasoning_details)
                })
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten(),
        );

        if full_history.is_empty() {
            return Err(EncodeError::request(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "OpenAI Chat Completions request has no provider-compatible messages after conversion",
            )));
        }

        // Image support is dialect-specific and unavailable during isolated result conversion.
        for msg in &mut full_history {
            if let Message::ToolResult { content, .. } = msg {
                if content.has_image() {
                    if !supports_image_tool_results {
                        // Reject unsupported images instead of silently removing tool output.
                        return Err(EncodeError::request(concat!(
                            "this provider does not accept an image in a tool result. ",
                            "Official OpenAI refuses it on Chat Completions (and the GPT-5 ",
                            "family accepts the request while ignoring the image); use the ",
                            "Responses API, which carries images in `function_call_output`, ",
                            "or a server that sets `SUPPORTS_IMAGE_TOOL_RESULTS` ",
                            "(llama.cpp does)",
                        )));
                    }
                    // An image cannot be flattened to a string, so array form is
                    // forced regardless of `tool_result_array_content`.
                    *content = content.to_array();
                    continue;
                }

                let normalized = if tool_result_array_content {
                    content.to_array()
                } else {
                    ToolResultContentValue::String(content.as_text())
                };

                *content = normalized;
            }
        }

        let history_has_tool_result = history_contains_tool_result(&full_history);

        let (mut tools, tool_choice) = if supports_tools {
            let tool_choice = tool_choice.map(ToolChoice::try_from).transpose()?;
            let tools: Vec<ToolDefinition> = tools
                .into_iter()
                .map(|tool| {
                    let def = ToolDefinition::from(tool);
                    if strict_tools { def.with_strict() } else { def }
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

        // Merge function tools to prevent flattened parameters from replacing typed tools.
        // Leave native tools for dialect-specific request preparation.
        let mut additional_params = additional_params;
        if supports_tools
            && let Some(map) = additional_params
                .as_mut()
                .and_then(serde_json::Value::as_object_mut)
            && let Some(raw_tools) = map.remove("tools")
        {
            let raw_tools =
                serde_json::from_value::<Vec<serde_json::Value>>(raw_tools).map_err(|err| {
                    EncodeError::request(format!(
                        "Invalid OpenAI Chat Completions `additional_params.tools` payload: {err}"
                    ))
                })?;
            let mut remaining = Vec::new();
            for raw_tool in raw_tools {
                let is_function_tool =
                    raw_tool.get("type").and_then(serde_json::Value::as_str) == Some("function");
                if is_function_tool {
                    let tool =
                        serde_json::from_value::<ToolDefinition>(raw_tool).map_err(|err| {
                            EncodeError::request(format!(
                                "Invalid function tool in OpenAI Chat Completions \
                                 `additional_params.tools`: {err}"
                            ))
                        })?;
                    tools.push(tool);
                } else {
                    remaining.push(raw_tool);
                }
            }
            if !remaining.is_empty() {
                map.insert("tools".to_string(), serde_json::Value::Array(remaining));
            }
        }

        if output_schema.is_some() && !supports_response_format {
            tracing::warn!(
                "Structured outputs are not supported by this provider; ignoring output_schema"
            );
        }

        // Defer schemas until a tool result exists unless the dialect supports both at once.
        let should_apply_response_format = output_schema.is_some()
            && supports_response_format
            && (response_format_with_tools || tools.is_empty() || history_has_tool_result);

        let additional_params = if let Some(schema) = output_schema
            && should_apply_response_format
        {
            let (name, schema_value) = super::structured_output_schema(schema);
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

        // The wire rejects tool_choice without advertised tools.
        let tool_choice = tool_choice.filter(|_| !tools.is_empty());
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

fn serialize_assistant_content_vec<S>(
    value: &[AssistantContent],
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
mod tests;

#[cfg(test)]
mod image_tool_result_gate_tests;
