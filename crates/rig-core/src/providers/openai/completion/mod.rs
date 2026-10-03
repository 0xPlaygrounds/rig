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

/// GPT-6 Astra, API ID `gpt-6-astra`: a reasoning model. Chat Completions
/// takes its function tools only at `reasoning_effort: "none"`, which it does
/// not support, so a Chat request with tools (the extractor's included) is
/// refused before it is sent: use the Responses wire.
pub const GPT_6_ASTRA: &str = "gpt-6-astra";

/// GPT-6.1 Sol, API ID `gpt-6.1-sol`: a reasoning model. Chat Completions
/// takes its function tools only at `reasoning_effort: "none"`, which it does
/// not support, so a Chat request with tools (the extractor's included) is
/// refused before it is sent: use the Responses wire.
pub const GPT_6_1_SOL: &str = "gpt-6.1-sol";

/// GPT-6 Sol, API ID `gpt-6-sol`: a reasoning model. Chat Completions takes
/// its function tools only at `reasoning_effort: "none"`: a Chat request with
/// tools is refused before it is sent unless `additional_params` carries
/// `"reasoning_effort": "none"`. Responses takes them at any effort.
pub const GPT_6_SOL: &str = "gpt-6-sol";

/// GPT-6 Luna, API ID `gpt-6-luna`: a reasoning model. Chat Completions takes
/// its function tools only at `reasoning_effort: "none"`: a Chat request with
/// tools is refused before it is sent unless `additional_params` carries
/// `"reasoning_effort": "none"`. Responses takes them at any effort.
pub const GPT_6_LUNA: &str = "gpt-6-luna";

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

/// GPT-5.4, API ID `gpt-5.4`: a reasoning model.
pub const GPT_5_4: &str = "gpt-5.4";

/// GPT-5.4 mini, API ID `gpt-5.4-mini`: a reasoning model.
pub const GPT_5_4_MINI: &str = "gpt-5.4-mini";

/// GPT-5.4 nano, API ID `gpt-5.4-nano`: a reasoning model.
pub const GPT_5_4_NANO: &str = "gpt-5.4-nano";

/// `gpt-5.2` completion model
pub const GPT_5_2: &str = "gpt-5.2";

/// GPT-5.2 Pro, API ID `gpt-5.2-pro`: a reasoning model served on the
/// Responses API only, without structured outputs.
pub const GPT_5_2_PRO: &str = "gpt-5.2-pro";

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
    #[serde(alias = "model", deserialize_with = "deserialize_assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Vec::is_empty")]
        content: Vec<AssistantContent>,
        // OpenAI-compatible providers expose hidden reasoning on this non-standard
        // field, and some require it to be echoed back on assistant tool-call turns.
        // Serialized as `reasoning_content` (llama.cpp/DeepSeek dialect); decoded
        // from either that key or OpenRouter's `reasoning`.
        #[serde(skip_serializing_if = "Option::is_none", rename = "reasoning_content")]
        reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: ToolResultContentValue,
    },
    /// An assistant message as the wire carries it: the provider's own
    /// message, or one rebuilt from a turn's blocks. Request-only: a reply
    /// is read as [`Self::Assistant`].
    #[serde(untagged, skip_deserializing)]
    Native(serde_json::Value),
}

/// An assistant message as compatible providers send it. The two reasoning
/// keys are separate fields because gateways relaying a `reasoning_content`
/// upstream behind a `reasoning` surface send both.
#[derive(Deserialize)]
struct AssistantMessageWire {
    #[serde(default, deserialize_with = "json_utils::string_or_vec")]
    content: Vec<AssistantContent>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    tool_calls: Vec<ToolCall>,
}

/// The fields of [`Message::Assistant`], in declaration order.
type AssistantFields = (
    Vec<AssistantContent>,
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<ToolCall>,
);

/// Decode [`Message::Assistant`], preferring `reasoning_content` over
/// `reasoning` as the streamed delta does.
fn deserialize_assistant<'de, D>(deserializer: D) -> Result<AssistantFields, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let wire = AssistantMessageWire::deserialize(deserializer)?;
    Ok((
        wire.content,
        wire.reasoning_content.or(wire.reasoning),
        wire.refusal,
        wire.name,
        wire.tool_calls,
    ))
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

/// Text or image content in a tool-result message. An image reaches it only
/// on a dialect that reads one
/// ([`Quirks::supports_image_tool_results`](crate::providers::openai::wire::Quirks::supports_image_tool_results)).
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
                name: tool.name.into(),
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
                Self::function(name.as_str())
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
        let tool_call_id = value.call.wire().into_owned();
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
                message::ToolResultContent::Image(image) => Ok(ToolResultContent::Image {
                    image_url: image_url(image)?,
                }),
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

/// The error for media in a form Chat Completions cannot carry, which the
/// adapter replaces before a request is encoded
/// ([`ReplayTarget::encodes`](crate::completion::ReplayTarget::encodes)).
fn unsendable(what: &str) -> message::MessageError {
    message::MessageError::ConversionError(format!(
        "Chat Completions cannot carry {what} in this form"
    ))
}

/// `data` as a URL a content part names: a URL, or typed base64 data as a
/// data URI.
fn media_url(data: DocumentSourceKind, mime: Option<&str>) -> Option<String> {
    match (data, mime) {
        (DocumentSourceKind::Url(url), _) => Some(url),
        (DocumentSourceKind::Base64(data), Some(mime)) => {
            Some(format!("data:{mime};base64,{data}"))
        }
        _ => None,
    }
}

/// The `image_url` an image is sent as.
fn image_url(image: message::Image) -> Result<ImageUrl, message::MessageError> {
    let mime = image.media_type.as_ref().map(MimeType::to_mime_type);
    Ok(ImageUrl {
        url: media_url(image.data, mime).ok_or_else(|| unsendable("an image"))?,
        detail: image.detail,
    })
}

impl TryFrom<message::UserContent> for UserContent {
    type Error = message::MessageError;

    fn try_from(value: message::UserContent) -> Result<Self, Self::Error> {
        let file = |file_data: Option<String>, file_id: Option<String>, filename: Option<&str>| {
            UserContent::File {
                file: FileData {
                    file_data,
                    file_id,
                    filename: filename.map(str::to_owned),
                },
            }
        };
        match value {
            message::UserContent::Text(message::Text { text, .. }) => {
                Ok(UserContent::Text { text })
            }
            // A user image always carries a detail level, `auto` by default.
            message::UserContent::Image(image) => {
                let url = image_url(image)?;
                Ok(UserContent::Image {
                    image_url: ImageUrl {
                        detail: Some(url.detail.unwrap_or_default()),
                        ..url
                    },
                })
            }
            message::UserContent::Document(message::Document {
                data, media_type, ..
            }) => match (data, media_type) {
                (DocumentSourceKind::FileId(id), _) => Ok(file(None, Some(id), None)),
                (DocumentSourceKind::Base64(data), Some(message::DocumentMediaType::PDF)) => {
                    Ok(file(
                        Some(format!("data:application/pdf;base64,{data}")),
                        None,
                        Some("document.pdf"),
                    ))
                }
                // OpenRouter and Mistral fetch a PDF a URL names.
                (DocumentSourceKind::Url(url), Some(message::DocumentMediaType::PDF)) => {
                    Ok(file(Some(url), None, Some("document.pdf")))
                }
                (DocumentSourceKind::String(text), media_type)
                    if media_type != Some(message::DocumentMediaType::PDF) =>
                {
                    Ok(UserContent::Text { text })
                }
                _ => Err(unsendable("a document")),
            },
            message::UserContent::Audio(message::Audio {
                data: DocumentSourceKind::Base64(data),
                media_type,
            }) => Ok(UserContent::Audio {
                input_audio: InputAudio {
                    data,
                    format: media_type.unwrap_or(AudioMediaType::MP3),
                },
            }),
            message::UserContent::Audio(_) => Err(unsendable("audio")),
            message::UserContent::Video(message::Video {
                data, media_type, ..
            }) => Ok(UserContent::Video {
                video_url: VideoUrl {
                    url: media_url(data, media_type.as_ref().map(MimeType::to_mime_type))
                        .ok_or_else(|| unsendable("a video"))?,
                },
            }),
            message::UserContent::ToolResult(_) => Err(message::MessageError::ConversionError(
                "Tool result is in unsupported format".into(),
            )),
        }
    }
}

/// Convert user content into ordered user and tool-result messages.
/// Group adjacent non-tool content. Return conversion errors for unsupported media.
pub fn user_content_to_messages(
    value: impl IntoIterator<Item = message::UserContent>,
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

/// One assistant turn as the message Chat Completions takes, rebuilt from
/// its blocks the way pi rebuilds it: text joined into a `content` string,
/// or a part array when a block is a content part (Mistral's thinking),
/// reasoning under the field it arrived in, `reasoning_details` and other
/// item fields verbatim, and each call with its canonical name and
/// arguments. The provider's own message is never sent. A message with
/// neither content nor tool calls is `None`, as pi skips it: providers
/// reject an empty assistant message.
pub fn assistant_message(turn: message::AssistantMessage) -> Option<Message> {
    use crate::providers::internal::rebuild::{Piece, Rebuilt, call_item};
    let rebuilt = Rebuilt::of(&turn);
    let mut wire = serde_json::Map::new();
    wire.insert("role".to_owned(), "assistant".into());
    if rebuilt.has_parts() {
        let parts: Vec<serde_json::Value> = rebuilt
            .pieces
            .iter()
            .filter_map(|piece| match piece {
                Piece::Text {
                    part: Some(part), ..
                }
                | Piece::Reasoning {
                    part: Some(part), ..
                } => Some(part.clone()),
                Piece::Text { text, part: None } => (!text.trim().is_empty())
                    .then(|| serde_json::json!({"type": "text", "text": text})),
                Piece::Opaque(item) => item.get("type").is_some().then(|| item.clone()),
                Piece::Reasoning { part: None, .. } => None,
            })
            .collect();
        wire.insert("content".to_owned(), parts.into());
    } else {
        let text = rebuilt.text();
        if !text.is_empty() {
            wire.insert("content".to_owned(), text.into());
        }
    }
    for (field, text) in rebuilt.reasoning(None) {
        wire.insert(field, text.into());
    }
    for piece in &rebuilt.pieces {
        if let Piece::Opaque(serde_json::Value::Object(fields)) = piece
            && !fields.contains_key("type")
        {
            wire.extend(fields.clone());
        }
    }
    wire.extend(rebuilt.fields);
    let calls: Vec<serde_json::Value> = rebuilt
        .calls
        .into_iter()
        .map(|(call, item)| {
            let id = call.id.wire().into_owned();
            call_item(&call, item, Some(id), true)
        })
        .collect();
    if !calls.is_empty() {
        wire.insert("tool_calls".to_owned(), calls.into());
    }
    let has_content = wire.contains_key("audio")
        || match wire.get("content") {
            Some(serde_json::Value::String(text)) => !text.is_empty(),
            Some(serde_json::Value::Array(parts)) => !parts.is_empty(),
            _ => false,
        };
    (has_content || wire.contains_key("tool_calls"))
        .then(|| Message::Native(serde_json::Value::Object(wire)))
}

/// The id slots of the tool calls a converted message carries, in order. A
/// call that came without an id gets an empty one, for the request's id
/// spelling to fill.
pub(crate) fn call_id_slots(message: &mut Message) -> Vec<&mut String> {
    match message {
        Message::ToolResult { tool_call_id, .. } => vec![tool_call_id],
        Message::Native(item) => item
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
            .into_iter()
            .flatten()
            .filter_map(|call| {
                let id = call
                    .as_object_mut()?
                    .entry("id")
                    .or_insert_with(|| serde_json::Value::String(String::new()));
                match id {
                    serde_json::Value::String(id) => Some(id),
                    _ => None,
                }
            })
            .collect(),
        Message::Assistant { tool_calls, .. } => {
            tool_calls.iter_mut().map(|call| &mut call.id).collect()
        }
        Message::System { .. } | Message::User { .. } => Vec::new(),
    }
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::system(&content)]),
            message::Message::User { content } => user_content_to_messages(content),
            message::Message::Assistant(turn) => Ok(assistant_message(turn).into_iter().collect()),
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            // Use the same wire-handle selection as tool-result conversion.
            id: tool_call.id.wire().into_owned(),
            r#type: ToolType::default(),
            function: Function {
                name: tool_call.function.name.into(),
                arguments: serde_json::Value::Object(tool_call.function.arguments),
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

/// OpenAI's chat-completions reply.
pub type CompletionResponse = ChatCompletionResponse<Usage>;

/// A chat-completions reply over the accounting `U` and choice `C`. Compatible
/// providers that add usage counters or choice fields read their replies back
/// with their own `U` and `C`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionResponse<U, C = Choice> {
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
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls",
        bound(deserialize = "C: serde::de::DeserializeOwned")
    )]
    pub choices: Vec<C>,
    pub usage: Option<U>,
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
        && let Some(max_tokens) = object.shift_remove("max_tokens")
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

pub struct OpenAIRequestParams {
    pub model: String,
    pub request: CoreCompletionRequest,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
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
}

impl TryFrom<OpenAIRequestParams> for CompletionRequest {
    type Error = EncodeError;

    fn try_from(params: OpenAIRequestParams) -> Result<Self, Self::Error> {
        let OpenAIRequestParams {
            model,
            request: req,
            strict_tools,
            tool_result_array_content,
            supports_response_format,
            response_format_with_tools,
            supports_tools,
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

        // Conversion can split text into separate messages, but keeps every
        // tool call and result in source order.
        let mut full_history = crate::providers::internal::wire_ids::WireIds::convert(
            chat_history,
            Vec::<Message>::try_from,
            call_id_slots,
        )?;

        if full_history.is_empty() {
            return Err(EncodeError::request(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "OpenAI Chat Completions request has no provider-compatible messages after conversion",
            )));
        }

        for msg in &mut full_history {
            if let Message::ToolResult { content, .. } = msg {
                // An image reaches a tool result only where the dialect reads
                // it, and has no string form.
                *content = if tool_result_array_content || content.has_image() {
                    content.to_array()
                } else {
                    ToolResultContentValue::String(content.as_text())
                };
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
            && let Some(raw_tools) = map.shift_remove("tools")
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

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod image_tool_result_gate_tests;
