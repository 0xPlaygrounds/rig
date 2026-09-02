// ================================================================
// OpenAI Completion API
// ================================================================

use super::client::ApiResponse;
use crate::completion::NormalizeCompletionResponse;
use crate::completion::{CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils::string_or_vec;
use crate::message::{AudioMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::providers::internal::completion_send::send_completion;
use crate::telemetry::{
    CompletionOperation, CompletionSpanBuilder, ProviderResponseExt, SpanCombinator,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{completion, json_utils, message};
use serde::{Deserialize, Serialize, Serializer};
use std::convert::Infallible;
use std::fmt;
use tracing::Instrument;

use std::str::FromStr;

pub mod streaming;

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
        audio: Option<AudioAssistant>,
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

/// One content part of a tool-result message.
///
/// Text is the only part official OpenAI accepts here; an image is refused with
/// a 400 on `gpt-4o` and, worse, accepted-and-discarded on the GPT-5 family.
/// Some OpenAI-compatible servers do honour an image — llama.cpp delivers one to
/// the model, measured — so the variant exists and emitting it is gated on
/// [`super::completion::OpenAICompatibleProvider::SUPPORTS_IMAGE_TOOL_RESULTS`].
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
    pub fn from_string(s: String, use_array_format: bool) -> Self {
        if use_array_format {
            ToolResultContentValue::Array(vec![ToolResultContent::from(s)])
        } else {
            ToolResultContentValue::String(s)
        }
    }

    /// The text of this tool result, with any non-text parts skipped.
    ///
    /// Lossy by construction: an image part has no textual rendering, so a
    /// caller flattening a result that carries one loses it. That is why
    /// the per-provider normalization in `TryFrom<OpenAIRequestParams>` refuses
    /// rather than flattens when a provider cannot carry the image.
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

    /// Convert into rig's tool-result content blocks, preserving image parts.
    ///
    /// The counterpart of the outbound conversion. A round trip through the
    /// wire and back must not quietly become text-only, or a replayed history
    /// says less than the one that produced it.
    pub fn into_message_content(self) -> Vec<message::ToolResultContent> {
        match self {
            ToolResultContentValue::String(text) => vec![message::ToolResultContent::text(text)],
            ToolResultContentValue::Array(parts) => parts
                .into_iter()
                .map(|part| match part {
                    ToolResultContent::Text { text } => message::ToolResultContent::text(text),
                    ToolResultContent::Image { image_url } => {
                        // A base64 data URI round-trips back to its parts;
                        // anything else stays a URL reference.
                        match parse_image_data_uri(&image_url.url) {
                            Some((mime, data)) => message::ToolResultContent::image_base64(
                                data,
                                message::ImageMediaType::from_mime_type(mime),
                                image_url.detail,
                            ),
                            None => message::ToolResultContent::image_url(
                                image_url.url,
                                None,
                                image_url.detail,
                            ),
                        }
                    }
                })
                .collect(),
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

/// Split a base64 data URI into `(mime, base64)`, or `None` for a plain URL.
///
/// `rsplit_once` on the marker rather than `split_once`, so a URL that happens
/// to contain `;base64,` earlier does not truncate the payload.
fn parse_image_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (mime, data) = rest.rsplit_once(";base64,")?;
    (!data.is_empty()).then_some((mime, data))
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
        // The wire requires a non-empty correlator: the provider-issued
        // call id when one exists, else rig's minted handle — which is
        // unique and non-empty by construction, unlike the old empty-
        // string sentinel.
        let tool_call_id = value.wire_call_id().to_owned();
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
                // Represented here, refused (or not) at the wire step: whether
                // an image may ride in a `role:"tool"` message is a per-server
                // fact, and this conversion has no provider context. See
                // `ToolResultContentValue::normalize_for_wire`.
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
                        // Deliberately payload-free: `DocumentSourceKind::Raw`
                        // Debug-formats as the entire byte vector, so `{other:?}`
                        // would dump a whole image into an error string. The
                        // user-image path one screen away avoids this the same way.
                        DocumentSourceKind::Raw(_) => {
                            return Err(message::MessageError::ConversionError(
                                "raw image bytes are not supported in a tool result; encode as \
                                 base64 first"
                                    .into(),
                            ));
                        }
                        // Named, never Debug-printed: `FileId` and `String`
                        // carry caller data and `Unknown` carries nothing worth
                        // quoting.
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

/// Convert rig user content into OpenAI chat messages.
///
/// This was `impl TryFrom<OneOrMany<UserContent>> for Vec<Message>`. With the
/// container gone both sides are foreign types, so the orphan rule forbids the
/// impl and it becomes a named function. It stays `pub`: the impl was reachable
/// from downstream code, and quietly demoting it to a private helper would
/// narrow the public API under cover of a type change.
pub fn user_content_to_messages(
    value: Vec<message::UserContent>,
) -> Result<Vec<Message>, message::MessageError> {
    fn flush_user_content(messages: &mut Vec<Message>, pending: &mut Vec<UserContent>) {
        // An empty flush is a legal no-op — it fires between consecutive
        // tool-result groups — not a conversion error. This early return is
        // the only emptiness decision here; the pushed content is non-empty
        // because of it.
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

/// Convert rig assistant content into OpenAI chat messages.
///
/// Free function for the same orphan-rule reason as
/// [`user_content_to_messages`], and `pub` for the same API-surface reason.
pub fn assistant_content_to_messages(
    value: Vec<message::AssistantContent>,
) -> Result<Vec<Message>, message::MessageError> {
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
            .map(std::convert::Into::into)
            .collect::<Vec<_>>(),
        reasoning_details: Vec::new(),
        images: Vec::new(),
    }])
}

impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::System { content } => Ok(vec![Message::system(&content)]),
            message::Message::User { content } => user_content_to_messages(content),
            message::Message::Assistant { content, .. } => assistant_content_to_messages(content),
        }
    }
}

impl From<message::ToolCall> for ToolCall {
    fn from(tool_call: message::ToolCall) -> Self {
        Self {
            // Keep the assistant echo consistent with the tool-result side:
            // the provider-issued call id when one exists (e.g. a
            // Responses-API history replayed via chat completions), else
            // rig's minted handle — never empty.
            id: tool_call.wire_call_id().to_owned(),
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
        message::ToolCall::from_wire(
            tool_call.id,
            message::ToolFunction {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        )
    }
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        Ok(match message {
            Message::User { content, .. } => message::Message::User {
                content: content.into_iter().map(std::convert::Into::into).collect(),
            },
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                refusal,
                ..
            } => {
                let mut assistant_content = Vec::new();

                if let Some(reasoning) = reasoning
                    && !reasoning.is_empty()
                {
                    assistant_content.push(message::AssistantContent::reasoning(reasoning));
                }

                // Either/or, not both: the fallback fires only when no part
                // carried text, so every part left is an empty one. Appending
                // them anyway would put an empty text block on the wire beside
                // the refusal and make this view of the message disagree with
                // the one `normalize` builds, which drops empty parts.
                if let Some(refusal) = assistant_refusal_fallback(&content, refusal.as_deref()) {
                    assistant_content.push(message::AssistantContent::text(refusal));
                } else {
                    assistant_content.extend(content.into_iter().map(|content| match content {
                        AssistantContent::Text { text, .. } => {
                            message::AssistantContent::text(text)
                        }
                        AssistantContent::Refusal { refusal } => {
                            message::AssistantContent::text(refusal)
                        }
                    }));
                }

                assistant_content.extend(
                    tool_calls
                        .into_iter()
                        .map(|tool_call| Ok(message::AssistantContent::ToolCall(tool_call.into())))
                        .collect::<Result<Vec<_>, _>>()?,
                );

                message::Message::Assistant {
                    id: None,
                    content: crate::message::require_non_empty(assistant_content, || {
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
                // OpenAI chat tool messages carry no tool name; this
                // conversion is lossy for name-keyed wires.
                // Every part is carried back, images included. Flattening with
                // `as_text()` would drop an image silently — the same loss the
                // outbound gate refuses to commit, and worse here because it
                // used to be impossible: before `ToolResultContent` grew an
                // image variant, such a body failed to deserialize at all, so
                // the loss was at least visible.
                content: vec![message::UserContent::tool_result_from_wire(
                    tool_call_id,
                    "",
                    content.into_message_content(),
                )],
            },

            // System messages should get stripped out when converting messages, this is just a
            // stop gap to avoid obnoxious error handling or panic occurring.
            Message::System { content, .. } => message::Message::User {
                content: content
                    .into_iter()
                    .map(|content| message::UserContent::text(content.text))
                    .collect(),
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

/// Normalize an OpenAI-compatible chat completion response.
///
/// The provider descriptor name is an *input* rather than a constant: this same
/// wire shape is shared by every OpenAI-compatible provider, so baking in
/// `"openai"` here would mislabel Groq, Together, DeepSeek and the rest. Taking
/// it as part of the conversion makes the correct name impossible to forget.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        use crate::providers::internal::openai_chat_completions_compatible as compat;

        let usage = self
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        compat::normalize_openai_response(
            provider,
            &self.choices,
            Some(self.id.as_str()),
            Some(self.model.as_str()),
            usage,
            |choice| choice.finish_reason.as_str(),
            |choice| match &choice.message {
                Message::Assistant {
                    content: wire_content,
                    tool_calls,
                    reasoning,
                    refusal,
                    ..
                } => {
                    let mut content = wire_content
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

                    if let Some(refusal) =
                        assistant_refusal_fallback(wire_content, refusal.as_deref())
                    {
                        content.push(completion::AssistantContent::text(refusal));
                    }

                    if let Some(reasoning) = reasoning {
                        // llama.cpp exposes hidden reasoning on a separate non-standard field.
                        // Keep it structured here so the non-streaming path matches streaming
                        // behavior and does not pollute plain-text response surfaces.
                        content.push(completion::AssistantContent::reasoning(reasoning));
                    }

                    content.extend(tool_calls.iter().map(|call| {
                        completion::AssistantContent::tool_call(
                            &call.id,
                            &call.function.name,
                            call.function.arguments.clone(),
                        )
                    }));
                    Some(content)
                }
                _ => None,
            },
        )
    }
}

impl ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        Some(self.id.as_str())
    }

    fn response_model_name(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn text_response(&self) -> Option<String> {
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

    fn usage(&self) -> Option<Self::Usage> {
        self.usage
    }
}

/// The assistant message's top-level `refusal`, when it is the turn's only
/// visible text.
///
/// This wire spells a refusal as a *sibling* of `content`
/// (`{"content": null, "refusal": "I'm sorry, …"}`); the `refusal` **content
/// part** modeled by [`AssistantContent::Refusal`] is the Responses API's
/// shape, which chat completions never sends. Every path that reads `content`
/// alone therefore drops a real refusal entirely, so all of them route the
/// fallback through here — one home for the rule, and no way for the raw text
/// view and the normalized response to disagree about whether a refusal is
/// content.
///
/// The verdict is taken from the wire parts themselves rather than from
/// whatever each caller built out of them, so a caller that discards empty
/// parts and one that keeps them cannot disagree about when the fallback
/// applies.
///
/// This is a *whole-message* rule and the three unary paths share it. The
/// streaming path cannot: it decides per delta, before it knows whether text
/// arrives later
/// ([`delta_text`](super::completion::streaming), which prefers a delta's own
/// content and falls back to its refusal). The two therefore agree on every
/// shape this wire has been observed to send — a refusal turn holds `content`
/// at `null` for its whole length — but would differ on a turn mixing both,
/// where this rule keeps only the text and the streaming rule would deliver
/// both in arrival order. That shape is pinned in
/// `delta_text_prefers_content_over_a_simultaneous_refusal` so the difference
/// is recorded rather than assumed away.
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
    /// Normalize this provider usage payload into rig's [`crate::completion::Usage`].
    pub fn to_normalized(&self) -> crate::completion::Usage {
        let mut usage = crate::providers::internal::completion_usage(
            self.prompt_tokens as u64,
            self.completion_tokens
                .unwrap_or_else(|| self.total_tokens.saturating_sub(self.prompt_tokens))
                as u64,
            self.total_tokens as u64,
            self.prompt_tokens_details
                .as_ref()
                .map_or(0, |d| d.cached_tokens as u64),
        );
        usage.reasoning_tokens = self
            .completion_tokens_details
            .as_ref()
            .map_or(0, |d| d.reasoning_tokens as u64);
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

    /// Response header carrying the provider's transport request id, when the
    /// provider reports one (OpenAI sends `x-request-id`). `None` — the
    /// default — means the provider does not report one; the normalized
    /// response's `provider_request_id` is then `None`, never an error.
    const REQUEST_ID_HEADER: Option<&'static str> = None;

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

    /// Whether this server honours an image inside a `role:"tool"` message.
    ///
    /// `false` everywhere by default, because official OpenAI does not: Chat
    /// Completions answers 400 on `gpt-4o`/`gpt-4o-mini`, and on the GPT-5
    /// family it answers 200 with the image discarded and the model describing
    /// something it never received. An array of *text* parts is fine on both, so
    /// this is about the image, not about array content.
    ///
    /// Some OpenAI-compatible servers do honour it. Measured on llama.cpp
    /// (`b1-6d05498`, Qwen3-VL-2B): a magenta/green/yellow square handed back
    /// through a tool is named correctly 3/3, matching a control that sends the
    /// same bytes in a `user` message — so the image genuinely reaches the
    /// model. Providers whose server does that set this `true`.
    ///
    /// When `false`, a tool result carrying an image is refused before the
    /// request leaves the process rather than being flattened to text (which
    /// would silently drop it) or sent (which the provider answers with a 400,
    /// or worse, accepts and ignores).
    const SUPPORTS_IMAGE_TOOL_RESULTS: bool = false;

    /// Map a streamed terminal reason for this compatible provider.
    ///
    /// The normalized Chat Completions field is the default contract. Gateway
    /// providers that also expose an upstream-native reason can override this
    /// to apply their documented precedence without teaching the shared wire
    /// adapter provider names or native vocabularies.
    fn map_streaming_finish_reason(
        &self,
        finish_reason: Option<&str>,
        _native_finish_reason: Option<&str>,
    ) -> Option<crate::completion::FinishReason> {
        finish_reason
            .filter(|reason| !reason.is_empty())
            .map(crate::providers::internal::openai_chat_completions_compatible::map_openai_finish_reason)
    }

    /// Whether `model`'s endpoint rejects the legacy `max_tokens` field and
    /// requires `max_completion_tokens` instead.
    ///
    /// OpenAI's reasoning-class models answer a capped request with
    /// `"Unsupported parameter: 'max_tokens' is not supported with this model.
    /// Use 'max_completion_tokens' instead."`, so such a request cannot
    /// succeed at all until the field is respelled.
    ///
    /// Scoped to the model rather than applied to every request on purpose:
    /// this same extension is how rig reaches OpenAI-*compatible* servers
    /// (mistral.rs, vLLM, llama.cpp, gateways), whose endpoints mostly know
    /// only the legacy field, and OpenAI's own non-reasoning models still take
    /// it. Everything outside the returned set keeps the bytes it always sent.
    /// The default is `false` — a provider that has not been observed to
    /// reject the legacy field says so by saying nothing.
    ///
    /// Azure OpenAI deliberately keeps the default even though it fronts the
    /// same models: an Azure model handle is a *deployment* name chosen by the
    /// account owner, so it carries no family information to classify. A
    /// capped reasoning deployment there still gets the provider's explicit
    /// `Unsupported parameter` error, which is the honest outcome until Azure
    /// can be given a signal that does not require guessing.
    fn requires_modern_output_cap(&self, model: &str) -> bool {
        let _ = model;
        false
    }

    /// The usage payload parsed from streaming chunks and carried on the
    /// final streaming response. OpenAI's [`Usage`] for most providers;
    /// providers with richer usage accounting (e.g. Mistral's cached-token
    /// fallbacks, DeepSeek's cache hit/miss counters) substitute their own.
    type StreamingUsage: Clone
        + Default
        + Into<crate::completion::Usage>
        + Serialize
        + serde::de::DeserializeOwned
        + Unpin
        + WasmCompatSend
        + WasmCompatSync
        + 'static;

    /// The chat-completions payload this provider returns.
    ///
    /// The normalization bound is stated over `(&str, Self::Response)` so the
    /// provider descriptor name is threaded through the conversion instead of
    /// being hardcoded by whichever wire type happens to implement it.
    type Response: serde::de::DeserializeOwned
        + Serialize
        + crate::telemetry::ProviderResponseExt<Usage: Into<crate::completion::Usage>>
        + crate::completion::NormalizeCompletionResponse
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
        CompletionRequest::try_from(OpenAIRequestParams {
            model,
            request,
            strict_tools: options.strict_tools,
            tool_result_array_content: options.tool_result_array_content,
            supports_response_format: Self::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: Self::SUPPORTS_TOOLS,
            supports_image_tool_results: Self::SUPPORTS_IMAGE_TOOL_RESULTS,
        })
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

    /// Map a provider-specific streaming detail payload onto a complete
    /// reasoning block — its identity and content — that the stream emits as
    /// the turn's own output. OpenRouter's `reasoning_details` entries of type
    /// `reasoning.encrypted` are the in-tree case: the wire carries them with
    /// `reasoning: null`, so this hook is the only place they can reach the
    /// aggregated choice (and, from there, the next turn's request).
    ///
    /// A detail maps to *either* a reasoning block or a
    /// [`decoration`](Self::decorate_streaming_tool_call), never both.
    fn streaming_detail_reasoning(
        &self,
        detail: &serde_json::Value,
    ) -> Option<(
        crate::streaming::StreamPartId,
        Option<crate::streaming::WireId>,
        crate::message::ReasoningContent,
    )> {
        let _ = detail;
        None
    }

    /// Extract a signature-only reasoning detail from a streamed compatible
    /// response. The default wire has no such extension; gateway providers
    /// can attach the signature to the shared plaintext reasoning lifecycle.
    fn streaming_reasoning_signature(&self, detail: &serde_json::Value) -> Option<String> {
        let _ = detail;
        None
    }

    /// Decorate a streamed tool call from a provider-specific streaming
    /// detail payload, matched by its established provider id. Most
    /// OpenAI-compatible providers do not emit such details.
    ///
    /// The decoration is an adapter-level event rewrite: it rides the
    /// adapter's tool-input end event onto the completed call; fragment
    /// assembly itself lives in the shared accumulator.
    fn decorate_streaming_tool_call(
        &self,
        detail: &serde_json::Value,
    ) -> Option<crate::streaming::ToolCallDecoration> {
        let _ = detail;
        None
    }
}

impl OpenAICompatibleProvider for super::OpenAICompletions {
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

    type StreamingUsage = Usage;
    type Response = CompletionResponse;

    fn requires_modern_output_cap(&self, model: &str) -> bool {
        is_openai_reasoning_model(model)
    }
}

/// Whether `model` names one of OpenAI's reasoning families, which take the
/// output cap only as `max_completion_tokens`.
///
/// Matched by family prefix rather than by an exhaustive list of releases: the
/// families are `gpt-5` and up, and the `o`-series (`o1`, `o3-mini`,
/// `o4-mini`, …), and each gains dated snapshots and size variants that an
/// enumerated list could not keep up with. A future family this misses keeps
/// today's behavior — the legacy field, and the provider's own explicit
/// `Unsupported parameter` error — rather than silently sending a field some
/// other backend does not know.
pub(crate) fn is_openai_reasoning_model(model: &str) -> bool {
    /// `gpt-5` … `gpt-9`, in any spelling the family uses (`gpt-5`,
    /// `gpt-5.1`, `gpt-5-nano`, `gpt-5-2025-08-07`).
    ///
    /// The major version is a single digit on purpose. Every released
    /// generation is spelled `gpt-<digit>` or `gpt-<digit>.<minor>`, so a
    /// multi-digit run (`gpt-45`, or a compatible server's own model name) is
    /// not a generation number and must not be read as one. A hypothetical
    /// `gpt-10` would fall through to the legacy field — today's behavior, and
    /// a visible provider error — rather than a field its backend may not know.
    fn is_numbered_gpt_family(model: &str, lowest: u32) -> bool {
        model
            .strip_prefix("gpt-")
            .and_then(|rest| rest.split(['.', '-']).next())
            .filter(|major| major.len() == 1)
            .and_then(|major| major.parse::<u32>().ok())
            .is_some_and(|major| major >= lowest)
    }

    /// `o1`, `o3`, `o4`, … — but not `openai-…` or any other `o` word.
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
) -> Result<serde_json::Value, CompletionError> {
    let mut body = serde_json::to_value(request)?;

    if modern_output_cap
        && let Some(object) = body.as_object_mut()
        && let Some(max_tokens) = object.remove("max_tokens")
    {
        // A caller who spelled the modern field themselves (through
        // `additional_params`) keeps their own value; the legacy key still has
        // to go, since reasoning models reject its mere presence — and behind
        // this endpoint there is no backend that wants it.
        object.entry("max_completion_tokens").or_insert(max_tokens);
    }

    Ok(body)
}

/// A chat-completions model over any [`OpenAICompatibleProvider`] extension.
/// This is the advertised path for OpenAI-compatible providers; see the
/// provider checklist in [`crate::providers`].
#[derive(Clone)]
pub struct GenericCompletionModel<Ext, H = crate::http_client::BoxedHttpClient> {
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
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    GenericCompletionModel<super::OpenAICompletions, H>;

impl<Ext, H> GenericCompletionModel<Ext, H> {
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
    /// See [`OpenAICompatibleProvider::SUPPORTS_IMAGE_TOOL_RESULTS`].
    pub supports_image_tool_results: bool,
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
            supports_image_tool_results,
            supports_response_format,
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

        let mut partial_history = Vec::new();
        partial_history.extend(chat_history);

        let mut full_history: Vec<Message> = Vec::new();
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten(),
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

        // Per-provider normalization of tool-result content. This is the only
        // place with both the parts and the provider's capabilities in hand —
        // `TryFrom<message::ToolResult>` has neither.
        for msg in &mut full_history {
            if let Message::ToolResult { content, .. } = msg {
                if content.has_image() {
                    if !supports_image_tool_results {
                        // Refused rather than flattened: `as_text()` would drop
                        // the image and send a tool result that silently says
                        // less than the caller asked for. Official OpenAI
                        // answers this shape with a 400 on gpt-4o and, on the
                        // GPT-5 family, a 200 with the image discarded — so a
                        // local error naming the constraint beats both.
                        return Err(CompletionError::RequestError(
                            concat!(
                                "this provider does not accept an image in a tool result. ",
                                "Official OpenAI refuses it on Chat Completions (and the GPT-5 ",
                                "family accepts the request while ignoring the image); use the ",
                                "Responses API, which carries images in `function_call_output`, ",
                                "or a server that sets `SUPPORTS_IMAGE_TOOL_RESULTS` ",
                                "(llama.cpp does)",
                            )
                            .into(),
                        ));
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

        // `additional_params` is flattened into the serialized request, so a raw
        // `tools` array left in it would silently replace the typed `tools`
        // field (the body is built via `serde_json::to_value`, where the
        // flattened key wins). Merge its function tools into the typed list
        // instead, mirroring the Responses API path (issue #1890). Entries that
        // are not function tools stay behind for the provider's
        // `prepare_request` hook — Groq, for one, folds its native tools
        // (`{"type": "browser_search"}`, ...) into `compound_custom` from there.
        let mut additional_params = additional_params;
        if supports_tools
            && let Some(map) = additional_params
                .as_mut()
                .and_then(serde_json::Value::as_object_mut)
            && let Some(raw_tools) = map.remove("tools")
        {
            let raw_tools =
                serde_json::from_value::<Vec<serde_json::Value>>(raw_tools).map_err(|err| {
                    CompletionError::RequestError(
                        format!(
                            "Invalid OpenAI Chat Completions `additional_params.tools` payload: {err}"
                        )
                        .into(),
                    )
                })?;
            let mut remaining = Vec::new();
            for raw_tool in raw_tools {
                let is_function_tool =
                    raw_tool.get("type").and_then(serde_json::Value::as_str) == Some("function");
                if is_function_tool {
                    let tool =
                        serde_json::from_value::<ToolDefinition>(raw_tool).map_err(|err| {
                            CompletionError::RequestError(
                                format!(
                                    "Invalid function tool in OpenAI Chat Completions \
                                 `additional_params.tools`: {err}"
                                )
                                .into(),
                            )
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
            supports_image_tool_results: false,
            supports_tools: true,
        })
    }
}

impl<Ext, H> GenericCompletionModel<Ext, H>
where
    Ext: OpenAICompatibleProvider,
{
    /// Whether outgoing requests for `model` spell the output-token cap
    /// `max_completion_tokens`; see
    /// [`OpenAICompatibleProvider::requires_modern_output_cap`].
    ///
    /// `model` is the request's resolved model, not the handle's: a per-request
    /// override changes which endpoint answers, so it has to decide the
    /// spelling too.
    pub(crate) fn sends_modern_output_cap(&self, model: &str) -> bool {
        self.client.provider().requires_modern_output_cap(model)
    }
}

impl<Ext, H> GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: crate::client::Provider
        + OpenAICompatibleProvider
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    H: Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a chat completion and return the provider's own wire response.
    ///
    /// This is the escape hatch for provider-specific fields rig does not
    /// normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one
    /// network request either way.
    ///
    /// The transport request id is not on the wire type and is dropped here;
    /// use [`Self::raw_completion_with_request_id`] when the typed route must
    /// reproduce everything `completion` returns.
    pub async fn raw_completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<Ext::Response, CompletionError> {
        self.raw_completion_with_request_id(completion_request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_completion`] plus the transport request id from the
    /// provider's request-id response header ([`OpenAICompatibleProvider::REQUEST_ID_HEADER`]).
    ///
    /// The pair exists because the wire type is substitutable — `Ext::Response`
    /// is whatever the compatible provider parses — so the transport id cannot
    /// live on it, while the normalized [`completion::CompletionResponse`]
    /// carries one. Without this method, `raw_completion(..)` followed by
    /// [`normalize`](crate::completion::NormalizeCompletionResponse::normalize)
    /// would silently lack the `provider_request_id` that
    /// [`CompletionModel::completion`](completion::CompletionModel::completion)
    /// reports — the typed escape hatch would not reproduce the normalized
    /// path. Reassemble with
    /// [`with_optional_provider_request_id`](completion::CompletionResponse::with_optional_provider_request_id).
    pub async fn raw_completion_with_request_id(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<(Ext::Response, Option<String>), CompletionError> {
        let system_instructions = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let options = CompletionModelOptions {
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            prompt_caching: self.prompt_caching,
        };
        let mut request = self.client.provider().build_completion_request(
            self.model.clone(),
            completion_request,
            options,
        )?;
        self.client.provider().prepare_request(&mut request)?;
        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &request.model,
            CompletionOperation::Chat,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();

        let modern_output_cap = self.sends_modern_output_cap(&request.model);
        let mut request_body = request_body(&request, modern_output_cap)?;
        self.client
            .provider()
            .finalize_request_body_with_options(&mut request_body, options)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI Chat Completions completion request",
            &request_body,
        );

        let body = serde_json::to_vec(&request_body)?;
        // Deliberately the configured model, not the per-request override:
        // Azure's deployment URL is pinned to the model handle.
        let path = self.client.provider().completion_path(&self.model);

        let req = self
            .client
            .post(&path)?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_completion::<_, ApiResponse<Ext::Response>, _>(
            &self.client,
            req,
            "OpenAI Chat Completions completion",
            Ext::REQUEST_ID_HEADER,
            |response| {
                let span = tracing::Span::current();
                span.record_response_metadata(response);
                let usage = response.usage().map(Into::into).unwrap_or_default();
                span.record_token_usage(&usage);
            },
        )
        .instrument(span)
        .await
    }
}

impl<Ext, H> completion::CompletionModel for GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
    Ext: crate::client::Provider
        + OpenAICompatibleProvider
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    H: Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    // OpenAI Chat Completions *defers* `response_format` while tools are present
    // and no tool result exists yet (see `should_apply_response_format`), then
    // applies it once a tool result is in the history. So the native constraint
    // does not suppress tool calls — they compose — which is what this flag
    // governs. (Caveat: a turn-1 answer with no tool call is therefore not
    // schema-constrained; `Native` is "guaranteed" only once tools have run.)
    // See issue #1928.
    fn capabilities(&self) -> completion::ProviderCapabilities {
        // Providers that drop `output_schema` (SUPPORTS_RESPONSE_FORMAT =
        // false) cannot compose native structured output with tools; the
        // agent then falls back to tool-mode enforcement as their
        // pre-migration hand-rolled models did.
        completion::ProviderCapabilities::default()
            .with_native_output_tool_composition(Ext::SUPPORTS_RESPONSE_FORMAT)
    }

    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `normalize` consumes the raw value.
        let (response, provider_request_id) = self
            .raw_completion_with_request_id(completion_request)
            .await?;
        let captured = serde_json::to_value(&response)?;
        Ok(response
            .normalize(Ext::PROVIDER_NAME)?
            .with_optional_provider_request_id(provider_request_id)
            .with_raw(captured))
    }

    async fn stream(
        &self,
        request: CoreCompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        GenericCompletionModel::stream(self, request).await
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
