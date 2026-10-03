use crate::error::ProviderError;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};
use thiserror::Error;

/// A provider-agnostic chat message.
///
/// Messages are role-tagged and may contain one or many content items, including
/// text, images, audio, documents, tool calls, and tool results. Provider modules
/// are responsible for translating these generic messages into provider-native
/// request bodies. That conversion may be lossy when a provider does not support
/// a particular content type.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// System message containing instruction text.
    System { content: String },

    /// User message containing one or more content types defined by `UserContent`.
    User { content: Vec<UserContent> },

    /// An assistant turn: its blocks in the order the provider produced
    /// them, and where they came from.
    Assistant(AssistantMessage),
}

mod identity;
mod native;

pub use identity::{CallId, EmptyCallId, EmptyToolName, LocalCallId, ProviderCallId, ToolName};
pub use native::{Api, Fingerprint, Native, Opaque, Origin, StopReason};

/// One assistant turn.
///
/// `content` holds one block per provider output item, in the order the
/// provider produced them. `origin` names the wire, provider and model that
/// produced the turn; a hand-built turn has none and always replays from its
/// canonical fields.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct AssistantMessage {
    /// The blocks, in provider order.
    pub content: Vec<AssistantContent>,
    /// Who produced the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
    /// How the turn ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopReason>,
}

impl AssistantMessage {
    /// A hand-built turn of `content`: no origin or stop.
    pub fn new(content: Vec<AssistantContent>) -> Self {
        Self {
            content,
            ..Self::default()
        }
    }

    /// The tool calls, in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
    }
}

/// Shared error text for an invalid empty response choice.
/// Provider decoders must exempt legal empty outcomes, including recognized
/// output truncation, before calling [`require_non_empty_response`].
pub const EMPTY_RESPONSE_ERROR: &str = "Response contained no message or tool call (empty)";

/// Returns `items` unchanged unless the list is empty, then calls `error` once.
/// Does not inspect individual items: empty text can carry replay signatures.
/// Request conversions that discard content must validate the converted list
/// when their wire requires at least one block.
pub fn require_non_empty<T, E>(items: Vec<T>, error: impl FnOnce() -> E) -> Result<Vec<T>, E> {
    if items.is_empty() {
        return Err(error());
    }
    Ok(items)
}

/// Returns a response error using [`EMPTY_RESPONSE_ERROR`] for an empty list.
/// Callers must handle provider-legal empty outcomes before invoking this guard.
pub fn require_non_empty_response<T>(items: Vec<T>) -> Result<Vec<T>, ProviderError> {
    require_non_empty(items, || {
        ProviderError::Response(EMPTY_RESPONSE_ERROR.to_owned())
    })
}

/// Returns `None` for an empty list or `Some(items)` otherwise.
/// Individual items are not inspected.
pub fn non_empty<T>(items: Vec<T>) -> Option<Vec<T>> {
    if items.is_empty() { None } else { Some(items) }
}

/// Returns whether the choice contains no nonempty text, tool call, or image.
/// Reasoning and provider-only items are not an answer, even when retained
/// in history.
pub fn turn_delivered_no_answer(choice: &[AssistantContent]) -> bool {
    !choice.iter().any(|content| match content {
        AssistantContent::Text(text) => !text.text.is_empty(),
        AssistantContent::ToolCall(_) | AssistantContent::Image(_) => true,
        AssistantContent::Reasoning(_) | AssistantContent::Opaque(_) => false,
    })
}

/// Why a run fails on a turn the provider failed (`stop` is
/// [`StopReason::Error`]) that delivered no answer: a failed turn with
/// nothing to show cannot end a run as a success. `None` for any other turn.
pub fn failed_turn_message(
    choice: &[AssistantContent],
    stop: Option<&StopReason>,
) -> Option<String> {
    match stop {
        Some(StopReason::Error(reason)) if turn_delivered_no_answer(choice) => Some(format!(
            "the provider failed the turn without an answer: {reason}"
        )),
        _ => None,
    }
}

/// User text, tool results, or media. Supported source kinds and media types
/// depend on the target provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    /// Plain text user content.
    Text(Text),
    /// Result of a tool call returned as user-visible context to the model.
    ToolResult(ToolResult),
    /// Image content.
    Image(Image),
    /// Audio content.
    Audio(Audio),
    /// Video content.
    Video(Video),
    /// Document content.
    Document(Document),
}

/// One block of an assistant turn: one provider output item.
/// Deserialization requires the lowercase `type` tag.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    /// Answer text.
    Text(Text),
    /// A tool call requested by the assistant.
    ToolCall(ToolCall),
    /// Reasoning the model showed.
    Reasoning(Reasoning),
    /// An image the assistant produced.
    Image(Image),
    /// A provider item with no canonical meaning.
    Opaque(Opaque),
}

impl AssistantContent {
    /// This block without its provider item: what a different provider can
    /// use, and what its fingerprint covers.
    pub fn canonical(&self) -> Self {
        let mut block = self.clone();
        if let Some(native) = block.native_slot() {
            *native = None;
        }
        block
    }

    /// Whether the block has nothing to send, so replay drops it: blank text
    /// or reasoning without a provider item that is still current (with
    /// one, its identity pairs it with what follows, and pi replays it
    /// whatever its text), and an opaque item that does not replay.
    pub fn is_blank(&self) -> bool {
        match self {
            Self::Text(Text { text, .. }) | Self::Reasoning(Reasoning { text, .. }) => {
                text.trim().is_empty() && self.native_item().is_none()
            }
            Self::Opaque(opaque) => !opaque.replay,
            Self::ToolCall(_) | Self::Image(_) => false,
        }
    }

    /// The fingerprint of the block's canonical fields, through a fixed,
    /// versioned projection rather than the block's serde layout, so a field
    /// added to a canonical type never stales stored items. A rig-issued
    /// call id counts as one placeholder: rig issues a fresh id each time it
    /// decodes a call the provider sent without one.
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::of(&self.projection())
    }

    /// Projection v1 of the canonical fields.
    fn projection(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Self::Text(text) => json!(["v1", "text", text.text]),
            Self::Reasoning(reasoning) => {
                json!(["v1", "reasoning", reasoning.text, reasoning.redacted])
            }
            Self::ToolCall(call) => {
                let id = match &call.id {
                    CallId::Provider(id) => id.as_str().to_owned(),
                    CallId::Local(_) => "~local".to_owned(),
                };
                json!([
                    "v1",
                    "call",
                    id,
                    call.function.name.as_str(),
                    call.function.arguments,
                    call.function.invalid_arguments,
                ])
            }
            Self::Image(image) => {
                json!(["v1", "image", image.media_type, image.detail, image.data,])
            }
            Self::Opaque(_) => json!(["v1", "opaque"]),
        }
    }

    fn native_slot(&mut self) -> Option<&mut Option<Native>> {
        match self {
            Self::Text(text) => Some(&mut text.native),
            Self::ToolCall(call) => Some(&mut call.native),
            Self::Reasoning(reasoning) => Some(&mut reasoning.native),
            Self::Image(image) => Some(&mut image.native),
            Self::Opaque(_) => None,
        }
    }

    /// The provider item this block was decoded from, held for its current
    /// canonical form. An [`Opaque`] block has no separate item.
    pub fn with_native(mut self, item: serde_json::Value) -> Self {
        let fingerprint = self.fingerprint();
        if let Some(native) = self.native_slot() {
            *native = Some(Native { item, fingerprint });
        }
        self
    }

    /// The provider item of an edited block: no longer current, but its
    /// identity keys still name the item it was.
    pub(crate) fn stale_item(&self) -> Option<&serde_json::Value> {
        let native = match self {
            Self::Text(text) => text.native.as_ref(),
            Self::ToolCall(call) => call.native.as_ref(),
            Self::Reasoning(reasoning) => reasoning.native.as_ref(),
            Self::Image(image) => image.native.as_ref(),
            Self::Opaque(_) => None,
        }?;
        (native.fingerprint != self.fingerprint()).then_some(&native.item)
    }

    /// The provider item, while the block is still what it was decoded
    /// from. An edited block has none: encoders rebuild it.
    pub fn native_item(&self) -> Option<&serde_json::Value> {
        let native = match self {
            Self::Text(text) => text.native.as_ref(),
            Self::ToolCall(call) => call.native.as_ref(),
            Self::Reasoning(reasoning) => reasoning.native.as_ref(),
            Self::Image(image) => image.native.as_ref(),
            Self::Opaque(_) => None,
        }?;
        (native.fingerprint == self.fingerprint()).then_some(&native.item)
    }
}

/// Reasoning the model showed: its text, or a redacted block with none.
/// Signatures, encrypted payloads and ids are provider data and live in
/// `native`.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Reasoning {
    /// The reasoning text, summaries included.
    pub text: String,
    /// Whether the provider withheld the text.
    #[serde(default, skip_serializing_if = "crate::json_utils::is_false")]
    pub redacted: bool,
    /// The provider item this block was decoded from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<Native>,
}

impl Reasoning {
    /// Reasoning text with no provider item.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }
}

/// The result of a tool call, sent back to the model.
///
/// Build it from the call it answers with [`ToolCall::result`], so its id
/// and name match the call.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolResult {
    /// The id of the answered call.
    pub call: CallId,
    /// Executed tool name, which may differ from the model-requested name after
    /// hook repair. Required for provider replay independently of call identity.
    pub name: ToolName,
    /// One or more content items produced by the tool.
    pub content: Vec<ToolResultContent>,
    /// Whether the tool failed, was refused, or never ran: `content` then
    /// says why.
    #[serde(default, skip_serializing_if = "crate::json_utils::is_false")]
    pub is_error: bool,
}

/// Describes one typed item in a tool result.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolResultContent {
    /// Literal text. Providers must not reinterpret it as structured JSON.
    Text(Text),
    /// An image supplied explicitly by the tool.
    Image(Image),
    /// Structured JSON supplied explicitly by the tool runtime.
    Json {
        /// The structured value.
        value: serde_json::Value,
    },
}

impl ToolResultContent {
    /// Borrow literal text content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            Self::Image(_) | Self::Json { .. } => None,
        }
    }

    /// Borrow structured JSON content.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json { value } => Some(value),
            Self::Text(_) | Self::Image(_) => None,
        }
    }

    /// Deserialize JSON content into a typed value.
    ///
    /// Structured JSON is decoded directly. Literal text is parsed only because
    /// the caller explicitly requested JSON decoding, which supports transcripts
    /// recorded before structured tool output was preserved canonically. This
    /// helper never changes the content sent to a model or provider.
    pub fn deserialize_json<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        match self {
            Self::Json { value } => T::deserialize(value),
            Self::Text(text) => serde_json::from_str(&text.text),
            Self::Image(_) => Err(<serde_json::Error as serde::de::Error>::custom(
                "cannot decode image tool-result content as JSON",
            )),
        }
    }
}

/// A tool call: its id, and the function and arguments requested.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    /// The call's one identity: the provider's id, or one rig issued when
    /// the provider sent none.
    pub id: CallId,
    /// Function name and JSON arguments requested by the model.
    pub function: ToolFunction,
    /// The provider item this call was decoded from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<Native>,
}

impl ToolCall {
    /// A call with `id` and no provider item.
    pub fn new(id: CallId, function: ToolFunction) -> Self {
        Self {
            id,
            function,
            native: None,
        }
    }

    /// A call the provider identified by `wire_id`; rig issues an id when it
    /// is empty.
    pub fn from_wire(wire_id: impl Into<String>, function: ToolFunction) -> Self {
        Self::new(CallId::from_wire(wire_id), function)
    }

    /// The result answering this call: its id and name, and `content`.
    pub fn result(&self, content: Vec<ToolResultContent>) -> ToolResult {
        ToolResult {
            call: self.id.clone(),
            name: self.function.name.clone(),
            content,
            is_error: false,
        }
    }

    /// The result reporting that this call failed, with `content` saying
    /// why.
    pub fn error_result(&self, content: Vec<ToolResultContent>) -> ToolResult {
        ToolResult {
            is_error: true,
            ..self.result(content)
        }
    }
}

/// A tool function to call: its name and its arguments, always a JSON
/// object.
///
/// Arguments the model sent that are not an object are kept as text in
/// `invalid_arguments`, with `arguments` holding what could be read of them
/// (`{}` when nothing could), so a malformed call never fails a reply and
/// every wire receives an object.
///
/// ```
/// use rig_core::message::{ToolFunction, ToolName};
///
/// let name = ToolName::new("search")?;
/// let call = ToolFunction::parse(name.clone(), r#"{"q": "ab"#);
/// assert_eq!(call.arguments["q"], "ab");
/// assert_eq!(call.invalid_arguments.as_deref(), Some(r#"{"q": "ab"#));
///
/// let call = ToolFunction::new(name, serde_json::json!("{\"q\":1}"));
/// assert_eq!(call.arguments["q"], 1);
/// assert!(call.invalid_arguments.is_none());
/// # Ok::<(), rig_core::message::EmptyToolName>(())
/// ```
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ToolFunction {
    /// Tool/function name to invoke.
    pub name: ToolName,
    /// The arguments.
    pub arguments: serde_json::Map<String, serde_json::Value>,
    /// The arguments as the model sent them, when they were not a JSON
    /// object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid_arguments: Option<String>,
}

impl ToolFunction {
    /// A call to `name` with `arguments`. An object is kept; a string
    /// holding an object is unwrapped once; `null` is `{}`; anything else is
    /// `{}` with its JSON text in `invalid_arguments`.
    pub fn new(name: ToolName, arguments: serde_json::Value) -> Self {
        use serde_json::Value;
        let (arguments, invalid_arguments) = match arguments {
            Value::Object(arguments) => (arguments, None),
            Value::Null => (serde_json::Map::new(), None),
            Value::String(text) => {
                return Self::parse(name, &text);
            }
            other => (serde_json::Map::new(), Some(other.to_string())),
        };
        Self {
            name,
            arguments,
            invalid_arguments,
        }
    }

    /// A call to `name` with the argument JSON `text`. Blank text is `{}`.
    /// Text that is not an object keeps what a cut-off object still states,
    /// or `{}`, and is kept in `invalid_arguments`. A string holding an
    /// object is unwrapped once.
    pub fn parse(name: ToolName, text: &str) -> Self {
        use serde_json::Value;
        let parsed = crate::json_utils::parse_tool_arguments(text);
        let (arguments, invalid) = match parsed {
            Ok(Value::Object(arguments)) => (arguments, false),
            Ok(Value::Null) => (serde_json::Map::new(), false),
            Ok(Value::String(inner)) => match serde_json::from_str(&inner) {
                Ok(Value::Object(arguments)) => (arguments, false),
                _ => (serde_json::Map::new(), true),
            },
            Ok(_) => (serde_json::Map::new(), true),
            Err(_) => (
                crate::json_utils::parse_partial_object(text).unwrap_or_default(),
                true,
            ),
        };
        Self {
            name,
            arguments,
            invalid_arguments: invalid.then(|| text.to_owned()),
        }
    }

    /// The arguments as a JSON value.
    pub fn arguments_value(&self) -> serde_json::Value {
        serde_json::Value::Object(self.arguments.clone())
    }
}

impl<'de> Deserialize<'de> for ToolFunction {
    /// Arguments stored in any JSON shape are read through
    /// [`ToolFunction::new`], so a history saved before arguments were
    /// always objects still loads.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Repr {
            name: ToolName,
            #[serde(default)]
            arguments: serde_json::Value,
            #[serde(default)]
            invalid_arguments: Option<String>,
        }
        let Repr {
            name,
            arguments,
            invalid_arguments,
        } = Repr::deserialize(deserializer)?;
        let mut function = Self::new(name, arguments);
        if invalid_arguments.is_some() {
            function.invalid_arguments = invalid_arguments;
        }
        Ok(function)
    }
}

/// Text. On an assistant turn, `native` holds the provider item the block
/// was decoded from; user and tool-result text leave it `None`.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    /// Text content.
    pub text: String,
    /// The provider item this block was decoded from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<Native>,
}

impl Text {
    /// Text with no provider item.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            native: None,
        }
    }

    /// Returns the inner text string.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { text, .. } = self;
        write!(f, "{text}")
    }
}

/// Image content containing image data and metadata about it. On an
/// assistant turn, `native` holds the provider item it was decoded from.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Image {
    /// Image source data.
    pub data: DocumentSourceKind,
    /// Image media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<ImageMediaType>,
    /// Provider-specific image detail preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
    /// The provider item this image was decoded from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<Native>,
}

/// The kind of image source (to be used).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
pub enum DocumentSourceKind {
    /// A file URL/URI.
    Url(String),
    /// A base-64 encoded string.
    Base64(String),
    /// A provider-side uploaded file identifier.
    FileId(String),
    /// Raw bytes
    Raw(Vec<u8>),
    /// A string (or a string literal).
    String(String),
    #[default]
    /// An unknown file source (there's nothing there).
    Unknown,
}

impl DocumentSourceKind {
    /// Create a URL-backed source.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    /// Create a base64-backed source.
    pub fn base64(base64_string: impl Into<String>) -> Self {
        Self::Base64(base64_string.into())
    }

    /// Create a provider file ID-backed source.
    pub fn file_id(file_id: impl Into<String>) -> Self {
        Self::FileId(file_id.into())
    }

    /// Create a string-backed source.
    pub fn string(input: impl Into<String>) -> Self {
        Self::String(input.into())
    }

    /// Return the contained URL, base64 string, or file ID, if this source stores one.
    pub fn try_into_inner(self) -> Option<String> {
        match self {
            Self::Url(s) | Self::Base64(s) | Self::FileId(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for DocumentSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Url(string) => write!(f, "{string}"),
            Self::Base64(string) => write!(f, "{string}"),
            Self::FileId(string) => write!(f, "{string}"),
            Self::String(string) => write!(f, "{string}"),
            Self::Raw(_) => write!(f, "<binary data>"),
            Self::Unknown => write!(f, "<unknown>"),
        }
    }
}

/// Audio content containing audio data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Audio {
    /// Audio source data.
    pub data: DocumentSourceKind,
    /// Audio media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<AudioMediaType>,
}

/// Video content containing video data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Video {
    /// Video source data.
    pub data: DocumentSourceKind,
    /// Video media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<VideoMediaType>,
    /// Provider-specific video fields, a JSON object (Gemini's
    /// `video_metadata`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Document content containing document data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Document {
    /// Document source data.
    pub data: DocumentSourceKind,
    /// Document media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
    /// Provider-specific document fields, a JSON object (Anthropic's
    /// `title`, `context` and `citations`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Content representation as base64, text, or a URL.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContentFormat {
    #[default]
    Base64,
    String,
    Url,
}

/// Helper enum that tracks the media type of the content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum MediaType {
    Image(ImageMediaType),
    Audio(AudioMediaType),
    Document(DocumentMediaType),
    Video(VideoMediaType),
}

/// Describes the image media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    JPEG,
    PNG,
    GIF,
    WEBP,
    HEIC,
    HEIF,
    SVG,
}

/// Describes the document media type of the content. Not every provider supports every media type.
/// Includes also programming languages as document types for providers who support code running.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    PDF,
    TXT,
    RTF,
    HTML,
    CSS,
    MARKDOWN,
    CSV,
    XML,
    Javascript,
    Python,
}

impl DocumentMediaType {
    pub fn is_code(&self) -> bool {
        matches!(self, Self::Javascript | Self::Python)
    }
}

/// Describes the audio media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    WAV,
    MP3,
    AIFF,
    AAC,
    OGG,
    FLAC,
    M4A,
    PCM16,
    PCM24,
}

/// Describes the video media type of the content. Not every provider supports every media type.
/// Convertible to and from MIME type strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VideoMediaType {
    AVI,
    MP4,
    MPEG,
    MOV,
    WEBM,
}

/// Describes the detail of the image content, which can be low, high, or auto (open-ai specific).
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}

impl Message {
    /// Clones the first text block of a user message, or returns `None`.
    pub fn rag_text(&self) -> Option<String> {
        match self {
            Message::User { content } => {
                for item in content.iter() {
                    if let UserContent::Text(Text { text, .. }) = item {
                        return Some(text.clone());
                    }
                }
                None
            }
            Message::System { .. } => None,
            _ => None,
        }
    }

    /// Creates a system instruction message.
    pub fn system(text: impl Into<String>) -> Self {
        Message::System {
            content: text.into(),
        }
    }

    /// Creates a user message containing one text block.
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::text(text)],
        }
    }

    /// Creates a hand-built assistant message containing one text block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::text(text)]))
    }

    /// Creates a user message containing a text tool result answering the
    /// call `call` to the tool `name`. To answer a call you hold, prefer
    /// [`ToolCall::result`] with [`Self::tool_results`].
    pub fn tool_result(call: CallId, name: ToolName, content: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::tool_result(
                call,
                name,
                vec![ToolResultContent::text(content)],
            )],
        }
    }

    /// Creates a user message carrying `results`, in order.
    pub fn tool_results(results: Vec<ToolResult>) -> Self {
        Message::User {
            content: results.into_iter().map(UserContent::ToolResult).collect(),
        }
    }
}

/// Generates media constructors without fetching or decoding source data.
macro_rules! media_ctors {
    () => {};
    (
        $(#[$meta:meta])* $name:ident => Image($kind:ident: $data:ty);
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name(
            data: impl Into<$data>,
            media_type: Option<ImageMediaType>,
            detail: Option<ImageDetail>,
        ) -> Self {
            Self::Image(Image {
                data: DocumentSourceKind::$kind(data.into()),
                media_type,
                detail,
                native: None,
            })
        }
        media_ctors! { $($rest)* }
    };
    (
        $(#[$meta:meta])* $name:ident => $variant:ident(params $mt:ty, $kind:ident: $data:ty);
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name(data: impl Into<$data>, media_type: Option<$mt>) -> Self {
            Self::$variant($variant {
                data: DocumentSourceKind::$kind(data.into()),
                media_type,
                additional_params: None,
            })
        }
        media_ctors! { $($rest)* }
    };
    (
        $(#[$meta:meta])* $name:ident => $variant:ident($mt:ty, $kind:ident: $data:ty);
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name(data: impl Into<$data>, media_type: Option<$mt>) -> Self {
            Self::$variant($variant {
                data: DocumentSourceKind::$kind(data.into()),
                media_type,
            })
        }
        media_ctors! { $($rest)* }
    };
}

impl UserContent {
    /// Creates user text content.
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into().into())
    }

    media_ctors! {
        /// Creates user image content from base64-encoded data.
        image_base64 => Image(Base64: String);
        /// Creates user image content from unencoded bytes.
        image_raw => Image(Raw: Vec<u8>);
        /// Creates user image content referencing a URL.
        image_url => Image(Url: String);
        /// Creates user audio content from base64-encoded data.
        audio_base64 => Audio(AudioMediaType, Base64: String);
        /// Creates user audio content from unencoded bytes.
        audio_raw => Audio(AudioMediaType, Raw: Vec<u8>);
        /// Creates user audio content referencing a URL.
        audio_url => Audio(AudioMediaType, Url: String);
        /// Creates user video content from base64-encoded data.
        video_base64 => Video(params VideoMediaType, Base64: String);
        /// Creates user video content from unencoded bytes.
        video_raw => Video(params VideoMediaType, Raw: Vec<u8>);
        /// Creates user video content referencing a URL.
        video_url => Video(params VideoMediaType, Url: String);
        /// Creates user document content from base64-encoded data.
        document_base64 => Document(params DocumentMediaType, Base64: String);
        /// Creates user document content from unencoded bytes.
        document_raw => Document(params DocumentMediaType, Raw: Vec<u8>);
        /// Creates user document content referencing a URL.
        document_url => Document(params DocumentMediaType, Url: String);
        /// Creates user document content from literal text, such as a plain
        /// text or Markdown file. Binary formats belong in
        /// [`Self::document_base64`] or [`Self::document_raw`].
        document_text => Document(params DocumentMediaType, String: String);
    }

    /// Creates a tool result answering the call `call` to the tool `name`.
    pub fn tool_result(call: CallId, name: ToolName, content: Vec<ToolResultContent>) -> Self {
        UserContent::ToolResult(ToolResult {
            call,
            name,
            content,
            is_error: false,
        })
    }
}

impl AssistantContent {
    /// Creates assistant text content.
    pub fn text(text: impl Into<String>) -> Self {
        AssistantContent::Text(text.into().into())
    }

    media_ctors! {
        /// Creates assistant image content from base64-encoded data.
        image_base64 => Image(Base64: String);
    }

    /// Creates a tool call from a provider-issued ID, name, and arguments.
    /// Rig issues an id when `id` is empty.
    pub fn tool_call(id: impl Into<String>, name: ToolName, arguments: serde_json::Value) -> Self {
        AssistantContent::ToolCall(ToolCall::from_wire(id, ToolFunction::new(name, arguments)))
    }

    /// Creates reasoning text with no provider item.
    pub fn reasoning(reasoning: impl Into<String>) -> Self {
        AssistantContent::Reasoning(Reasoning::new(reasoning))
    }
}

impl ToolResultContent {
    /// Creates literal text tool-result content.
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text(text.into().into())
    }

    /// Creates structured JSON tool-result content.
    pub fn json(value: serde_json::Value) -> Self {
        ToolResultContent::Json { value }
    }

    media_ctors! {
        /// Creates tool-result image content from base64-encoded data.
        image_base64 => Image(Base64: String);
        /// Creates tool-result image content from raw, unencoded bytes.
        image_raw => Image(Raw: Vec<u8>);
        /// Creates tool-result image content referencing a URL.
        image_url => Image(Url: String);
    }
}

/// Trait for converting between MIME types and media types.
pub trait MimeType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized;
    fn to_mime_type(&self) -> &'static str;
}

impl MimeType for MediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        ImageMediaType::from_mime_type(mime_type)
            .map(MediaType::Image)
            .or_else(|| DocumentMediaType::from_mime_type(mime_type).map(MediaType::Document))
            .or_else(|| AudioMediaType::from_mime_type(mime_type).map(MediaType::Audio))
            .or_else(|| VideoMediaType::from_mime_type(mime_type).map(MediaType::Video))
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            MediaType::Image(media_type) => media_type.to_mime_type(),
            MediaType::Audio(media_type) => media_type.to_mime_type(),
            MediaType::Document(media_type) => media_type.to_mime_type(),
            MediaType::Video(media_type) => media_type.to_mime_type(),
        }
    }
}

// Emits both directions of a [`MimeType`] impl from a single pair list, so a
// variant's parse and emit spellings cannot drift apart. Extra `| "alias"`
// spellings parse to the same variant; only the first (canonical) string is
// emitted by `to_mime_type`.
macro_rules! impl_mime_type {
    ($ty:ident { $($variant:ident => $canonical:literal $(| $alias:literal)*),+ $(,)? }) => {
        impl MimeType for $ty {
            fn from_mime_type(mime_type: &str) -> Option<Self> {
                match mime_type {
                    $($canonical $(| $alias)* => Some($ty::$variant),)+
                    _ => None,
                }
            }

            fn to_mime_type(&self) -> &'static str {
                match self {
                    $($ty::$variant => $canonical,)+
                }
            }
        }
    };
}

impl_mime_type!(ImageMediaType {
    JPEG => "image/jpeg",
    PNG => "image/png",
    GIF => "image/gif",
    WEBP => "image/webp",
    HEIC => "image/heic",
    HEIF => "image/heif",
    SVG => "image/svg+xml",
});

impl_mime_type!(DocumentMediaType {
    PDF => "application/pdf",
    TXT => "text/plain",
    RTF => "text/rtf",
    HTML => "text/html",
    CSS => "text/css",
    MARKDOWN => "text/markdown" | "text/md",
    CSV => "text/csv",
    XML => "text/xml",
    Javascript => "application/x-javascript" | "text/x-javascript",
    Python => "application/x-python" | "text/x-python",
});

impl_mime_type!(AudioMediaType {
    WAV => "audio/wav",
    MP3 => "audio/mp3",
    AIFF => "audio/aiff",
    AAC => "audio/aac",
    OGG => "audio/ogg",
    FLAC => "audio/flac",
    M4A => "audio/m4a",
    PCM16 => "audio/pcm16",
    PCM24 => "audio/pcm24",
});

impl_mime_type!(VideoMediaType {
    AVI => "video/avi",
    MP4 => "video/mp4",
    MPEG => "video/mpeg",
    MOV => "video/mov",
    WEBM => "video/webm",
});

impl std::str::FromStr for ImageDetail {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(ImageDetail::Low),
            "high" => Ok(ImageDetail::High),
            "auto" => Ok(ImageDetail::Auto),
            _ => Err(()),
        }
    }
}

/// `From` impls for [`Text`] from string-like types.
macro_rules! text_from {
    ($($src:ty),+ $(,)?) => {$(
        impl From<$src> for Text {
            fn from(text: $src) -> Self {
                Text::new(text)
            }
        }
    )+};
}

text_from!(String, &String, &str);

/// `From<String>` impls that forward into a content type's `text` constructor.
macro_rules! text_content_from_string {
    ($($ty:ident),+ $(,)?) => {$(
        impl From<String> for $ty {
            fn from(text: String) -> Self {
                $ty::text(text)
            }
        }
    )+};
}

text_content_from_string!(ToolResultContent, AssistantContent, UserContent);

/// One-line `From<T> for Message` forwards: convert the value, wrap it in the
/// named content variant, and build a single-content message.
macro_rules! single_content_message_from {
    (User { $($src:ty => $variant:ident),+ $(,)? }) => {$(
        impl From<$src> for Message {
            fn from(value: $src) -> Self {
                Message::User {
                    content: vec![UserContent::$variant(value.into())],
                }
            }
        }
    )+};
    (Assistant { $($src:ty => $variant:ident),+ $(,)? }) => {$(
        impl From<$src> for Message {
            fn from(value: $src) -> Self {
                Message::Assistant(AssistantMessage::new(vec![AssistantContent::$variant(
                    value.into(),
                )]))
            }
        }
    )+};
}

single_content_message_from!(User {
    String => Text,
    &str => Text,
    &String => Text,
    Text => Text,
    Image => Image,
    Audio => Audio,
    Document => Document,
    ToolResult => ToolResult,
});

single_content_message_from!(Assistant {
    ToolCall => ToolCall,
});

impl FromStr for Text {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl From<&Message> for Message {
    fn from(msg: &Message) -> Self {
        msg.clone()
    }
}

impl From<AssistantContent> for Message {
    fn from(content: AssistantContent) -> Self {
        Message::Assistant(AssistantMessage::new(vec![content]))
    }
}

impl From<AssistantMessage> for Message {
    fn from(message: AssistantMessage) -> Self {
        Message::Assistant(message)
    }
}

impl From<UserContent> for Message {
    fn from(content: UserContent) -> Self {
        Message::User {
            content: vec![content],
        }
    }
}

impl From<Vec<AssistantContent>> for Message {
    fn from(content: Vec<AssistantContent>) -> Self {
        Message::Assistant(AssistantMessage::new(content))
    }
}

impl From<Vec<UserContent>> for Message {
    fn from(content: Vec<UserContent>) -> Self {
        Message::User { content }
    }
}

#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific {
        function_names: Vec<ToolName>,
    },
}

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}

impl From<MessageError> for ProviderError {
    fn from(error: MessageError) -> Self {
        ProviderError::request(error)
    }
}

#[cfg(test)]
mod tests;
