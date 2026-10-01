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

    /// Assistant message containing one or more content types defined by `AssistantContent`.
    Assistant {
        /// Provider-assigned assistant message ID, when available.
        id: Option<String>,
        content: Vec<AssistantContent>,
    },
}

mod identity;

pub use identity::{
    CallId, EmptyCallId, EmptyToolName, Issuer, LocalCallId, ProviderCallId, Sealed, ToolName,
};

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

/// Concatenates reasoning, text, and trailing content in that order without
/// dropping items. Each group's input order is preserved.
pub fn ordered_assistant_content(
    reasoning_items: impl IntoIterator<Item = Sealed<Reasoning>>,
    text_items: impl IntoIterator<Item = AssistantContent>,
    trailing_items: impl IntoIterator<Item = AssistantContent>,
) -> Vec<AssistantContent> {
    let mut content_items = reasoning_items
        .into_iter()
        .map(AssistantContent::Reasoning)
        .collect::<Vec<_>>();
    content_items.extend(text_items);
    content_items.extend(trailing_items);
    content_items
}

/// Returns whether the choice contains no nonempty text, tool call, or image.
/// Reasoning alone is not an answer, even when retained in history.
pub fn turn_delivered_no_answer(choice: &[AssistantContent]) -> bool {
    !choice.iter().any(|content| match content {
        // Real text is an answer; an empty block delivers nothing.
        AssistantContent::Text(text) => !text.text.is_empty(),
        AssistantContent::ToolCall(_) => true,
        AssistantContent::Image(_) => true,
        // The one exclusion: scratch work, not an answer.
        AssistantContent::Reasoning(_) => false,
    })
}

/// Groups streamed choices as reasoning, text, tool calls, then images,
/// preserving order within each group. Choices without reasoning or tool calls
/// retain their original order.
pub fn canonical_streamed_choice(choice: Vec<AssistantContent>) -> Vec<AssistantContent> {
    let regroup = choice.iter().any(|part| {
        matches!(
            part,
            AssistantContent::Reasoning(_) | AssistantContent::ToolCall(_)
        )
    });
    if !regroup {
        return choice;
    }
    let mut reasoning = Vec::new();
    let mut text = Vec::new();
    let mut calls = Vec::new();
    let mut images = Vec::new();
    for part in choice {
        match part {
            AssistantContent::Reasoning(block) => reasoning.push(block),
            AssistantContent::Text(_) => text.push(part),
            AssistantContent::ToolCall(_) => calls.push(part),
            AssistantContent::Image(_) => images.push(part),
        }
    }
    ordered_assistant_content(reasoning, text, calls.into_iter().chain(images))
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

/// Assistant text, tool calls, reasoning, or images.
/// Deserialization requires the lowercase `type` tag.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    /// Plain assistant text.
    Text(Text),
    /// Tool call requested by the assistant.
    ToolCall(ToolCall),
    /// Structured reasoning emitted by the assistant, readable only by the
    /// service that issued it.
    Reasoning(Sealed<Reasoning>),
    /// Image content emitted by the assistant.
    Image(Image),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
/// A typed reasoning block used by providers that emit structured thinking data.
pub enum ReasoningContent {
    /// Plain reasoning text with an optional provider signature.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Provider-encrypted reasoning payload.
    Encrypted(String),
    /// Redacted reasoning payload preserved as opaque data.
    Redacted { data: String },
    /// Provider-generated reasoning summary text.
    Summary(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
/// Assistant reasoning payload with an optional provider-supplied identifier.
/// A message carries it [`Sealed`] to the service that issued it:
/// signatures, encrypted and redacted payloads and reasoning ids only mean
/// something there.
pub struct Reasoning {
    /// Provider reasoning identifier, when supplied by the upstream API.
    pub id: Option<String>,
    /// Ordered reasoning content blocks.
    pub content: Vec<ReasoningContent>,
}

impl Reasoning {
    /// Create a new reasoning item from a single item
    pub fn new(input: &str) -> Self {
        Self::new_with_signature(input, None)
    }

    /// Create a new reasoning item from a single text item and optional signature.
    pub fn new_with_signature(input: &str, signature: Option<String>) -> Self {
        Self {
            id: None,
            content: vec![ReasoningContent::Text {
                text: input.to_string(),
                signature,
            }],
        }
    }

    /// This reasoning, readable only by `issuer`.
    pub fn sealed(self, issuer: impl Into<Issuer>) -> Sealed<Self> {
        Sealed::new(issuer, self)
    }

    /// Set a provider reasoning ID.
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Create reasoning content from multiple text blocks.
    pub fn multi(input: Vec<String>) -> Self {
        Self {
            id: None,
            content: input
                .into_iter()
                .map(|text| ReasoningContent::Text {
                    text,
                    signature: None,
                })
                .collect(),
        }
    }

    /// Create a redacted reasoning block.
    pub fn redacted(data: impl Into<String>) -> Self {
        Self {
            id: None,
            content: vec![ReasoningContent::Redacted { data: data.into() }],
        }
    }

    /// Create an encrypted reasoning block.
    pub fn encrypted(data: impl Into<String>) -> Self {
        Self {
            id: None,
            content: vec![ReasoningContent::Encrypted(data.into())],
        }
    }

    /// Create one reasoning block containing summary items.
    pub fn summaries(input: Vec<String>) -> Self {
        Self {
            id: None,
            content: input.into_iter().map(ReasoningContent::Summary).collect(),
        }
    }

    /// Render reasoning as displayable text by joining text-like blocks with newlines.
    pub fn display_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|content| match content {
                ReasoningContent::Text { text, .. } => Some(text.as_str()),
                ReasoningContent::Summary(summary) => Some(summary.as_str()),
                ReasoningContent::Redacted { data } => Some(data.as_str()),
                ReasoningContent::Encrypted(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the first text reasoning block, if present.
    pub fn first_text(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Return the first signature from text reasoning, if present.
    pub fn first_signature(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Text {
                signature: Some(signature),
                ..
            } => Some(signature.as_str()),
            _ => None,
        })
    }

    /// Return the first encrypted reasoning payload, if present.
    pub fn encrypted_content(&self) -> Option<&str> {
        self.content.iter().find_map(|content| match content {
            ReasoningContent::Encrypted(data) => Some(data.as_str()),
            _ => None,
        })
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

/// Describes a tool call with an id and function to call, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    /// The call's one identity: the provider's id, or one rig issued when
    /// the provider sent none.
    pub id: CallId,
    /// Function name and JSON arguments requested by the model.
    pub function: ToolFunction,
    /// Opaque provider signature preserved for replay. Rig does not verify it.
    #[serde(default)]
    pub signature: Option<String>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    #[serde(default)]
    pub additional_params: Option<serde_json::Value>,
}

impl ToolCall {
    /// A call with `id`.
    pub fn new(id: CallId, function: ToolFunction) -> Self {
        Self {
            id,
            function,
            signature: None,
            additional_params: None,
        }
    }

    /// A call the provider identified by `wire_id`; rig issues an id when it
    /// is empty.
    pub fn from_wire(wire_id: impl Into<String>, function: ToolFunction) -> Self {
        Self::new(CallId::from_wire(wire_id), function)
    }

    /// The dual-identifier provider boundary (OpenAI Responses): `item_id`
    /// is the output-item handle (`fc_…`), `call_id` the correlator
    /// (`call_…`). Rig issues an id when `call_id` is empty.
    pub fn from_dual_wire(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        function: ToolFunction,
    ) -> Self {
        Self::new(CallId::from_dual_wire(item_id, call_id), function)
    }

    /// The result answering this call: its id and name, and `content`.
    pub fn result(&self, content: Vec<ToolResultContent>) -> ToolResult {
        ToolResult {
            call: self.id.clone(),
            name: self.function.name.clone(),
            content,
        }
    }

    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

/// Describes a tool function to call with a name and arguments, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolFunction {
    /// Tool/function name to invoke.
    pub name: ToolName,
    /// JSON arguments for the tool/function.
    pub arguments: serde_json::Value,
}

impl ToolFunction {
    /// Create a tool function call payload.
    pub fn new(name: ToolName, arguments: serde_json::Value) -> Self {
        Self { name, arguments }
    }
}

/// Nonempty JSON object of provider-specific content metadata, serialized as
/// an object under a named `additional_params` field rather than flattened.
/// Constructors return `None` for empty maps. Bare deserialization rejects
/// empty or non-object values; [`optional_additional_params`] maps null and
/// empty objects to absence. Providers must replay only their own metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AdditionalParams(serde_json::Map<String, serde_json::Value>);

impl AdditionalParams {
    /// The canonical constructor: `None` when the map is empty.
    pub fn new(map: serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        if map.is_empty() {
            None
        } else {
            Some(Self(map))
        }
    }

    /// Build from `(key, value)` entries; `None` when the iterator yields
    /// none. `Option<(K, Value)>` is such an iterator, so a conditional
    /// single-key params reads as
    /// `AdditionalParams::from_entries(guard.then(|| (key, value)))`.
    pub fn from_entries<K, I>(entries: I) -> Option<Self>
    where
        K: Into<String>,
        I: IntoIterator<Item = (K, serde_json::Value)>,
    {
        Self::new(
            entries
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// The value stored under `key`, when present.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// The underlying (non-empty) object.
    pub fn as_map(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.0
    }

    /// The params as a bare JSON object value.
    pub fn into_value(self) -> serde_json::Value {
        serde_json::Value::Object(self.0)
    }

    /// Deep-merge `incoming` into `self`: arrays concatenate (streamed
    /// citation deltas), objects merge recursively, scalars take the
    /// incoming value.
    pub fn merge(&mut self, incoming: Self) {
        fn merge_maps(
            existing: &mut serde_json::Map<String, serde_json::Value>,
            incoming: serde_json::Map<String, serde_json::Value>,
        ) {
            for (key, incoming_value) in incoming {
                match existing.get_mut(&key) {
                    Some(existing_value) => merge_value(existing_value, incoming_value),
                    None => {
                        existing.insert(key, incoming_value);
                    }
                }
            }
        }
        fn merge_value(existing: &mut serde_json::Value, incoming: serde_json::Value) {
            match (existing, incoming) {
                (
                    serde_json::Value::Object(existing_map),
                    serde_json::Value::Object(incoming_map),
                ) => merge_maps(existing_map, incoming_map),
                (
                    serde_json::Value::Array(existing_array),
                    serde_json::Value::Array(mut incoming_array),
                ) => existing_array.append(&mut incoming_array),
                (existing, incoming) => *existing = incoming,
            }
        }
        merge_maps(&mut self.0, incoming.0);
    }

    /// Returns the object under the provider's own key, or `None` for absent
    /// or non-object values. Use [`Self::get`] to diagnose malformed values.
    pub fn wire_extras(
        &self,
        wire_key: &str,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.0.get(wire_key).and_then(serde_json::Value::as_object)
    }

    /// Owned counterpart of [`Self::wire_extras`] for serialization paths
    /// that already own the params (the common replay case): extracts the
    /// wire's object without cloning. Same gate semantics.
    pub fn into_wire_extras(
        mut self,
        wire_key: &str,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        match self.0.remove(wire_key) {
            Some(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        }
    }

    /// Returns absence for null or empty objects, metadata for nonempty objects,
    /// or the original value as an error for other shapes.
    pub fn try_from_value(value: serde_json::Value) -> Result<Option<Self>, serde_json::Value> {
        match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Object(map) => Ok(Self::new(map)),
            other => Err(other),
        }
    }
}

impl From<AdditionalParams> for serde_json::Value {
    fn from(params: AdditionalParams) -> Self {
        params.into_value()
    }
}

impl std::ops::Index<&str> for AdditionalParams {
    type Output = serde_json::Value;

    /// Returns the value under `key`.
    ///
    /// # Panics
    /// Panics if the key is absent.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, key: &str) -> &serde_json::Value {
        &self.0[key]
    }
}

impl<'de> Deserialize<'de> for AdditionalParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match Self::try_from_value(serde_json::Value::deserialize(deserializer)?) {
            Ok(Some(params)) => Ok(params),
            // `null` and `{}` canonicalize to absence, which a bare
            // (non-`Option`) slot cannot express.
            Ok(None) => Err(serde::de::Error::custom(
                "`additional_params` carries no data — omit the field (an `Option` \
                 field routed through `optional_additional_params` canonicalizes \
                 `{}` and `null` to absent)",
            )),
            Err(_) => Err(serde::de::Error::custom(
                "`additional_params` must be a non-empty JSON object",
            )),
        }
    }
}

/// Returns dot-separated paths whose original values are missing or changed
/// after a round trip. Ignores added keys, null object members, and missing
/// object members whose original value was an empty object. Array positions
/// are compared individually.
///
/// ```
/// use rig_core::message;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let original = serde_json::json!({
///     "role": "assistant",
///     "content": [{"type": "text", "text": "cited", "citations": ["not re-nested"]}],
/// });
/// let loaded: message::Message = serde_json::from_value(original.clone())?;
/// let round_tripped = serde_json::to_value(&loaded)?;
/// let lost = message::keys_lost_in_round_trip(&original, &round_tripped);
/// assert_eq!(lost, vec!["content.0.citations".to_string()]);
/// # Ok(())
/// # }
/// ```
pub fn keys_lost_in_round_trip(
    original: &serde_json::Value,
    round_tripped: &serde_json::Value,
) -> Vec<String> {
    fn walk(
        original: &serde_json::Value,
        round_tripped: &serde_json::Value,
        path: &mut String,
        lost: &mut Vec<String>,
    ) {
        match (original, round_tripped) {
            (serde_json::Value::Object(original_map), serde_json::Value::Object(round_map)) => {
                for (key, original_value) in original_map {
                    if original_value.is_null() {
                        continue;
                    }
                    let checkpoint = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(key);
                    match round_map.get(key) {
                        Some(round_value) => walk(original_value, round_value, path, lost),
                        // Empty objects may canonicalize to absent metadata.
                        None => {
                            if !original_value
                                .as_object()
                                .is_some_and(serde_json::Map::is_empty)
                            {
                                lost.push(path.clone());
                            }
                        }
                    }
                    path.truncate(checkpoint);
                }
            }
            (serde_json::Value::Array(original_items), serde_json::Value::Array(round_items)) => {
                for (index, original_value) in original_items.iter().enumerate() {
                    let checkpoint = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(&index.to_string());
                    match round_items.get(index) {
                        Some(round_value) => walk(original_value, round_value, path, lost),
                        None => lost.push(path.clone()),
                    }
                    path.truncate(checkpoint);
                }
            }
            (original, round_tripped) => {
                if original != round_tripped {
                    lost.push(path.clone());
                }
            }
        }
    }

    let mut lost = Vec::new();
    walk(original, round_tripped, &mut String::new(), &mut lost);
    lost
}

/// Deserializes optional metadata, mapping null and empty objects to `None`.
/// Nonempty objects produce metadata; other shapes return a deserialization error.
pub fn optional_additional_params<'de, D>(
    deserializer: D,
) -> Result<Option<AdditionalParams>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(value) => AdditionalParams::try_from_value(value).map_err(|_| {
            serde::de::Error::custom("`additional_params` must be a JSON object (or null)")
        }),
    }
}

/// Text with optional provider metadata under the named `additional_params` key.
/// Unknown sibling fields are ignored on decode, not captured for replay.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    /// Text content.
    pub text: String,
    /// Provider-specific text fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

impl Text {
    /// Construct a new text block with no provider-specific fields.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            additional_params: None,
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

/// Image content containing image data and metadata about it.
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
    /// Provider-specific image fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
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
    /// Provider-specific audio fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// Video content containing video data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Video {
    /// Video source data.
    pub data: DocumentSourceKind,
    /// Video media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<VideoMediaType>,
    /// Provider-specific video fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
}

/// Document content containing document data and metadata about it.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Document {
    /// Document source data.
    pub data: DocumentSourceKind,
    /// Document media type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
    /// Provider-specific document fields.
    #[serde(
        default,
        deserialize_with = "optional_additional_params",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_params: Option<AdditionalParams>,
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

    /// Whether a service replaying reasoning `issuers` issued has anything to
    /// read in this message: false only for an assistant message whose every
    /// part is reasoning none of them opens.
    pub fn replays_to(&self, issuers: &[Issuer]) -> bool {
        match self {
            Message::Assistant { content, .. } => content.iter().any(|part| match part {
                AssistantContent::Reasoning(reasoning) => reasoning.open_for(issuers).is_some(),
                _ => true,
            }),
            Message::System { .. } | Message::User { .. } => true,
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

    /// Creates an assistant message containing one text block and no provider ID.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::text(text)],
        }
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
                additional_params: None,
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
        video_base64 => Video(VideoMediaType, Base64: String);
        /// Creates user video content from unencoded bytes.
        video_raw => Video(VideoMediaType, Raw: Vec<u8>);
        /// Creates user video content referencing a URL.
        video_url => Video(VideoMediaType, Url: String);
        /// Creates user document content from base64-encoded data.
        document_base64 => Document(DocumentMediaType, Base64: String);
        /// Creates user document content from unencoded bytes.
        document_raw => Document(DocumentMediaType, Raw: Vec<u8>);
        /// Creates user document content referencing a URL.
        document_url => Document(DocumentMediaType, Url: String);
        /// Creates user document content from literal text, such as a plain
        /// text or Markdown file. Binary formats belong in
        /// [`Self::document_base64`] or [`Self::document_raw`].
        document_text => Document(DocumentMediaType, String: String);
    }

    /// Creates a tool result answering the call `call` to the tool `name`.
    pub fn tool_result(call: CallId, name: ToolName, content: Vec<ToolResultContent>) -> Self {
        UserContent::ToolResult(ToolResult {
            call,
            name,
            content,
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
        AssistantContent::ToolCall(ToolCall::from_wire(id, ToolFunction { name, arguments }))
    }

    /// Dual-identifier variant (OpenAI Responses): `id` is the output-item
    /// handle (`fc_…`), `call_id` the correlator (`call_…`).
    pub fn tool_call_with_call_id(
        id: impl Into<String>,
        call_id: String,
        name: ToolName,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall::from_dual_wire(
            id,
            call_id,
            ToolFunction { name, arguments },
        ))
    }

    /// Creates reasoning text issued by `issuer`.
    pub fn reasoning(issuer: impl Into<Issuer>, reasoning: impl AsRef<str>) -> Self {
        AssistantContent::Reasoning(Reasoning::new(reasoning.as_ref()).sealed(issuer))
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
                Text {
                    text: text.into(),
                    additional_params: None,
                }
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
                Message::Assistant {
                    id: None,
                    content: vec![AssistantContent::$variant(value.into())],
                }
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
        Message::Assistant {
            id: None,
            content: vec![content],
        }
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
        Message::Assistant { id: None, content }
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
