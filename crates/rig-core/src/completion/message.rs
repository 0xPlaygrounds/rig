use serde::{Deserialize, Serialize};
use std::{convert::Infallible, str::FromStr};
use thiserror::Error;

use super::CompletionError;

// ================================================================
// Message models
// ================================================================

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

/// The shared wording for a response whose converted choice is empty.
///
/// Most provider decodes reject that state through
/// [`require_non_empty_response`]; anthropic uses this constant directly (it
/// branches on `stop_reason` before deciding emptiness, so the helper does
/// not fit its shape). Sharing the literal keeps a wording change from
/// silently forking the error text across wires; a guard rejecting a
/// *different* state (a role mismatch, a missing message) keeps its own
/// text instead.
pub const EMPTY_RESPONSE_ERROR: &str = "Response contained no message or tool call (empty)";

/// Reject an empty content list, with the error the call site chose.
///
/// Message content is a `Vec`, so "no content" is representable in the type.
/// Most wires nevertheless reject it — a completion that carried no message and
/// no tool call is a provider defect, and a history message with no blocks has
/// nothing to send — and at least one call site depends on that rejection as
/// control flow rather than as a diagnostic.
///
/// These guards used to be a side effect of the non-empty container's
/// constructor, which meant every site borrowed the same context-free "cannot
/// create with an empty vector". Stated explicitly here, each site keeps its own
/// message, which is where the useful detail lives.
///
/// Two rules for anyone extending this:
///
/// - It is **mostly** a guard for the **response** direction. Empty assistant
///   content is legal at the rig level — a tool-call-only turn, a truncated
///   stream — but a provider returning nothing where its protocol promises
///   content is malformed, and that is what most of these call sites detect.
///   Request-direction emptiness at the rig level is rejected once, at the
///   request boundary — but a few request-conversion sites also use this guard,
///   because non-empty rig content can still convert to zero *wire* blocks
///   (e.g. assistant content whose only parts have no representation on that
///   wire), and only the provider's own conversion can see that. If your
///   request `TryFrom` can drop parts, guard the converted list too.
/// - The check is on the **whole list**, never on individual items. A visibly
///   empty block can still carry data that must survive a round trip: reasoning
///   signatures and encrypted reasoning attach to blocks whose text is empty.
///   Emptiness is a property of the list, not of its members.
pub fn require_non_empty<T, E>(items: Vec<T>, error: impl FnOnce() -> E) -> Result<Vec<T>, E> {
    if items.is_empty() {
        return Err(error());
    }
    Ok(items)
}

/// [`require_non_empty`] with the shared response-direction rejection — the
/// one-line guard for a provider decode whose converted choice is empty and
/// that has no extra context to add. Pairing the guard with
/// [`EMPTY_RESPONSE_ERROR`] here keeps the wording from forking per wire.
/// The one decode not routed through here is anthropic's, whose emptiness
/// decision is entangled with `stop_reason` branching — it uses
/// [`EMPTY_RESPONSE_ERROR`] directly so the wording still cannot fork.
pub fn require_non_empty_response<T>(items: Vec<T>) -> Result<Vec<T>, CompletionError> {
    require_non_empty(items, || {
        CompletionError::ResponseError(EMPTY_RESPONSE_ERROR.to_owned())
    })
}

/// The `Option` sibling of [`require_non_empty`]: `None` for an empty list,
/// `Some(items)` otherwise.
///
/// This is the one home for the "an empty list means absent" rule, wherever a
/// list is being placed into an `Option`-shaped slot (an optional response
/// content, an optional message) rather than validated. The same whole-list
/// rule applies: never decide emptiness per item.
pub fn non_empty<T>(items: Vec<T>) -> Option<Vec<T>> {
    if items.is_empty() { None } else { Some(items) }
}

/// Describes the content of a message, which can be text, a tool result, an image, audio, or
///  a document. Dependent on provider supporting the content type. Multimedia content is generally
///  base64 (defined by it's format) encoded but additionally supports urls (for some providers).
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

/// Describes responses from a provider which is either text or a tool call.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum AssistantContent {
    /// Plain assistant text.
    Text(Text),
    /// Tool call requested by the assistant.
    ToolCall(ToolCall),
    /// Structured reasoning emitted by the assistant.
    Reasoning(Reasoning),
    /// Image content emitted by the assistant.
    Image(Image),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
#[non_exhaustive]
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
#[non_exhaustive]
/// Assistant reasoning payload with an optional provider-supplied identifier.
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

    /// Set or clear the provider reasoning ID.
    pub fn optional_id(mut self, id: Option<String>) -> Self {
        self.id = id;
        self
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

/// Tool result content containing information about a tool call and it's resulting content.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolResult {
    /// Which call this result answers — rig's correlation handle, always
    /// present. Copied from the answered [`ToolCall::id`], which is minted
    /// at the provider boundary when the provider issued no identifier.
    pub call: ToolCallId,
    /// What the provider issued for the answered call, if anything — the
    /// only identifiers that may travel back on that provider's wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderCallId>,
    /// Name of the tool that produced this result — the *executed* tool,
    /// which can differ from the model's call when a hook repaired it.
    ///
    /// Required: several wires key the replay on it (Gemini's
    /// `functionResponse.name`, Ollama's tool messages), and an identifier
    /// is not a name — rig used to smuggle the name through the id, which
    /// collided two calls to the same tool and misnamed cross-provider
    /// replays (review 84a43e9e #5).
    pub name: String,
    /// One or more content items produced by the tool.
    pub content: Vec<ToolResultContent>,
}

impl ToolResult {
    /// The identifier for a wire whose call-id slot is *required*: the
    /// provider-issued `call_id` when the provider issued one, else rig's
    /// minted handle — always non-empty.
    ///
    /// Wires whose id slot is *optional* (Gemini REST, gRPC) must read
    /// [`ToolResult::provider`] directly instead: minted handles never
    /// travel upstream there.
    pub fn wire_call_id(&self) -> &str {
        self.provider
            .as_ref()
            .map_or(self.call.as_str(), |provider| provider.call_id.as_str())
    }
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
            Self::Json { value } => serde_json::from_value(value.clone()),
            Self::Text(text) => serde_json::from_str(&text.text),
            Self::Image(_) => Err(<serde_json::Error as serde::de::Error>::custom(
                "cannot decode image tool-result content as JSON",
            )),
        }
    }
}

/// Error adopting the empty string as a tool-call identifier.
///
/// Absence is `None` on [`ToolCall::provider`] (or a minted [`ToolCallId`]),
/// never `""` — the empty-string sentinel is unrepresentable on these types.
#[derive(Debug, thiserror::Error)]
#[error("a tool-call identifier cannot be the empty string; absence is `None` or a minted id")]
pub struct EmptyToolCallId;

/// Rig's tool-call correlation handle: non-empty by construction, minted at
/// the provider boundary when the provider issued no identifier.
///
/// Mirrors the streaming layer's `WireId` idiom — one idiom, not two —
/// except that it is *required and minted* rather than optional: correlation
/// must always work, since a [`ToolResult`] must always name the call it
/// answers. Provider provenance lives on [`ToolCall::provider`], so a
/// consumer can still see that the provider issued nothing.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Adopt a provider-issued identifier. `None` for the empty string:
    /// absence is not an id.
    pub fn new(id: impl Into<String>) -> Option<Self> {
        let id = id.into();
        if id.is_empty() { None } else { Some(Self(id)) }
    }

    /// Mint a fresh, unique handle (21-character URL-safe id).
    pub fn mint() -> Self {
        Self(crate::id::generate())
    }

    /// Adopt `id` when non-empty, mint otherwise — the boundary guard for
    /// wires that may omit the identifier.
    pub fn new_or_mint(id: impl Into<String>) -> Self {
        Self::new(id).unwrap_or_else(Self::mint)
    }

    /// The correlation handle for the given provider identity: the
    /// provider's `call_id` when the provider issued one, minted when it
    /// did not. The single derivation the message and streaming layers
    /// share — a result correlates with its call because both derive the
    /// handle from the same provider identity.
    ///
    /// (`ProviderCallId`'s constructors reject the empty string, but its
    /// fields are public, so an empty `call_id` from a literal
    /// construction still mints rather than producing an empty handle.)
    pub fn for_provider(provider: Option<&ProviderCallId>) -> Self {
        provider
            .and_then(|provider| Self::new(provider.call_id.clone()))
            .unwrap_or_else(Self::mint)
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume into the underlying string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ToolCallId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::ops::Deref for ToolCallId {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for ToolCallId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ToolCallId {
    type Error = EmptyToolCallId;

    fn try_from(id: String) -> Result<Self, Self::Error> {
        Self::new(id).ok_or(EmptyToolCallId)
    }
}

impl From<ToolCallId> for String {
    fn from(id: ToolCallId) -> Self {
        id.0
    }
}

impl PartialEq<str> for ToolCallId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ToolCallId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// Wire shape for [`ProviderCallId`], so deserialization enforces the
/// non-empty `call_id` invariant.
#[derive(Deserialize)]
struct ProviderCallIdWire {
    call_id: String,
    #[serde(default)]
    item_id: Option<String>,
}

/// What the provider issued for a call — the only identifiers that may
/// travel back on that provider's wire.
///
/// Dual-identifier wires need both: OpenAI Responses issues an item id
/// (`fc_…`) *and* a `call_id` (`call_…`), and expects the right one in each
/// position. Single-identifier wires carry their id in `call_id` and leave
/// `item_id` empty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ProviderCallIdWire")]
pub struct ProviderCallId {
    /// The call-correlation identifier the provider expects echoed back.
    pub call_id: String,
    /// The output-item id issued alongside `call_id` on dual-identifier
    /// wires (OpenAI Responses `fc_…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

impl ProviderCallId {
    /// Adopt a provider-issued call identifier. `None` for the empty
    /// string: absence is not an id.
    pub fn new(call_id: impl Into<String>) -> Option<Self> {
        let call_id = call_id.into();
        if call_id.is_empty() {
            None
        } else {
            Some(Self {
                call_id,
                item_id: None,
            })
        }
    }

    /// Attach the dual-wire output-item id (empty strings are dropped).
    pub fn with_item_id(mut self, item_id: impl Into<String>) -> Self {
        let item_id = item_id.into();
        self.item_id = (!item_id.is_empty()).then_some(item_id);
        self
    }

    /// Derive the provider identity from a streaming part's optional wire
    /// handles. A dual wire carries `(call_id, item id)`; a single wire's id
    /// arrives as the tool/part id alone and becomes the `call_id`; with
    /// neither, the identity is absent. The empty-string filtering is
    /// load-bearing: [`ProviderCallId::new`] returns `None` on empty, so an
    /// empty `call_id` must fall through to the single-id arm rather than
    /// erase a real tool id.
    ///
    /// Both streaming surfaces (the parts accumulator and the raw
    /// `ToolCall` lift) derive through here so they cannot disagree.
    /// [`ToolCall::from_dual_wire`] is deliberately different — a dual wire
    /// that omits its `call_id` has no single-id fallback — and stays
    /// separate.
    pub fn from_optional_wire(call_id: Option<String>, tool_id: Option<String>) -> Option<Self> {
        let call_id = call_id.filter(|call_id| !call_id.is_empty());
        match (call_id, tool_id) {
            (Some(call_id), tool_id) => Self::new(call_id).map(|provider| match tool_id {
                Some(tool_id) => provider.with_item_id(tool_id),
                None => provider,
            }),
            (None, Some(tool_id)) => Self::new(tool_id),
            (None, None) => None,
        }
    }
}

impl TryFrom<ProviderCallIdWire> for ProviderCallId {
    type Error = EmptyToolCallId;

    fn try_from(wire: ProviderCallIdWire) -> Result<Self, Self::Error> {
        let Some(provider) = Self::new(wire.call_id) else {
            return Err(EmptyToolCallId);
        };
        Ok(match wire.item_id {
            Some(item_id) => provider.with_item_id(item_id),
            None => provider,
        })
    }
}

/// Describes a tool call with an id and function to call, generally produced by a provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    /// Rig's correlation handle. Always present; minted when the provider
    /// issued none.
    pub id: ToolCallId,
    /// What the provider issued, if anything — the only identifiers that
    /// may go back on the wire as *that provider's* handles. `None` means
    /// the provider issued no id (id-less wires such as Gemini REST or
    /// older Ollama daemons).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderCallId>,
    /// Function name and JSON arguments requested by the model.
    pub function: ToolFunction,
    /// Optional cryptographic signature for the tool call.
    ///
    /// This field is used by some providers (e.g., Google) to provide a signature
    /// that can verify the authenticity and integrity of the tool call. When present,
    /// it allows verification that the tool call was actually generated by the model
    /// and has not been tampered with.
    ///
    /// This is an optional, provider-specific feature and will be `None` for providers
    /// that don't support tool call signatures.
    #[serde(default)]
    pub signature: Option<String>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    #[serde(default)]
    pub additional_params: Option<serde_json::Value>,
}

impl ToolCall {
    /// A call with an explicit correlation handle and no provider-issued id.
    pub fn new(id: ToolCallId, function: ToolFunction) -> Self {
        Self {
            id,
            provider: None,
            function,
            signature: None,
            additional_params: None,
        }
    }

    /// The single-identifier provider boundary: adopt the wire's id when it
    /// issued one, mint when it did not (empty or absent ids mint).
    pub fn from_wire(wire_id: impl Into<String>, function: ToolFunction) -> Self {
        let provider = ProviderCallId::new(wire_id);
        let id = ToolCallId::for_provider(provider.as_ref());
        Self {
            id,
            provider,
            function,
            signature: None,
            additional_params: None,
        }
    }

    /// The dual-identifier provider boundary (OpenAI Responses): `item_id`
    /// is the output-item handle (`fc_…`), `call_id` the correlator
    /// (`call_…`). The correlator drives rig's id; empty ids mint.
    pub fn from_dual_wire(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        function: ToolFunction,
    ) -> Self {
        let provider = ProviderCallId::new(call_id).map(|provider| {
            let item_id = item_id.into();
            provider.with_item_id(item_id)
        });
        let id = ToolCallId::for_provider(provider.as_ref());
        Self {
            id,
            provider,
            function,
            signature: None,
            additional_params: None,
        }
    }

    /// Attach provider-issued identifiers.
    pub fn with_provider(mut self, provider: ProviderCallId) -> Self {
        self.provider = Some(provider);
        self
    }

    /// The identifier for a wire whose call-id slot is *required*: the
    /// provider-issued `call_id` when the provider issued one, else rig's
    /// minted handle — always non-empty.
    ///
    /// Wires whose id slot is *optional* (Gemini REST, gRPC) must read
    /// [`ToolCall::provider`] directly instead: minted handles never travel
    /// upstream there.
    pub fn wire_call_id(&self) -> &str {
        self.provider
            .as_ref()
            .map_or(self.id.as_str(), |provider| provider.call_id.as_str())
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
    pub name: String,
    /// JSON arguments for the tool/function.
    pub arguments: serde_json::Value,
}

impl ToolFunction {
    /// Create a tool function call payload.
    pub fn new(name: String, arguments: serde_json::Value) -> Self {
        Self { name, arguments }
    }
}

// ================================================================
// Base content models
// ================================================================

/// Basic text content.
///
/// `additional_params` carries provider-specific fields that arrive on text
/// content blocks (e.g. Anthropic returns citation metadata on assistant text
/// blocks). It is flattened during serialization so the inner JSON keys appear
/// at the same level as `text`, matching the wire format of provider APIs.
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Text {
    /// Text content.
    pub text: String,
    /// Provider-specific text fields.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
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
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl Image {
    pub fn try_into_url(self) -> Result<String, MessageError> {
        match self.data {
            DocumentSourceKind::Url(url) => Ok(url),
            DocumentSourceKind::Base64(data) => {
                let Some(media_type) = self.media_type else {
                    return Err(MessageError::ConversionError(
                        "A media type is required to create a valid base64-encoded image URL"
                            .to_string(),
                    ));
                };

                Ok(format!(
                    "data:image/{ty};base64,{data}",
                    ty = media_type.to_mime_type()
                ))
            }
            unknown => Err(MessageError::ConversionError(format!(
                "Tried to convert unknown type to a URL: {unknown:?}"
            ))),
        }
    }
}

/// The kind of image source (to be used).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
#[non_exhaustive]
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
    pub fn url(url: &str) -> Self {
        Self::Url(url.to_string())
    }

    /// Create a base64-backed source.
    pub fn base64(base64_string: &str) -> Self {
        Self::Base64(base64_string.to_string())
    }

    /// Create a provider file ID-backed source.
    pub fn file_id(file_id: &str) -> Self {
        Self::FileId(file_id.to_string())
    }

    /// Create a raw byte source.
    pub fn raw(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Raw(bytes.into())
    }

    /// Create a string-backed source.
    pub fn string(input: &str) -> Self {
        Self::String(input.into())
    }

    /// Create an unknown source placeholder.
    pub fn unknown() -> Self {
        Self::Unknown
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
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
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
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
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
    /// Provider-specific document fields.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

/// Describes the format of the content, which can be base64 or string.
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

// ================================================================
// Impl. for message models
// ================================================================

impl Message {
    /// This helper method is primarily used to extract the first string prompt from a `Message`.
    /// Since `Message` might have more than just text content, we need to find the first text.
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

    /// Helper constructor to make creating system messages easier.
    pub fn system(text: impl Into<String>) -> Self {
        Message::System {
            content: text.into(),
        }
    }

    /// Helper constructor to make creating user messages easier.
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::text(text)],
        }
    }

    /// Helper constructor to make creating assistant messages easier.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::text(text)],
        }
    }

    /// Helper constructor to make creating assistant messages easier.
    pub fn assistant_with_id(id: String, text: impl Into<String>) -> Self {
        Message::Assistant {
            id: Some(id),
            content: vec![AssistantContent::text(text)],
        }
    }

    /// Helper constructor to make creating tool result messages easier.
    /// `call` is the answered call's correlation handle — echo
    /// [`ToolCall::id`]; it is never recorded as a provider-issued
    /// identifier (see [`UserContent::tool_result`]). `name` is the
    /// executed tool's name.
    pub fn tool_result(
        call: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Message::User {
            content: vec![UserContent::tool_result(
                call,
                name,
                vec![ToolResultContent::text(content)],
            )],
        }
    }
}

impl UserContent {
    /// Helper constructor to make creating user text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text(text.into().into())
    }

    /// Helper constructor to make creating user image content easier.
    pub fn image_base64(
        data: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user image content from raw unencoded bytes easier.
    pub fn image_raw(
        data: impl Into<Vec<u8>>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            detail,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user image content easier.
    pub fn image_url(
        url: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        UserContent::Image(Image {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user audio content easier.
    pub fn audio(data: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user audio content from raw unencoded bytes easier.
    pub fn audio_raw(data: impl Into<Vec<u8>>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper to create an audio resource from a URL
    pub fn audio_url(url: impl Into<String>, media_type: Option<AudioMediaType>) -> Self {
        UserContent::Audio(Audio {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user video content easier.
    pub fn video(data: impl Into<String>, media_type: Option<VideoMediaType>) -> Self {
        UserContent::Video(Video {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating user video content from raw unencoded bytes easier.
    pub fn video_raw(data: impl Into<Vec<u8>>, media_type: Option<VideoMediaType>) -> Self {
        UserContent::Video(Video {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper to create a video resource from a URL
    pub fn video_url(url: impl Into<String>, media_type: Option<VideoMediaType>) -> Self {
        UserContent::Video(Video {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user document content easier.
    /// This creates a document that assumes the data being passed in is a raw string.
    pub fn document(data: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
        let data: String = data.into();
        UserContent::Document(Document {
            data: DocumentSourceKind::string(&data),
            media_type,
            additional_params: None,
        })
    }

    /// Helper to create a document from raw unencoded bytes
    pub fn document_raw(data: impl Into<Vec<u8>>, media_type: Option<DocumentMediaType>) -> Self {
        UserContent::Document(Document {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper to create a document from a URL
    pub fn document_url(url: impl Into<String>, media_type: Option<DocumentMediaType>) -> Self {
        UserContent::Document(Document {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            ..Default::default()
        })
    }

    /// Helper constructor to make creating user tool result content easier.
    ///
    /// `call` is the answered call's correlation handle — echo
    /// [`ToolCall::id`] (an empty string mints a fresh handle). It is
    /// recorded as the handle only, never as a provider-issued identifier:
    /// a bare string cannot prove provider provenance, and stamping a
    /// minted handle as one would send it upstream on wires whose id slot
    /// is optional. When you hold provider identifiers, use
    /// [`UserContent::tool_result_for`] (from an executed [`ToolCall`]) or
    /// [`UserContent::tool_result_from_wire`] (from the provider's wire).
    /// `name` is the executed tool's name (required — several wires key
    /// the replay on it).
    pub fn tool_result(
        call: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_mint(call),
            provider: None,
            name: name.into(),
            content,
        })
    }

    /// Tool result content at the single-identifier provider boundary —
    /// the inbound-converter form, mirroring [`ToolCall::from_wire`]:
    /// `wire_id` came off the provider's wire, so it is recorded as the
    /// provider-issued identifier (empty records none and mints the
    /// handle).
    pub fn tool_result_from_wire(
        wire_id: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        let provider = ProviderCallId::new(wire_id);
        let call = ToolCallId::for_provider(provider.as_ref());
        UserContent::ToolResult(ToolResult {
            call,
            provider,
            name: name.into(),
            content,
        })
    }

    /// Tool result content answering a specific call — the form the agent
    /// drivers use: `call`/`provider` come from the executed [`ToolCall`],
    /// `name` is the *executed* tool's name (which can differ from the
    /// model's call when a hook repaired it).
    pub fn tool_result_for(
        call: ToolCallId,
        provider: Option<ProviderCallId>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        UserContent::ToolResult(ToolResult {
            call,
            provider,
            name: name.into(),
            content,
        })
    }

    /// Tool result content for a dual-identifier wire (OpenAI Responses):
    /// `item_id` is the output-item handle (`fc_…`), `call_id` the
    /// correlator (`call_…`). Empty ids record no provider id and mint.
    pub fn tool_result_with_call_id(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        content: Vec<ToolResultContent>,
    ) -> Self {
        let provider = ProviderCallId::new(call_id).map(|provider| provider.with_item_id(item_id));
        let call = ToolCallId::for_provider(provider.as_ref());
        UserContent::ToolResult(ToolResult {
            call,
            provider,
            name: name.into(),
            content,
        })
    }
}

impl AssistantContent {
    /// Helper constructor to make creating assistant text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        AssistantContent::Text(text.into().into())
    }

    /// Helper constructor to make creating assistant image content easier.
    pub fn image_base64(
        data: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        AssistantContent::Image(Image {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make creating assistant tool call content easier.
    ///
    /// `id` is the provider-issued identifier when one exists; an empty
    /// `id` records no provider id and mints the correlation handle.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction {
                name: name.into(),
                arguments,
            },
        ))
    }

    /// Dual-identifier variant (OpenAI Responses): `id` is the output-item
    /// handle (`fc_…`), `call_id` the correlator (`call_…`).
    pub fn tool_call_with_call_id(
        id: impl Into<String>,
        call_id: String,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        AssistantContent::ToolCall(ToolCall::from_dual_wire(
            id,
            call_id,
            ToolFunction {
                name: name.into(),
                arguments,
            },
        ))
    }

    pub fn reasoning(reasoning: impl AsRef<str>) -> Self {
        AssistantContent::Reasoning(Reasoning::new(reasoning.as_ref()))
    }
}

impl ToolResultContent {
    /// Helper constructor to make creating tool result text content easier.
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text(text.into().into())
    }

    /// Helper constructor for structured JSON tool-result content.
    pub fn json(value: serde_json::Value) -> Self {
        ToolResultContent::Json { value }
    }

    /// Helper constructor to make tool result images from a base64-encoded string.
    pub fn image_base64(
        data: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Base64(data.into()),
            media_type,
            detail,
            additional_params: None,
        })
    }

    /// Helper constructor to make tool result images from a base64-encoded string.
    pub fn image_raw(
        data: impl Into<Vec<u8>>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Raw(data.into()),
            media_type,
            detail,
            ..Default::default()
        })
    }

    /// Helper constructor to make tool result images from a URL.
    pub fn image_url(
        url: impl Into<String>,
        media_type: Option<ImageMediaType>,
        detail: Option<ImageDetail>,
    ) -> Self {
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::Url(url.into()),
            media_type,
            detail,
            additional_params: None,
        })
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
            .or_else(|| {
                DocumentMediaType::from_mime_type(mime_type)
                    .map(MediaType::Document)
                    .or_else(|| {
                        AudioMediaType::from_mime_type(mime_type)
                            .map(MediaType::Audio)
                            .or_else(|| {
                                VideoMediaType::from_mime_type(mime_type).map(MediaType::Video)
                            })
                    })
            })
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

impl MimeType for ImageMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "image/jpeg" => Some(ImageMediaType::JPEG),
            "image/png" => Some(ImageMediaType::PNG),
            "image/gif" => Some(ImageMediaType::GIF),
            "image/webp" => Some(ImageMediaType::WEBP),
            "image/heic" => Some(ImageMediaType::HEIC),
            "image/heif" => Some(ImageMediaType::HEIF),
            "image/svg+xml" => Some(ImageMediaType::SVG),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            ImageMediaType::JPEG => "image/jpeg",
            ImageMediaType::PNG => "image/png",
            ImageMediaType::GIF => "image/gif",
            ImageMediaType::WEBP => "image/webp",
            ImageMediaType::HEIC => "image/heic",
            ImageMediaType::HEIF => "image/heif",
            ImageMediaType::SVG => "image/svg+xml",
        }
    }
}

impl MimeType for DocumentMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "application/pdf" => Some(DocumentMediaType::PDF),
            "text/plain" => Some(DocumentMediaType::TXT),
            "text/rtf" => Some(DocumentMediaType::RTF),
            "text/html" => Some(DocumentMediaType::HTML),
            "text/css" => Some(DocumentMediaType::CSS),
            "text/md" | "text/markdown" => Some(DocumentMediaType::MARKDOWN),
            "text/csv" => Some(DocumentMediaType::CSV),
            "text/xml" => Some(DocumentMediaType::XML),
            "application/x-javascript" | "text/x-javascript" => Some(DocumentMediaType::Javascript),
            "application/x-python" | "text/x-python" => Some(DocumentMediaType::Python),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            DocumentMediaType::PDF => "application/pdf",
            DocumentMediaType::TXT => "text/plain",
            DocumentMediaType::RTF => "text/rtf",
            DocumentMediaType::HTML => "text/html",
            DocumentMediaType::CSS => "text/css",
            DocumentMediaType::MARKDOWN => "text/markdown",
            DocumentMediaType::CSV => "text/csv",
            DocumentMediaType::XML => "text/xml",
            DocumentMediaType::Javascript => "application/x-javascript",
            DocumentMediaType::Python => "application/x-python",
        }
    }
}

impl MimeType for AudioMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self> {
        match mime_type {
            "audio/wav" => Some(AudioMediaType::WAV),
            "audio/mp3" => Some(AudioMediaType::MP3),
            "audio/aiff" => Some(AudioMediaType::AIFF),
            "audio/aac" => Some(AudioMediaType::AAC),
            "audio/ogg" => Some(AudioMediaType::OGG),
            "audio/flac" => Some(AudioMediaType::FLAC),
            "audio/m4a" => Some(AudioMediaType::M4A),
            "audio/pcm16" => Some(AudioMediaType::PCM16),
            "audio/pcm24" => Some(AudioMediaType::PCM24),
            _ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            AudioMediaType::WAV => "audio/wav",
            AudioMediaType::MP3 => "audio/mp3",
            AudioMediaType::AIFF => "audio/aiff",
            AudioMediaType::AAC => "audio/aac",
            AudioMediaType::OGG => "audio/ogg",
            AudioMediaType::FLAC => "audio/flac",
            AudioMediaType::M4A => "audio/m4a",
            AudioMediaType::PCM16 => "audio/pcm16",
            AudioMediaType::PCM24 => "audio/pcm24",
        }
    }
}

impl MimeType for VideoMediaType {
    fn from_mime_type(mime_type: &str) -> Option<Self>
    where
        Self: Sized,
    {
        match mime_type {
            "video/avi" => Some(VideoMediaType::AVI),
            "video/mp4" => Some(VideoMediaType::MP4),
            "video/mpeg" => Some(VideoMediaType::MPEG),
            "video/mov" => Some(VideoMediaType::MOV),
            "video/webm" => Some(VideoMediaType::WEBM),
            &_ => None,
        }
    }

    fn to_mime_type(&self) -> &'static str {
        match self {
            VideoMediaType::AVI => "video/avi",
            VideoMediaType::MP4 => "video/mp4",
            VideoMediaType::MPEG => "video/mpeg",
            VideoMediaType::MOV => "video/mov",
            VideoMediaType::WEBM => "video/webm",
        }
    }
}

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

// ================================================================
// FromStr, From<String>, and From<&str> impls
// ================================================================

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text {
            text,
            additional_params: None,
        }
    }
}

impl From<&String> for Text {
    fn from(text: &String) -> Self {
        text.to_owned().into()
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

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

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message::User {
            content: vec![UserContent::Text(text.into())],
        }
    }
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message::User {
            content: vec![UserContent::Text(text.into())],
        }
    }
}

impl From<&String> for Message {
    fn from(text: &String) -> Self {
        Message::User {
            content: vec![UserContent::Text(text.into())],
        }
    }
}

impl From<Text> for Message {
    fn from(text: Text) -> Self {
        Message::User {
            content: vec![UserContent::Text(text)],
        }
    }
}

impl From<Image> for Message {
    fn from(image: Image) -> Self {
        Message::User {
            content: vec![UserContent::Image(image)],
        }
    }
}

impl From<Audio> for Message {
    fn from(audio: Audio) -> Self {
        Message::User {
            content: vec![UserContent::Audio(audio)],
        }
    }
}

impl From<Document> for Message {
    fn from(document: Document) -> Self {
        Message::User {
            content: vec![UserContent::Document(document)],
        }
    }
}

impl From<String> for ToolResultContent {
    fn from(text: String) -> Self {
        ToolResultContent::text(text)
    }
}

impl From<String> for AssistantContent {
    fn from(text: String) -> Self {
        AssistantContent::text(text)
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::text(text)
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

impl From<ToolCall> for Message {
    fn from(tool_call: ToolCall) -> Self {
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(tool_call)],
        }
    }
}

impl From<ToolResult> for Message {
    fn from(tool_result: ToolResult) -> Self {
        Message::User {
            content: vec![UserContent::ToolResult(tool_result)],
        }
    }
}

impl From<ToolResultContent> for Message {
    fn from(tool_result_content: ToolResultContent) -> Self {
        Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: ToolCallId::mint(),
                provider: None,
                name: String::new(),
                content: vec![tool_result_content],
            })],
        }
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
        function_names: Vec<String>,
    },
}

// ================================================================
// Error types
// ================================================================

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}

impl From<MessageError> for CompletionError {
    fn from(error: MessageError) -> Self {
        CompletionError::RequestError(error.into())
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::{Message, Reasoning, ReasoningContent, Text, ToolResultContent};

    mod vec_content_serde {
        use super::super::{AssistantContent, Message, UserContent};

        #[test]
        fn message_content_still_serializes_as_a_plain_sequence() {
            // The removed container serialized as a bare sequence, which is why
            // this migration changes no persisted history and no recorded
            // provider fixture. Pin the wire shape so that stays true.
            let message = Message::User {
                content: vec![UserContent::text("hi")],
            };
            let json = serde_json::to_value(&message).expect("serialize");
            assert_eq!(
                json,
                serde_json::json!({
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                })
            );
        }

        #[test]
        fn message_content_round_trips_byte_identically() {
            let message = Message::Assistant {
                id: Some("msg_1".to_owned()),
                content: vec![AssistantContent::text("hello")],
            };
            let encoded = serde_json::to_string(&message).expect("serialize");
            let decoded: Message = serde_json::from_str(&encoded).expect("deserialize");
            assert_eq!(
                serde_json::to_string(&decoded).expect("re-serialize"),
                encoded
            );
        }

        #[test]
        fn an_empty_content_array_now_deserializes() {
            // The container's `Deserialize` implemented only `visit_seq` and
            // rejected `[]`. That is the single input whose behaviour this
            // migration changes: it was an error, and it is now an empty list.
            let message: Message =
                serde_json::from_value(serde_json::json!({"role": "user", "content": []}))
                    .expect("an empty content list is representable now");
            let Message::User { content } = message else {
                panic!("expected a user message");
            };
            assert!(content.is_empty());
        }
    }

    #[test]
    fn reasoning_constructors_and_accessors_work() {
        let single = Reasoning::new("think");
        assert_eq!(single.first_text(), Some("think"));
        assert_eq!(single.first_signature(), None);

        let signed = Reasoning::new_with_signature("signed", Some("sig-1".to_string()));
        assert_eq!(signed.first_text(), Some("signed"));
        assert_eq!(signed.first_signature(), Some("sig-1"));

        let multi = Reasoning::multi(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(multi.display_text(), "a\nb");
        assert_eq!(multi.first_text(), Some("a"));

        let redacted = Reasoning::redacted("redacted-value");
        assert_eq!(redacted.display_text(), "redacted-value");
        assert_eq!(redacted.first_text(), None);

        let encrypted = Reasoning::encrypted("enc");
        assert_eq!(encrypted.encrypted_content(), Some("enc"));
        assert_eq!(encrypted.display_text(), "");

        let summaries = Reasoning::summaries(vec!["s1".to_string(), "s2".to_string()]);
        assert_eq!(summaries.display_text(), "s1\ns2");
        assert_eq!(summaries.encrypted_content(), None);
    }

    #[test]
    fn reasoning_content_serde_roundtrip() {
        let variants = vec![
            ReasoningContent::Text {
                text: "plain".to_string(),
                signature: Some("sig".to_string()),
            },
            ReasoningContent::Encrypted("opaque".to_string()),
            ReasoningContent::Redacted {
                data: "redacted".to_string(),
            },
            ReasoningContent::Summary("summary".to_string()),
        ];

        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let roundtrip: ReasoningContent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(roundtrip, variant);
        }
    }

    #[test]
    fn system_message_constructor_and_serde_roundtrip() {
        let message = Message::system("You are concise.");

        match &message {
            Message::System { content } => assert_eq!(content, "You are concise."),
            _ => panic!("Expected system message"),
        }

        let json = serde_json::to_string(&message).expect("serialize");
        let roundtrip: Message = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, message);
    }

    #[test]
    fn current_schema_tool_call_json_round_trips_without_provider_promotion() {
        // A minted handle with no provider must stay provider-less —
        // nothing in the round trip may invent provider provenance.
        let call = super::ToolCall::new(
            super::ToolCallId::new("minted-handle").expect("non-empty"),
            super::ToolFunction {
                name: "add".to_string(),
                arguments: serde_json::json!({}),
            },
        );

        let json = serde_json::to_value(&call).expect("serialize");
        assert!(json.get("call_id").is_none());
        let roundtrip: super::ToolCall = serde_json::from_value(json).expect("deserialize");
        assert_eq!(roundtrip.provider, None);
        assert_eq!(roundtrip, call);
    }

    #[test]
    fn legacy_call_id_key_is_ignored_not_lifted() {
        // The pre-provider-split lift is deleted: a legacy `call_id` key is
        // an unknown field, so it deserializes with the key ignored — `id`
        // is read as rig's handle and `provider` stays absent. Pinned so a
        // future change (e.g. making the key a hard error) is a decision,
        // not an accident; the hand-migration recipe lives in MIGRATING.
        let legacy = serde_json::json!({
            "id": "fc_123",
            "call_id": "call_abc",
            "function": {"name": "add", "arguments": {"x": 1}},
        });

        let call: super::ToolCall = serde_json::from_value(legacy).expect("deserialize");
        assert_eq!(call.id, "fc_123");
        assert_eq!(call.provider, None);
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct ExecutorLikeResponse {
        output: serde_json::Value,
        logs: Vec<String>,
        execution_time_ms: u64,
    }

    #[test]
    fn tool_result_content_decodes_structured_and_legacy_json() {
        let response = ExecutorLikeResponse {
            output: serde_json::json!({"answer": 42}),
            logs: vec!["computed".to_string()],
            execution_time_ms: 7,
        };
        let value = serde_json::to_value(&response).expect("serialize response");

        let structured = ToolResultContent::json(value.clone());
        assert_eq!(structured.as_json(), Some(&value));
        assert_eq!(structured.as_text(), None);
        assert_eq!(
            structured
                .deserialize_json::<ExecutorLikeResponse>()
                .expect("decode structured response"),
            response
        );

        let legacy_json = value.to_string();
        let legacy_text = ToolResultContent::Text(Text::new(legacy_json.clone()));
        assert_eq!(legacy_text.as_text(), Some(legacy_json.as_str()));
        assert_eq!(legacy_text.as_json(), None);
        assert_eq!(
            legacy_text
                .deserialize_json::<ExecutorLikeResponse>()
                .expect("decode legacy response"),
            response
        );

        let image = ToolResultContent::image_url("https://example.com/result.png", None, None);
        let image_error = image.deserialize_json::<ExecutorLikeResponse>();
        assert!(image_error.is_err());
        if let Err(error) = image_error {
            assert_eq!(
                error.to_string(),
                "cannot decode image tool-result content as JSON"
            );
        }
    }
}
