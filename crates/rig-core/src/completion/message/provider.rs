//! Provider data a message carries without giving it a canonical meaning: a
//! whole output item kept verbatim ([`ProviderItem`]) and the metadata one
//! wire dialect attaches to a text block ([`TextExtras`]).
//!
//! The enum variant names the exact wire dialect, so an encoder only reads
//! its own variant and one dialect's data never reaches another's request. A
//! provider item is also [`Sealed`](super::Sealed) to the service that
//! issued it, because its ids and payloads mean nothing to another service.
//!
//! ```
//! use rig_core::message::{AssistantContent, ProviderItem, Verbatim};
//!
//! let block = serde_json::json!({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1"});
//! let item = ProviderItem::AnthropicMessages(Verbatim::try_from(block)?);
//! assert_eq!(item.kind(), "web_fetch_tool_result");
//! let content = AssistantContent::Provider(item.sealed("anthropic"));
//! assert!(matches!(content, AssistantContent::Provider(_)));
//! # Ok::<(), rig_core::message::NotAnObject>(())
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Issuer, Sealed};

/// One wire object exactly as the provider sent it, its `type` tag included.
///
/// Rig never reads or writes keys inside it: it is replayed byte for byte
/// (up to key order) to the dialect that produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Verbatim(Map<String, Value>);

/// A provider item was not a JSON object.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("a provider item must be a JSON object")]
pub struct NotAnObject(pub Value);

impl Verbatim {
    /// The object as the provider sent it.
    pub fn new(object: Map<String, Value>) -> Self {
        Self(object)
    }

    /// The wire `type` tag, or `""` when the object has none.
    pub fn kind(&self) -> &str {
        self.0
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The object.
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.0
    }

    /// The object, owned.
    pub fn into_map(self) -> Map<String, Value> {
        self.0
    }

    /// The object decoded as `T`, for a caller that models the item.
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        T::deserialize(Value::Object(self.0.clone()))
    }
}

/// The object as compact JSON.
impl std::fmt::Display for Verbatim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = serde_json::to_string(&self.0).map_err(|_| std::fmt::Error)?;
        f.write_str(&text)
    }
}

impl TryFrom<Value> for Verbatim {
    type Error = NotAnObject;

    fn try_from(value: Value) -> Result<Self, NotAnObject> {
        match value {
            Value::Object(object) => Ok(Self(object)),
            other => Err(NotAnObject(other)),
        }
    }
}

impl From<Verbatim> for Value {
    fn from(verbatim: Verbatim) -> Self {
        Value::Object(verbatim.0)
    }
}

/// An output item with no canonical meaning (a hosted tool's call or result,
/// a compaction, an item type Rig does not know yet), kept in its place in
/// the turn and replayed only to the dialect named by its variant.
///
/// Serialized as `{"dialect": "...", "item": {...}}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dialect", content = "item", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProviderItem {
    /// One Anthropic Messages content block.
    AnthropicMessages(Verbatim),
    /// One OpenAI Responses output item.
    OpenAiResponses(Verbatim),
}

impl ProviderItem {
    /// The verbatim object, whatever its dialect.
    pub fn verbatim(&self) -> &Verbatim {
        match self {
            Self::AnthropicMessages(item) | Self::OpenAiResponses(item) => item,
        }
    }

    /// The item's wire `type` tag.
    pub fn kind(&self) -> &str {
        self.verbatim().kind()
    }

    /// This item, readable only by `issuer`.
    pub fn sealed(self, issuer: impl Into<Issuer>) -> Sealed<Self> {
        Sealed::new(issuer, self)
    }
}

/// Metadata one wire dialect attached to a text block. Only that dialect's
/// encoder reads it; every other encoder sends the text without it.
///
/// It is not sealed to a service: it describes the block in the dialect's
/// schema (citations, annotations, a message phase) rather than carrying a
/// service's private state, so every service speaking the dialect accepts it.
///
/// Serialized as `{"dialect": "...", <fields>}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dialect", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TextExtras {
    /// Anthropic Messages text metadata.
    AnthropicMessages(crate::providers::anthropic::completion::TextExtras),
    /// OpenAI Responses text metadata.
    OpenAiResponses(crate::providers::openai::responses_api::TextExtras),
}

impl TextExtras {
    /// Merge `incoming` into `self` when both are the same dialect; a
    /// different dialect replaces `self`. Each dialect defines its own merge.
    pub fn merge(&mut self, incoming: Self) {
        match (self, incoming) {
            (Self::AnthropicMessages(existing), Self::AnthropicMessages(incoming)) => {
                existing.merge(incoming);
            }
            (Self::OpenAiResponses(existing), Self::OpenAiResponses(incoming)) => {
                existing.merge(incoming);
            }
            (existing, incoming) => *existing = incoming,
        }
    }
}
