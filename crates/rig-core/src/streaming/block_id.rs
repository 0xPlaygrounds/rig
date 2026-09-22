//! Stream-block identifiers with provider-issued or locally minted provenance.
//! Events for one block share a key. Minted keys are scoped to a stream and
//! must not become provider identifiers in replayed history.
//!
//! ```
//! use rig_core::streaming::{BlockId, MintKind};
//!
//! let key = BlockId::minted(MintKind::Text, 0);
//! assert!(key.is_minted());
//! assert_eq!(key.wire_str(), None);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

/// What kind of part a minted identity was fabricated for.
///
/// The kind partitions minted keys per subsystem so independent minters
/// need no coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MintKind {
    /// Reasoning blocks on constant-id wires (gemini REST, ollama,
    /// chat-compat `reasoning_content`, candle).
    Reasoning,
    /// Opaque reasoning payloads without provider IDs. A separate kind prevents
    /// encrypted blocks from replacing accumulated reasoning text.
    EncryptedReasoning,
    /// Content blocks on index-as-id wires (anthropic, bedrock).
    Block,
    /// OpenAI Responses `output_index` fallback for delta events lacking
    /// `item_id`.
    Output,
    /// Tool-call fragments whose wire omits the tool-call id.
    Tool,
    /// Text blocks opened by a bare `Message` on wires that never announce
    /// text-block boundaries.
    Text,
}

impl MintKind {
    /// The minted key for a wire-supplied index (anthropic's content-block
    /// index pattern). Unsigned by contract: signed wire index types must be
    /// converted at the adapter boundary, so a negative index is a decode
    /// error there rather than a divergent identity here.
    pub const fn for_wire_index(self, index: u64) -> BlockId {
        BlockId::minted(self, index)
    }

    /// Parse [`MintKind::as_str`]'s rendering.
    pub fn parse_name(name: &str) -> Option<Self> {
        [
            Self::Reasoning,
            Self::EncryptedReasoning,
            Self::Block,
            Self::Output,
            Self::Tool,
            Self::Text,
        ]
        .into_iter()
        .find(|kind| kind.as_str() == name)
    }

    /// The stable name used when a minted id is rendered.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::EncryptedReasoning => "encrypted_reasoning",
            Self::Block => "block",
            Self::Output => "output",
            Self::Tool => "tool",
            Self::Text => "text",
        }
    }
}

/// Identity of one streamed block for the life of its stream.
///
/// Serializes as `"wire:<id>"` or `"minted:<kind>:<index>"`; other encodings
/// are rejected. Display uses the provider ID or `{kind}-{index}`.
/// String conversions create wire IDs without validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockId {
    /// An identifier the provider put on the wire.
    Wire(String),
    /// A key rig minted at a stream boundary because the wire supplied none.
    ///
    /// Indices restart for each stream. Cross-turn maps must pair this key
    /// with a turn identifier.
    Minted {
        /// The subsystem that minted this key.
        kind: MintKind,
        /// Per-stream counter or unsigned wire index.
        index: u64,
    },
}

impl From<String> for BlockId {
    fn from(id: String) -> Self {
        Self::Wire(id)
    }
}

impl From<&str> for BlockId {
    fn from(id: &str) -> Self {
        Self::Wire(id.to_owned())
    }
}

impl BlockId {
    /// Creates a provider-issued key. The caller must supply a nonempty ID;
    /// use [`SyntheticIds`] when the provider supplies none.
    ///
    /// # Panics
    /// Panics on an empty ID when debug assertions are enabled.
    pub fn wire(id: impl Into<String>) -> Self {
        let id = id.into();
        debug_assert!(
            !id.is_empty(),
            "an empty wire id is not an id: mint instead"
        );
        Self::Wire(id)
    }

    /// A key minted at a stream boundary because the wire supplied none.
    /// `const` so per-stream constant keys can live in `const` items.
    pub const fn minted(kind: MintKind, index: u64) -> Self {
        Self::Minted { kind, index }
    }

    /// Parses a minted display name such as `tool-3`.
    /// Returns `None` for unknown kinds or invalid unsigned indices.
    pub fn from_minted_name(id: &str) -> Option<Self> {
        let (kind, index) = id.rsplit_once('-')?;
        let kind = MintKind::parse_name(kind)?;
        let index = index.parse().ok()?;
        Some(Self::Minted { kind, index })
    }

    pub const fn is_minted(&self) -> bool {
        matches!(self, Self::Minted { .. })
    }

    /// Returns the provider identifier, or `None` for a minted key.
    pub fn wire_str(&self) -> Option<&str> {
        match self {
            Self::Wire(wire) => Some(wire),
            Self::Minted { .. } => None,
        }
    }
}

impl Serialize for BlockId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Wire(wire) => serializer.serialize_str(&format!("wire:{wire}")),
            Self::Minted { kind, index } => {
                serializer.serialize_str(&format!("minted:{}:{index}", kind.as_str()))
            }
        }
    }
}

impl<'de> Deserialize<'de> for BlockId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let text = String::deserialize(deserializer)?;
        if let Some(wire) = text.strip_prefix("wire:") {
            return Ok(Self::Wire(wire.to_owned()));
        }
        if let Some(rest) = text.strip_prefix("minted:")
            && let Some((kind, index)) = rest.rsplit_once(':')
        {
            let kind = MintKind::parse_name(kind)
                .ok_or_else(|| D::Error::custom(format!("unknown mint kind `{kind}`")))?;
            let index = index
                .parse::<u64>()
                .map_err(|_| D::Error::custom(format!("invalid mint index `{index}`")))?;
            return Ok(Self::Minted { kind, index });
        }
        Err(D::Error::custom(format!(
            "a block id is `wire:<id>` or `minted:<kind>:<index>`, got `{text}`"
        )))
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wire(wire) => f.write_str(wire),
            Self::Minted { kind, index } => write!(f, "{}-{index}", kind.as_str()),
        }
    }
}

/// A provider-issued identifier, or `None` for the empty string: absence is
/// not an id, so no serializer ever needs an empty-string filter.
pub fn non_empty_id(id: impl Into<String>) -> Option<String> {
    let id = id.into();
    if id.is_empty() { None } else { Some(id) }
}

/// Per-stream counter for locally minted block keys.
/// These keys must not be sent as provider identifiers.
#[derive(Debug)]
pub struct SyntheticIds {
    kind: MintKind,
    next: u64,
}

impl SyntheticIds {
    /// A minter for `kind`.
    pub fn new(kind: MintKind) -> Self {
        Self { kind, next: 0 }
    }

    /// Keys for the Responses `output_index` fallback.
    pub fn output() -> Self {
        Self::new(MintKind::Output)
    }

    /// Keys for tool-call fragments whose wire supplies no tool-call id.
    pub fn tool() -> Self {
        Self::new(MintKind::Tool)
    }

    /// Keys for text blocks opened by a bare `Message`.
    pub fn text() -> Self {
        Self::new(MintKind::Text)
    }

    /// Returns the current key and advances the counter, saturating at `u64::MAX`.
    pub fn mint(&mut self) -> BlockId {
        let id = self.kind.for_wire_index(self.next);
        self.next = self.next.saturating_add(1);
        id
    }
}

#[cfg(test)]
mod tests;
