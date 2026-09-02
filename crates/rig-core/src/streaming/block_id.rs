//! Stream-block identity: the one key a streamed part is known by.
//!
//! One streamed part — a text block, a reasoning block, a tool call under
//! assembly — has exactly one identity for the life of its stream, the
//! [`BlockId`]. Every event about the part carries it: the accumulator keys
//! its maps by it, and the public stream items carry the same value, so a
//! consumer correlates a part's deltas with its completed block by simple
//! equality, and a host can persist or serialize an in-flight stream event
//! without a second correlator.
//!
//! The id keeps its **provenance**: a [`BlockId::Wire`] is an identifier the
//! provider actually issued; a [`BlockId::Minted`] was fabricated at the
//! adapter boundary because the wire supplied none. Provenance is data (the
//! id is serde) and it is load-bearing — the accumulator's adoption rule and
//! the sequence laws ask [`BlockId::is_minted`] — but it is never a *durable*
//! provider handle: the identifiers that travel back on a provider's wire
//! are the plain strings on the replayable message types
//! ([`crate::message::Reasoning::id`], [`crate::message::ToolCall::id`]),
//! populated from a wire id only, never from a minted key.

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
    /// Encrypted/opaque reasoning payloads on id-less wires (openrouter's
    /// `reasoning.encrypted` detail). A distinct kind from
    /// [`MintKind::Reasoning`] so a whole encrypted block can never restate —
    /// and replace — the text block accumulating under the wire's constant
    /// reasoning key.
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
/// `Eq + Hash + Clone` for keying, serde for the wire, `Display` for logs
/// (`wire id` as-is; a minted id as `{kind}-{index}`). Construction goes
/// through [`BlockId::wire`] and [`BlockId::minted`] /
/// [`MintKind::for_wire_index`]; a bare string converts to a wire id.
///
/// The serde form is one string, so a block id can key a JSON map:
/// `"wire:<id>"` or `"minted:<kind>:<index>"`. Decoding rejects anything
/// else — provenance is never guessed from the shape of an id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockId {
    /// An identifier the provider put on the wire.
    Wire(String),
    /// A key rig minted at a stream boundary because the wire supplied none.
    Minted {
        /// The subsystem that minted this key.
        kind: MintKind,
        /// Position within the mint's own sequence (a counter or the wire's
        /// unsigned index).
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
    /// A key derived from a wire-supplied identifier.
    pub fn wire(id: impl Into<String>) -> Self {
        Self::Wire(id.into())
    }

    /// A key minted at a stream boundary because the wire supplied none.
    /// `const` so per-stream constant keys can live in `const` items.
    pub const fn minted(kind: MintKind, index: u64) -> Self {
        Self::Minted { kind, index }
    }

    /// Whether this key was minted at a stream boundary (stream-internal
    /// lifecycle bookkeeping: minted-key reasoning items close on
    /// interleaving output).
    pub const fn is_minted(&self) -> bool {
        matches!(self, Self::Minted { .. })
    }

    /// The wire-supplied identifier this key was derived from, when it was.
    /// A minted key has none — and never becomes a durable provider handle.
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

/// Fabricated per-stream keys for wires that carry none.
///
/// Every id-less wire mints keys the same way — a [`MintKind`] plus a
/// counter or the wire's own unsigned index — and the result is a
/// [`BlockId::Minted`] that keys the stream and never reaches a request.
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

    /// Mint the next counter-based key (vercel's `blockCounter++` pattern).
    pub fn mint(&mut self) -> BlockId {
        let id = self.kind.for_wire_index(self.next);
        self.next = self.next.saturating_add(1);
        id
    }
}

#[cfg(test)]
mod tests;
