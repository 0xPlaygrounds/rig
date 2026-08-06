//! Provenance-carrying part identity for the streaming layer.
//!
//! One streamed part has exactly one identity, and that identity knows where
//! it came from: [`PartId::Wire`] is an identifier the provider put on the
//! wire, and [`PartId::Minted`] is an identifier rig fabricated at a stream
//! boundary because the wire supplied none. The two have different rights:
//!
//! - A **wire** identity is durable. It may populate the replayable message
//!   types ([`crate::message::Reasoning::id`], [`crate::message::ToolCall::id`])
//!   and travel upstream on the next request.
//! - A **minted** identity exists to key accumulation while the stream is
//!   live. It never reaches a request: the only way to obtain a
//!   request-serializable id is [`WireId`], and a [`WireId`] can only be
//!   produced from [`PartId::Wire`]. There is deliberately no `Serialize`
//!   impl on [`PartId`] and no accessor that renders a minted identity into
//!   the durable id space — the leak is a compile error, not a runtime gate.
//!
//! Reference designs: pydantic-ai's `_parts_manager` keeps the accumulation
//! key in a private side map that dies with the stream; vercel's content
//! types have no id field for a block id to leak into. Both reach the same
//! invariant this type states directly: *the stream accumulation key and the
//! durable provider handle are different things.*

/// What kind of part a minted identity was fabricated for.
///
/// The kind partitions the minted id space so identities minted by different
/// subsystems (reasoning blocks, anthropic/bedrock content blocks, Responses
/// output items, tool-call fragments, text blocks) can never collide even
/// when their indices do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MintKind {
    /// Reasoning blocks on constant-id wires (gemini REST, ollama,
    /// chat-compat `reasoning_content`, candle).
    Reasoning,
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
    /// The minted identity for a wire-supplied index (anthropic's
    /// content-block index pattern). Unsigned by contract: signed wire index
    /// types must be converted at the adapter boundary, so a negative index
    /// is a decode error there rather than a divergent identity here.
    pub fn for_wire_index(self, index: u64) -> PartId {
        PartId::Minted { kind: self, index }
    }

    /// Stable lowercase name, used only for display/debug rendering.
    fn name(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::Block => "block",
            Self::Output => "output",
            Self::Tool => "tool",
            Self::Text => "text",
        }
    }
}

/// Identity of one streamed part, carrying its provenance in the type.
///
/// See the module docs for the contract. `PartId` deliberately implements
/// neither `Serialize` nor `Deserialize`: it is a stream-scoped key, not a
/// durable value. The public stream items render it for consumer-side
/// correlation via [`PartId::render`]; the durable message types receive an
/// id only through [`PartId::into_wire_id`] / [`PartId::as_wire`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PartId {
    /// Identity the provider put on the wire. Round-trips upstream.
    Wire(String),
    /// Identity rig minted at a stream boundary. Keys accumulation for the
    /// life of the stream and never leaves it.
    Minted {
        /// The subsystem that minted this identity.
        kind: MintKind,
        /// Position within the mint's own sequence (a counter or the wire's
        /// unsigned index). Unsigned by construction: no wire index can
        /// render a shape the identity machinery disagrees about.
        index: u64,
    },
}

/// A bare string is by definition a wire identity — fabricating a
/// [`PartId::Minted`] requires naming a [`MintKind`] explicitly (normally via
/// [`SyntheticIds`]), so no conversion can accidentally launder a fabricated
/// id into the wire-genuine space.
impl From<String> for PartId {
    fn from(id: String) -> Self {
        Self::Wire(id)
    }
}

impl From<&str> for PartId {
    fn from(id: &str) -> Self {
        Self::Wire(id.to_owned())
    }
}

impl PartId {
    /// A wire-supplied identity.
    pub fn wire(id: impl Into<String>) -> Self {
        Self::Wire(id.into())
    }

    /// The wire identity, if this id is wire-genuine. `None` for minted ids:
    /// this is the funnel that keeps fabricated identities out of the
    /// durable message types.
    pub fn as_wire(&self) -> Option<&str> {
        match self {
            Self::Wire(id) => Some(id),
            Self::Minted { .. } => None,
        }
    }

    /// Consume into the request-serializable [`WireId`], if wire-genuine.
    pub fn into_wire_id(self) -> Option<WireId> {
        match self {
            Self::Wire(id) => Some(WireId(id)),
            Self::Minted { .. } => None,
        }
    }

    /// Whether this identity was minted at a stream boundary.
    pub fn is_minted(&self) -> bool {
        matches!(self, Self::Minted { .. })
    }

    /// Render for consumer-side correlation on the public stream items.
    ///
    /// The rendering of a minted id is namespaced (`rig:reasoning:0`) and
    /// unique **within one stream** only; it restarts on every turn of a
    /// multi-turn run, so consumers must not key across streams by it. It is
    /// not a provider identifier and there is no path from this string back
    /// into a request: durable ids come only from [`PartId::into_wire_id`].
    pub fn render(&self) -> String {
        match self {
            Self::Wire(id) => id.clone(),
            Self::Minted { kind, index } => format!("rig:{}:{}", kind.name(), index),
        }
    }
}

/// A request-serializable, wire-genuine identifier.
///
/// The only constructor is [`PartId::into_wire_id`], which refuses minted
/// identities — request serializers that require this type therefore cannot
/// receive a fabricated id, by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WireId(String);

impl WireId {
    /// The wire identifier, ready for a request payload.
    pub fn into_string(self) -> String {
        self.0
    }

    /// Borrow the wire identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Fabricated per-stream identity for wires that carry none.
///
/// Every id-less wire mints ids the same way — a [`MintKind`] plus a counter
/// or the wire's own unsigned index — and the result is a [`PartId::Minted`]
/// that structurally cannot reach a request. This is the **only** mint:
/// providers never hand-roll a minted identity.
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

    /// Ids for reasoning blocks on constant-id wires.
    pub fn reasoning() -> Self {
        Self::new(MintKind::Reasoning)
    }

    /// Ids for content blocks on index-as-id wires.
    pub fn block() -> Self {
        Self::new(MintKind::Block)
    }

    /// Ids for the Responses `output_index` fallback.
    pub fn output() -> Self {
        Self::new(MintKind::Output)
    }

    /// Ids for tool-call fragments whose wire supplies no tool-call id.
    pub fn tool() -> Self {
        Self::new(MintKind::Tool)
    }

    /// Ids for text blocks opened by a bare `Message`.
    pub fn text() -> Self {
        Self::new(MintKind::Text)
    }

    /// Mint the next counter-based id (vercel's `blockCounter++` pattern).
    pub fn mint(&mut self) -> PartId {
        let id = self.for_index(self.next);
        self.next = self.next.saturating_add(1);
        id
    }

    /// The id for a stable wire-supplied index; see
    /// [`MintKind::for_wire_index`].
    pub fn for_index(&self, index: u64) -> PartId {
        self.kind.for_wire_index(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compile-time contract, stated as a runtime probe of the API
    /// shape: a minted id has no accessor into the durable id space. (The
    /// stronger property — `PartId: !Serialize` and `WireId`'s constructor
    /// privacy — is enforced by the `identity_leak` compile-fail tests.)
    #[test]
    fn minted_ids_have_no_wire_rendering() {
        let minted = PartId::Minted {
            kind: MintKind::Reasoning,
            index: 0,
        };
        assert_eq!(minted.as_wire(), None);
        assert!(minted.into_wire_id().is_none());
    }

    #[test]
    fn wire_ids_round_trip() {
        let id = PartId::wire("rs_123");
        assert_eq!(id.as_wire(), Some("rs_123"));
        assert_eq!(
            id.into_wire_id().expect("wire-genuine").into_string(),
            "rs_123"
        );
    }

    #[test]
    fn mint_counts_up_per_stream() {
        let mut ids = SyntheticIds::reasoning();
        assert_eq!(
            ids.mint(),
            PartId::Minted {
                kind: MintKind::Reasoning,
                index: 0
            }
        );
        assert_eq!(
            ids.mint(),
            PartId::Minted {
                kind: MintKind::Reasoning,
                index: 1
            }
        );
    }

    /// Injectivity across kinds and indices: two mints are equal only when
    /// both kind and index are equal, so identities minted by different
    /// subsystems can never collide. (Structural for an enum of plain data —
    /// pinned so a future change to the shape keeps the property.)
    #[test]
    fn minted_ids_are_injective_across_kinds_and_indices() {
        let kinds = [
            MintKind::Reasoning,
            MintKind::Block,
            MintKind::Output,
            MintKind::Tool,
            MintKind::Text,
        ];
        let indices = [0u64, 1, 7, u64::MAX];
        let mut seen = std::collections::HashSet::new();
        for kind in kinds {
            for index in indices {
                assert!(
                    seen.insert(PartId::Minted { kind, index }),
                    "collision at {kind:?}:{index}"
                );
            }
        }
        // Minted ids never collide with wire ids, including a wire id that
        // happens to spell a minted rendering.
        for kind in kinds {
            for index in indices {
                let minted = PartId::Minted { kind, index };
                assert_ne!(minted, PartId::wire(minted.render()));
            }
        }
    }
}
