//! Stream-part identity: an opaque accumulation key, and the separate
//! durable provider handle.
//!
//! One streamed part has two distinct identity concerns, carried as two
//! values (never fused — review 84a43e9e root cause B):
//!
//! - [`StreamPartId`] — the **accumulation key**. Opaque: `Eq + Hash` and
//!   nothing else. It has no rendering, no serialization, and no path into a
//!   request or a public stream item; it exists to key the accumulator's
//!   maps for the life of one stream and then dies. Because nothing can
//!   observe it, an adapter may freely compose it
//!   ([`StreamPartId::Composite`], vercel's `` `${item.id}:0` `` move) or
//!   mint it ([`SyntheticIds`]) without any global-uniqueness obligation.
//! - [`WireId`] — the **durable provider handle**, present only when the
//!   provider actually issued one. It is the only value that may populate
//!   the replayable message types ([`crate::message::Reasoning::id`],
//!   [`crate::message::ToolCall::id`]) and travel upstream. Its only
//!   constructor rejects the empty string, so "absent" is `Option::None` —
//!   never a fabricated `""` a serializer must remember to filter.
//!
//! Consumer-facing correlation uses neither: public stream items carry
//! rig-generated correlators (`internal_call_id` for tool calls, the
//! part-scoped correlators the stream mints for reasoning), unique per run
//! by construction — pydantic-ai's part-index shape.
//!
//! Reference designs: vercel-ai-sdk carries the composite stream key on the
//! event and the durable handle in `providerMetadata.openai.itemId`;
//! pydantic-ai's `VendorId = Hashable` is an arbitrary private key with
//! durable ids as separate part fields.

/// What kind of part a minted identity was fabricated for.
///
/// The kind partitions minted keys per subsystem so independent minters
/// need no coordination. This is bookkeeping, not a public contract: the
/// key is opaque, so overlapping kinds could at most confuse a debugger,
/// never a consumer or a serializer.
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
    /// The minted key for a wire-supplied index (anthropic's content-block
    /// index pattern). Unsigned by contract: signed wire index types must be
    /// converted at the adapter boundary, so a negative index is a decode
    /// error there rather than a divergent identity here.
    pub fn for_wire_index(self, index: u64) -> StreamPartId {
        StreamPartId::Minted { kind: self, index }
    }
}

/// Opaque accumulation key of one streamed part.
///
/// `Eq + Hash + Clone + Debug` and nothing else — deliberately no
/// `Serialize`/`Deserialize`, no rendering, and no accessor into the
/// durable id space (see the module docs; the `identity_leak` compile-fail
/// suite pins this). Keys derived from wire ids ([`StreamPartId::Wire`])
/// stay distinguishable from minted ones because the accumulator's
/// interleaving-boundary lifecycle still asks
/// [`StreamPartId::is_minted`]; that discriminant is stream-internal
/// bookkeeping, not provenance a serializer may consult.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamPartId {
    /// A key derived from an identifier the provider put on the wire.
    ///
    /// This is a *key*, not the durable handle: the handle travels
    /// separately as [`WireId`] on the events that have one.
    Wire(String),
    /// A key rig minted at a stream boundary because the wire supplied none.
    Minted {
        /// The subsystem that minted this key.
        kind: MintKind,
        /// Position within the mint's own sequence (a counter or the wire's
        /// unsigned index).
        index: u64,
    },
    /// A key composed from a parent key and a sub-index — the Responses
    /// multi-part-under-one-item-id shape (vercel's `` `${item.id}:0` ``).
    /// Legal precisely because the key is unobservable.
    Composite {
        /// The parent part's key.
        parent: Box<StreamPartId>,
        /// Position under the parent.
        ordinal: u32,
    },
}

/// A bare string is by definition a wire-derived key — fabricating a
/// [`StreamPartId::Minted`] requires naming a [`MintKind`] explicitly
/// (normally via [`SyntheticIds`]), so no conversion can launder a
/// fabricated key into the wire-derived space. The empty string converts
/// too — as a *key* that is harmless (it can collide only with itself
/// within one stream) — but it carries no durable handle: [`WireId`]
/// construction is separate and rejects emptiness.
impl From<String> for StreamPartId {
    fn from(id: String) -> Self {
        Self::Wire(id)
    }
}

impl From<&str> for StreamPartId {
    fn from(id: &str) -> Self {
        Self::Wire(id.to_owned())
    }
}

impl StreamPartId {
    /// A key derived from a wire-supplied identifier.
    pub fn wire(id: impl Into<String>) -> Self {
        Self::Wire(id.into())
    }

    /// A key composed under `self` at `ordinal` (the Responses sibling
    /// shape).
    pub fn composed(&self, ordinal: u32) -> Self {
        Self::Composite {
            parent: Box::new(self.clone()),
            ordinal,
        }
    }

    /// Whether this key was minted at a stream boundary (stream-internal
    /// lifecycle bookkeeping: minted-key reasoning items close on
    /// interleaving output). A composite key inherits its parent's answer.
    pub fn is_minted(&self) -> bool {
        match self {
            Self::Wire(_) => false,
            Self::Minted { .. } => true,
            Self::Composite { parent, .. } => parent.is_minted(),
        }
    }
}

/// The durable provider handle: an identifier the provider actually issued,
/// ready for the replayable message types and request payloads.
///
/// The only constructor rejects the empty string, so an absent handle is
/// `Option::None` by construction — no serializer ever needs an
/// empty-string filter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WireId(String);

impl WireId {
    /// A provider-issued identifier. `None` for the empty string: absence
    /// is not an id.
    pub fn new(id: impl Into<String>) -> Option<Self> {
        let id = id.into();
        if id.is_empty() { None } else { Some(Self(id)) }
    }

    /// The identifier, ready for a request payload.
    pub fn into_string(self) -> String {
        self.0
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Fabricated per-stream keys for wires that carry none.
///
/// Every id-less wire mints keys the same way — a [`MintKind`] plus a
/// counter or the wire's own unsigned index — and the result is a
/// [`StreamPartId::Minted`] that, like every stream key, structurally
/// cannot reach a request or a public stream item.
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

    /// Keys for reasoning blocks on constant-id wires.
    pub fn reasoning() -> Self {
        Self::new(MintKind::Reasoning)
    }

    /// Keys for content blocks on index-as-id wires.
    pub fn block() -> Self {
        Self::new(MintKind::Block)
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
    pub fn mint(&mut self) -> StreamPartId {
        let id = self.for_index(self.next);
        self.next = self.next.saturating_add(1);
        id
    }

    /// The key for a stable wire-supplied index; see
    /// [`MintKind::for_wire_index`].
    pub fn for_index(&self, index: u64) -> StreamPartId {
        self.kind.for_wire_index(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The opaque-key contract, at the API-shape level (the stronger
    /// property — no `Serialize`, no rendering, `WireId` rejecting
    /// emptiness at its only constructor — is enforced by the
    /// `identity_leak` compile-fail tests).
    #[test]
    fn an_absent_provider_handle_is_none_not_empty() {
        assert!(WireId::new("").is_none());
        assert_eq!(
            WireId::new("rs_123").expect("non-empty").into_string(),
            "rs_123"
        );
    }

    #[test]
    fn mint_counts_up_per_stream() {
        let mut ids = SyntheticIds::reasoning();
        assert_eq!(
            ids.mint(),
            StreamPartId::Minted {
                kind: MintKind::Reasoning,
                index: 0
            }
        );
        assert_eq!(
            ids.mint(),
            StreamPartId::Minted {
                kind: MintKind::Reasoning,
                index: 1
            }
        );
    }

    /// Distinctness within one stream is what the accumulator needs; the
    /// key space is otherwise obligation-free (opaque). Composite keys are
    /// distinct from their parents and from each other, and inherit the
    /// minted lifecycle discriminant.
    #[test]
    fn composite_keys_are_distinct_and_inherit_mintedness() {
        let wire = StreamPartId::wire("rs_1");
        assert_ne!(wire, wire.composed(0));
        assert_ne!(wire.composed(0), wire.composed(1));
        assert!(!wire.composed(3).is_minted());

        let minted = MintKind::Reasoning.for_wire_index(0);
        assert!(minted.composed(2).is_minted());
        assert_ne!(minted.composed(0), wire.composed(0));
    }

    /// Keys minted by different subsystems stay distinct even at equal
    /// indices — bookkeeping hygiene (nothing can observe a collision, but
    /// the accumulator's maps deserve distinct keys anyway).
    #[test]
    fn minted_keys_are_distinct_across_kinds_and_indices() {
        let kinds = [
            MintKind::Reasoning,
            MintKind::Block,
            MintKind::Output,
            MintKind::Tool,
            MintKind::Text,
        ];
        let mut seen = std::collections::HashSet::new();
        for kind in kinds {
            for index in [0u64, 1, 7, u64::MAX] {
                assert!(
                    seen.insert(StreamPartId::Minted { kind, index }),
                    "collision at {kind:?}:{index}"
                );
            }
        }
    }
}
