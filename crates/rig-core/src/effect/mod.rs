//! The effect protocol as data.
//!
//! An *effect* is one request an agent (or any host) makes of the outside
//! world — a completion, a tool call, an embedding, a conversation-memory
//! operation, a retrieval — expressed as a value rather than as a call on a
//! trait object. The bus (`rig_agent::bus`) carries these values to the
//! handler registered for a [`HandlerKey`] and carries the [`Outcome`] back;
//! an `EffectLog` (`rig_effect_log`) records every exchange so a run can be
//! replayed.
//!
//! Everything in this module is serde, `Clone + Send + Sync + 'static`, with
//! no lifetimes and no `dyn` (asserted at compile time on every target). A
//! host that stores an [`EffectKind`] in a component re-dispatches it by
//! cloning it; a scene stores a [`HandlerDescriptor`] and re-binds a handle at
//! load.
//!
//! # Vocabulary
//!
//! The in-tree kinds are *transcriptions* of the six impl-side traits —
//! [`CompletionModel`](crate::completion::CompletionModel),
//! [`Tool`](crate::tool::Tool), [`EmbeddingModel`](crate::embeddings::EmbeddingModel)
//! (and its image twin), [`ConversationMemory`](crate::memory::ConversationMemory),
//! [`VectorStoreIndex`](crate::vector_store::VectorStoreIndex),
//! [`RerankModel`](crate::rerank::RerankModel) — one arm per method, with
//! the rules:
//!
//! - a generic result type parameter does not cross the wire:
//!   `VectorStoreIndex::top_n<T>` becomes [`RetrieveQuery::TopN`] answering
//!   JSON documents, and the typed view deserialises on the client side;
//! - `ConversationMemory` is three ops ([`MemoryOp`]); the `*_owned`
//!   conveniences are the adapter's business, not the wire's;
//! - no impl-side method takes `&mut self`, so no op needs an exclusive form;
//! - only [`EffectKind::Completion`] with `stream: true` streams; every other
//!   kind is unary.
//!
//! Out-of-tree kinds go through [`EffectKind::Custom`]. In-tree arms are added
//! as breaking changes: the enums are exhaustive on purpose, so every `match`
//! stays a complete census of the vocabulary.

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    completion::{CompletionRequest, CompletionResponse, Message, ModelRef, ProviderCapabilities},
    embeddings::{EmbeddingResponse, ImageEmbeddingResponse},
    error::ErrorReport,
    id::ConversationId,
    rerank::RerankResponse,
    streaming::StreamEvent,
    tool::ToolResult,
    vector_store::request::{Filter, VectorSearchRequest},
    wasm_compat::WasmCompatSend,
};

/// The identity of one dispatch, minted by the dispatcher.
///
/// It is the correlation key between a bus-tap [`EffectRecord`], a hook
/// observation, and a host's own bookkeeping (a Bevy driver maps it to an
/// `Entity`). Ids are unique per dispatcher and strictly increasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EffectId(u64);

impl EffectId {
    /// Build an id from its raw value (a replayer restoring a log does this).
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EffectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "effect:{}", self.0)
    }
}

/// Which registered handler serves an effect.
///
/// A key is a plain string: the builder generates them (`model`, `tool:add`),
/// hosts choose their own. It is the serde half of a handle — a scene stores
/// the key plus its [`HandlerDescriptor`] and re-binds at load.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HandlerKey(Arc<str>);

impl HandlerKey {
    /// Build a key from any string-like value.
    pub fn new(key: impl Into<Arc<str>>) -> Self {
        Self(key.into())
    }

    /// The key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The key read by its grammar: see [`KeyParts`]. Never fails — every
    /// string is some key — and round-trips through [`KeyParts::to_key`].
    pub fn parts(&self) -> KeyParts {
        KeyParts::parse(&self.0)
    }
}

/// A key's parts, by the grammar every key the tree mints follows:
///
/// ```text
/// [<owner>/]<name>[#<generation>]     where <name> is [<kind>:]<label>
/// ```
///
/// `golden/tool:add#2` is owner `golden`, kind `tool`, label `add`,
/// generation `2`; `host/note` is owner `host`, no kind, label `note`;
/// `model:fast` is no owner, kind `model`, label `fast`. The owner is the
/// text before the first `/`, the kind the text before the first `:` of
/// what follows, the generation the digits after the last `#`; a label
/// holds anything else, so a key whose label contains one of those
/// separators is read as its grammar says, not as its author meant. The
/// string is the wire form; this is how a reader takes it apart without
/// a hand parser of its own.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyParts {
    /// The program or host the key belongs to (`golden`, `host`).
    pub owner: Option<Arc<str>>,
    /// The family-shaped prefix of the name (`model`, `tool`, `retrieve`).
    pub kind: Option<Arc<str>>,
    /// The name proper: a model label, a tool name, `memory`, `note`.
    pub label: Arc<str>,
    /// The disambiguating generation (`#2`): the same tool registered
    /// again under a fresh key. Only canonical `u64` decimal suffixes are
    /// generations; leading zeroes and overflow remain literal label text.
    pub generation: Option<u64>,
}

impl KeyParts {
    /// Read `key` by the grammar.
    pub fn parse(key: &str) -> Self {
        let (owner, rest) = match key.split_once('/') {
            Some((owner, rest)) if !owner.is_empty() && !rest.is_empty() => {
                (Some(Arc::from(owner)), rest)
            }
            _ => (None, key),
        };
        let (rest, generation) = match rest.rsplit_once('#') {
            Some((head, digits))
                if !digits.is_empty()
                    && (digits.len() == 1 || !digits.starts_with('0'))
                    && digits.bytes().all(|b| b.is_ascii_digit()) =>
            {
                match digits.parse::<u64>() {
                    Ok(generation) => (head, Some(generation)),
                    Err(_) => (rest, None),
                }
            }
            _ => (rest, None),
        };
        let (kind, label) = match rest.split_once(':') {
            Some((kind, label)) if !kind.is_empty() && !label.is_empty() => {
                (Some(Arc::from(kind)), label)
            }
            _ => (None, rest),
        };
        Self {
            owner,
            kind,
            label: Arc::from(label),
            generation,
        }
    }

    /// The parts as the key they came from (`Display` writes the same).
    pub fn to_key(&self) -> HandlerKey {
        HandlerKey::from(self.to_string())
    }
}

impl fmt::Display for KeyParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(owner) = &self.owner {
            write!(f, "{owner}/")?;
        }
        if let Some(kind) = &self.kind {
            write!(f, "{kind}:")?;
        }
        f.write_str(&self.label)?;
        if let Some(generation) = self.generation {
            write!(f, "#{generation}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for HandlerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandlerKey({:?})", &*self.0)
    }
}

impl fmt::Display for HandlerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for HandlerKey {
    fn from(key: &str) -> Self {
        Self(Arc::from(key))
    }
}

impl From<String> for HandlerKey {
    fn from(key: String) -> Self {
        Self(Arc::from(key))
    }
}

impl AsRef<str> for HandlerKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// Transparent string (de)serialization without serde's `rc` feature — the
// same choice `ModelRef` makes; the two `Arc<str>` sites in this module use
// hand-written impls rather than enabling a feature on the dependency.
impl Serialize for HandlerKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for HandlerKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let key = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Ok(Self(Arc::from(&*key)))
    }
}

/// Serde for a bare `Arc<str>` field (the `Custom` kind label).
mod arc_str {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        value: &Arc<str>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<str>, D::Error> {
        let value = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Ok(Arc::from(&*value))
    }
}

/// `Option<Arc<str>>` as an optional string on the wire.
mod opt_arc_str {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        value: &Option<Arc<str>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&**value),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Arc<str>>, D::Error> {
        let value = <Option<std::borrow::Cow<'de, str>>>::deserialize(deserializer)?;
        Ok(value.map(|value| Arc::from(&*value)))
    }
}

/// The families of effect — the discriminant a typed view checks at bind
/// time and the label a log line prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectFamily {
    /// A completion request to a model.
    Completion,
    /// A tool call.
    Tool,
    /// A text or image embedding request.
    Embed,
    /// A reranking request.
    Rerank,
    /// A conversation-memory operation.
    Memory,
    /// A vector-store retrieval.
    Retrieve,
    /// An out-of-tree kind.
    Custom,
}

impl EffectFamily {
    /// The family's stable label.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Completion => "completion",
            Self::Tool => "tool_call",
            Self::Embed => "embed",
            Self::Rerank => "rerank",
            Self::Memory => "memory",
            Self::Retrieve => "retrieve",
            Self::Custom => "custom",
        }
    }
}

impl fmt::Display for EffectFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A type-level family marker: what a typed view (`Handle<F>`) is generic
/// over, and what it knows: the request a typed dispatch of the family
/// takes, the answer it resolves to, and how each maps onto the wire's
/// [`EffectKind`] and [`Outcome`]. Implemented by the unit types in
/// [`family`] and by [`family::Custom<E>`] for a host's [`CustomEffect`];
/// sealed — hosts define custom *effects*, never new families (the
/// transcription rule keeps the vocabulary to the six impl-side traits).
pub trait Family: sealed::Sealed + Clone + Copy + Send + Sync + 'static {
    /// The family this marker names.
    const FAMILY: EffectFamily;
    /// What a typed dispatch of this family takes.
    type Request: WasmCompatSend + 'static;
    /// What it resolves to.
    type Answer: WasmCompatSend + 'static;
    /// The wire form of a request, or the report for a request that has
    /// none (a [`CustomEffect`] whose `Serialize` fails). The in-tree
    /// families always have one; a typed dispatch of a request without one
    /// is pre-failed by the bus and never reaches a handler or a log.
    fn wrap(request: Self::Request) -> Result<EffectKind, ErrorReport>;
    /// The typed answer, or the report for an outcome of another family.
    fn unwrap(outcome: Outcome) -> Result<Self::Answer, ErrorReport>;
    /// The report [`Family::unwrap`] gives for an outcome of another family.
    fn mismatch(outcome: &Outcome) -> ErrorReport {
        ErrorReport::new(
            crate::error::ErrorKind::Internal,
            format!(
                "expected a {} outcome, the handler answered {}",
                Self::FAMILY,
                outcome.family()
            ),
        )
    }
}

mod sealed {
    pub trait Sealed {}
}

/// What a handler serves, as a type: a [`Family`] (`Some(F::FAMILY)`), or
/// [`family::Dynamic`] (`None`) for a handler that answers whatever it is
/// given — a replayer, an erased handler. A typed key can be proven only
/// against a handler with a family. Sealed.
pub trait Served: sealed::Sealed + 'static {
    /// The family, when the handler has one.
    const SERVED: Option<EffectFamily>;
}

// `Served` is sealed; a handler's family is a `Family` marker or `Dynamic`,
// and an error about it should say so rather than suggest this impl.
#[diagnostic::do_not_recommend]
impl<F: Family> Served for F {
    const SERVED: Option<EffectFamily> = Some(F::FAMILY);
}

/// A tool call as a typed request: the name and the raw JSON arguments.
/// The context the tool runs with is not part of the request: it travels
/// beside the dispatch, attached by the driver to the handler's sink
/// (`OutcomeSink::scope::<ToolContext>()`), and what the tool publishes
/// comes back the same way ([`crate::tool::PublishedContext`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// The tool's name (the name the model calls it by).
    pub name: String,
    /// The arguments as a JSON string.
    pub args: String,
}

/// A host's own effect, typed the way a [`ToolContext`](crate::tool::ToolContext) value is: a
/// declared kind label and a declared answer type, both serde. The wire
/// form is [`EffectKind::Custom`] / [`Outcome::Custom`]; the type never
/// crosses it.
pub trait CustomEffect: Serialize + serde::de::DeserializeOwned + WasmCompatSend + 'static {
    /// The kind label this effect dispatches under; a handler's
    /// [`FamilyDescriptor::Custom`] must name the same label.
    const KIND: &'static str;
    /// What the handler answers.
    type Answer: Serialize + serde::de::DeserializeOwned + WasmCompatSend + 'static;
}

mod key;

pub use key::Key;

/// A key for a model handler: `model:<label>`.
pub fn model_key(label: &str) -> HandlerKey {
    HandlerKey::from(format!("model:{label}"))
}

/// A key for a tool handler: `tool:<name>`.
pub fn tool_key(name: &str) -> HandlerKey {
    HandlerKey::from(format!("tool:{name}"))
}

/// The family markers.
pub mod family {
    use std::marker::PhantomData;

    use super::{
        CustomEffect, EffectFamily, EffectKind, EmbedInputs, EmbedOutputs, Family, MemoryOp,
        MemoryOutcome, Outcome, RerankRequest, RetrieveQuery, RetrievedDocuments, ToolCallRequest,
        sealed::Sealed,
    };
    use crate::{
        completion::{CompletionRequest, CompletionResponse},
        error::{ErrorKind, ErrorReport},
        rerank::RerankResponse,
        tool::ToolResult,
    };

    macro_rules! marker {
        ($($(#[$doc:meta])* $name:ident => $family:ident, $request:ty, $answer:ty, $wrap:expr, $unwrap:expr;)+) => {$(
            $(#[$doc])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
            pub struct $name;

            impl Sealed for $name {}

            impl Family for $name {
                const FAMILY: EffectFamily = EffectFamily::$family;
                type Request = $request;
                type Answer = $answer;

                fn wrap(request: Self::Request) -> Result<EffectKind, ErrorReport> {
                    let wrap: fn(Self::Request) -> EffectKind = $wrap;
                    Ok(wrap(request))
                }

                fn unwrap(outcome: Outcome) -> Result<Self::Answer, ErrorReport> {
                    let unwrap: fn(Outcome) -> Result<Self::Answer, ErrorReport> = $unwrap;
                    unwrap(outcome)
                }
            }
        )+};
    }

    marker! {
        /// The completion family: a unary completion (a streaming dispatch is
        /// `ModelHandle::stream`, not a typed request).
        Completion => Completion, CompletionRequest, CompletionResponse,
            |request| EffectKind::Completion { request, stream: false },
            |outcome| match outcome {
                Outcome::Completion(response) => Ok(response),
                other => Err(Completion::mismatch(&other)),
            };
        /// The tool family.
        Tool => Tool, ToolCallRequest, ToolResult,
            |request| EffectKind::ToolCall { name: request.name, args: request.args },
            |outcome| match outcome {
                Outcome::ToolResult { result } => Ok(result),
                other => Err(Tool::mismatch(&other)),
            };
        /// The embedding family.
        Embed => Embed, EmbedInputs, EmbedOutputs,
            |inputs| EffectKind::Embed { inputs },
            |outcome| match outcome {
                Outcome::Embeddings(outputs) => Ok(outputs),
                other => Err(Embed::mismatch(&other)),
            };
        /// The reranking family.
        Rerank => Rerank, RerankRequest, RerankResponse,
            |request| EffectKind::Rerank { request },
            |outcome| match outcome {
                Outcome::Reranked(response) => Ok(response),
                other => Err(Rerank::mismatch(&other)),
            };
        /// The conversation-memory family.
        Memory => Memory, MemoryOp, MemoryOutcome,
            |op| EffectKind::Memory { op },
            |outcome| match outcome {
                Outcome::Memory(answer) => Ok(answer),
                other => Err(Memory::mismatch(&other)),
            };
        /// The retrieval family.
        Retrieve => Retrieve, RetrieveQuery, RetrievedDocuments,
            |query| EffectKind::Retrieve { query },
            |outcome| match outcome {
                Outcome::Documents(documents) => Ok(documents),
                other => Err(Retrieve::mismatch(&other)),
            };
    }

    /// A handler with no one family: a replayer answering whatever its log
    /// holds, or an erased handler forwarding to whatever it wraps.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct Dynamic;

    impl Sealed for Dynamic {}

    impl super::Served for Dynamic {
        const SERVED: Option<EffectFamily> = None;
    }

    /// The family of one host-defined effect `E`: dispatches
    /// [`EffectKind::Custom`] under `E::KIND` and answers `E::Answer`.
    pub struct Custom<E: CustomEffect>(PhantomData<fn() -> E>);

    impl<E: CustomEffect> Custom<E> {
        /// The marker.
        pub const fn new() -> Self {
            Self(PhantomData)
        }
    }

    // Written by hand: a derive would demand `E: Clone` (and friends), and
    // the marker must be `Copy` for every `E`.
    impl<E: CustomEffect> Clone for Custom<E> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: CustomEffect> Copy for Custom<E> {}
    impl<E: CustomEffect> Default for Custom<E> {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<E: CustomEffect> std::fmt::Debug for Custom<E> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Custom<{}>", E::KIND)
        }
    }
    impl<E: CustomEffect> PartialEq for Custom<E> {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }
    impl<E: CustomEffect> Eq for Custom<E> {}
    impl<E: CustomEffect> std::hash::Hash for Custom<E> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            E::KIND.hash(state);
        }
    }

    impl<E: CustomEffect> Sealed for Custom<E> {}

    impl<E: CustomEffect> Family for Custom<E> {
        const FAMILY: EffectFamily = EffectFamily::Custom;
        type Request = E;
        type Answer = E::Answer;

        fn wrap(request: E) -> Result<EffectKind, ErrorReport> {
            // An effect that does not serialize is a defect in `E`: the
            // dispatch is refused with the serde error as its report, so no
            // handler serves and no log records a request that had no
            // wire form.
            serde_json::to_value(&request)
                .map(|payload| EffectKind::Custom {
                    kind: std::sync::Arc::from(E::KIND),
                    payload,
                })
                .map_err(|error| {
                    ErrorReport::new(
                        ErrorKind::Request,
                        format!("the `{}` effect did not serialize: {error}", E::KIND),
                    )
                })
        }

        fn unwrap(outcome: Outcome) -> Result<E::Answer, ErrorReport> {
            match outcome {
                Outcome::Custom { payload: value } => {
                    serde_json::from_value(value).map_err(|error| {
                        ErrorReport::new(
                            ErrorKind::Internal,
                            format!(
                                "the answer to the `{}` effect did not deserialize: {error}",
                                E::KIND
                            ),
                        )
                    })
                }
                other => Err(Self::mismatch(&other)),
            }
        }
    }
}

/// What a registered handler is: its key and its family-specific description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandlerDescriptor {
    /// The key the handler is registered under.
    pub key: HandlerKey,
    /// The family and its advertised metadata.
    pub family: FamilyDescriptor,
    /// The layers wrapped around the handler, outermost first: each
    /// [`Intercept`](crate::serve::Intercept)'s name. Part of what serves
    /// the key — a log's handler table names them, and a program recorded
    /// under one layer stack refuses a replay under another, as it refuses
    /// another hook stack. Empty for a bare handler.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<String>,
}

/// The family-keyed description of a handler. The variant *is* the family:
/// binding a typed view compares [`Family::FAMILY`] against
/// [`FamilyDescriptor::family`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum FamilyDescriptor {
    /// A completion model.
    Completion {
        /// The model's label.
        model: ModelRef,
        /// The capability snapshot a runtime prepares requests against.
        capabilities: ProviderCapabilities,
    },
    /// A tool.
    Tool {
        /// The tool's name (the name the model calls it by).
        name: String,
        /// The tool's description.
        description: String,
        /// The JSON schema of the tool's arguments.
        parameters: serde_json::Value,
        /// Present when the tool is retrievable by embedding.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        embedding: Option<ToolEmbeddingDescriptor>,
    },
    /// An embedding model.
    Embed {
        /// The model's label.
        model: String,
        /// The vector dimension, when the model declares one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dims: Option<usize>,
        /// The largest batch the model accepts.
        max_documents: usize,
        /// Whether the model embeds text or images.
        modality: EmbedModality,
    },
    /// A reranking model.
    Rerank {
        /// The model's label.
        model: String,
        /// The largest batch the model accepts.
        max_documents: usize,
    },
    /// A conversation-memory backend.
    Memory {},
    /// A vector-store index.
    Retrieve {},
    /// A handler for an out-of-tree kind: the label it serves.
    Custom {
        /// The [`EffectKind::Custom`] kind label this handler answers.
        kind: String,
    },
}

impl FamilyDescriptor {
    /// The family this descriptor belongs to.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion { .. } => EffectFamily::Completion,
            Self::Tool { .. } => EffectFamily::Tool,
            Self::Embed { .. } => EffectFamily::Embed,
            Self::Rerank { .. } => EffectFamily::Rerank,
            Self::Memory {} => EffectFamily::Memory,
            Self::Retrieve {} => EffectFamily::Retrieve,
            Self::Custom { .. } => EffectFamily::Custom,
        }
    }
}

/// The embedding context of a retrievable tool: what
/// [`ToolEmbedding`](crate::tool::ToolEmbedding) advertises, as data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolEmbeddingDescriptor {
    /// The tool's serialized context.
    pub context: serde_json::Value,
    /// The documents the tool is retrieved by.
    pub embedding_docs: Vec<String>,
}

/// Which modality an embedding handler serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedModality {
    /// Text documents.
    Text,
    /// Image bytes.
    Image,
}

/// One effect: what a handler is asked to do.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "a completion request is the common case and is moved, not copied; boxing it would put an allocation on every dispatch"
)]
pub enum EffectKind {
    /// A completion request.
    Completion {
        /// The prepared request.
        request: CompletionRequest,
        /// Whether the response streams (`dispatch_stream`) or is unary.
        stream: bool,
    },
    /// A tool call. The context the tool runs with is not on the wire:
    /// see [`ToolCallRequest`].
    ToolCall {
        /// The tool's name.
        name: String,
        /// The raw JSON arguments as the model produced them.
        args: String,
    },
    /// An embedding request.
    Embed {
        /// The inputs to embed.
        inputs: EmbedInputs,
    },
    /// A reranking request.
    Rerank {
        /// The query and the documents.
        request: RerankRequest,
    },
    /// A conversation-memory operation.
    Memory {
        /// The operation.
        op: MemoryOp,
    },
    /// A retrieval.
    Retrieve {
        /// The query.
        query: RetrieveQuery,
    },
    /// An out-of-tree effect.
    Custom {
        /// The host-defined kind label.
        #[serde(with = "arc_str")]
        kind: Arc<str>,
        /// The host-defined payload.
        payload: serde_json::Value,
    },
}

impl EffectKind {
    /// The family of this effect.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion { .. } => EffectFamily::Completion,
            Self::ToolCall { .. } => EffectFamily::Tool,
            Self::Embed { .. } => EffectFamily::Embed,
            Self::Rerank { .. } => EffectFamily::Rerank,
            Self::Memory { .. } => EffectFamily::Memory,
            Self::Retrieve { .. } => EffectFamily::Retrieve,
            Self::Custom { .. } => EffectFamily::Custom,
        }
    }

    /// A stable label for logs and overlays — never the payload. `Custom`
    /// returns its own kind label.
    pub fn name(&self) -> &str {
        match self {
            Self::Custom { kind, .. } => kind,
            Self::Completion { .. }
            | Self::ToolCall { .. }
            | Self::Embed { .. }
            | Self::Rerank { .. }
            | Self::Memory { .. }
            | Self::Retrieve { .. } => self.family().name(),
        }
    }

    /// Whether this effect answers as a stream. Only a streaming completion
    /// does; every other kind is unary.
    pub const fn streams(&self) -> bool {
        match self {
            Self::Completion { stream, .. } => *stream,
            Self::ToolCall { .. }
            | Self::Embed { .. }
            | Self::Rerank { .. }
            | Self::Memory { .. }
            | Self::Retrieve { .. }
            | Self::Custom { .. } => false,
        }
    }
}

/// The inputs of an embedding request: the transcription of
/// `EmbeddingModel::embed_texts` and `ImageEmbeddingModel::embed_images`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "modality", content = "inputs", rename_all = "snake_case")]
pub enum EmbedInputs {
    /// Text documents.
    Texts(Vec<String>),
    /// Image bytes.
    Images(Vec<Vec<u8>>),
}

/// A reranking request: the transcription of
/// [`RerankModel::rerank`](crate::rerank::RerankModel::rerank).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankRequest {
    /// The query the documents are ranked against.
    pub query: String,
    /// The documents, in input order.
    pub documents: Vec<String>,
}

/// A conversation-memory operation: the transcription of
/// [`ConversationMemory`](crate::memory::ConversationMemory).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MemoryOp {
    /// Load a conversation's history.
    Load {
        /// The conversation.
        conversation: ConversationId,
    },
    /// Append messages to a conversation.
    Append {
        /// The conversation.
        conversation: ConversationId,
        /// The messages, in order.
        messages: Vec<Message>,
    },
    /// Clear a conversation.
    Clear {
        /// The conversation.
        conversation: ConversationId,
    },
}

/// A retrieval: the transcription of
/// [`VectorStoreIndex`](crate::vector_store::VectorStoreIndex) over the
/// dynamic filter. The typed result parameter of `top_n<T>` stays on the
/// client side.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "query", rename_all = "snake_case")]
pub enum RetrieveQuery {
    /// Scored documents.
    TopN {
        /// The search.
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    },
    /// Scored ids only.
    TopNIds {
        /// The search.
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    },
}

/// What a handler answered.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "a completion response is the common case and is moved, not copied"
)]
pub enum Outcome {
    /// A unary completion.
    Completion(CompletionResponse),
    /// A tool call's result. The values the tool published into its
    /// context are not on the wire: they come back beside the sink
    /// ([`crate::tool::PublishedContext`]).
    ToolResult {
        /// The result.
        result: ToolResult,
    },
    /// Embeddings.
    Embeddings(EmbedOutputs),
    /// A reranking.
    Reranked(RerankResponse),
    /// A memory operation's answer.
    Memory(MemoryOutcome),
    /// Retrieved documents.
    Documents(RetrievedDocuments),
    /// An out-of-tree answer.
    Custom {
        /// The answer, nested so every JSON shape is representable.
        payload: serde_json::Value,
    },
}

impl Outcome {
    /// The family this outcome answers.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion(_) => EffectFamily::Completion,
            Self::ToolResult { .. } => EffectFamily::Tool,
            Self::Embeddings(_) => EffectFamily::Embed,
            Self::Reranked(_) => EffectFamily::Rerank,
            Self::Memory(_) => EffectFamily::Memory,
            Self::Documents(_) => EffectFamily::Retrieve,
            Self::Custom { .. } => EffectFamily::Custom,
        }
    }
}

/// The answer to an [`EmbedInputs`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modality", content = "response", rename_all = "snake_case")]
pub enum EmbedOutputs {
    /// Text embeddings.
    Texts(EmbeddingResponse),
    /// Image embeddings.
    Images(ImageEmbeddingResponse),
}

/// The answer to a [`MemoryOp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "memory", rename_all = "snake_case")]
pub enum MemoryOutcome {
    /// The loaded history.
    Loaded {
        /// The messages.
        messages: Vec<Message>,
    },
    /// The append succeeded.
    Appended,
    /// The clear succeeded.
    Cleared,
}

/// The answer to a [`RetrieveQuery`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "retrieved", content = "results", rename_all = "snake_case")]
pub enum RetrievedDocuments {
    /// Scored documents: `(score, id, document)`.
    Scored(Vec<(f64, String, serde_json::Value)>),
    /// Scored ids: `(score, id)`.
    Ids(Vec<(f64, String)>),
}

/// One delivery observed by a scheduled consumer. Entries are ordered by
/// visibility, independently of dispatch ids and handler completion order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delivery {
    /// The schedule pass in which the consumer observed this delivery.
    /// Pass numbers group deliveries; they are not wall-clock timestamps.
    pub batch: u64,
    /// The effect whose state became visible.
    pub id: EffectId,
    /// The visible transition.
    pub kind: DeliveryKind,
}

/// What a scheduled consumer collected at one observation boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "delivery", rename_all = "snake_case")]
pub enum DeliveryKind {
    /// The effect's outcome became visible.
    Outcome,
    /// A batch of stream items became visible together. Includes error items;
    /// successful event bytes remain in the effect record's event sequence.
    Stream {
        /// Items collected since the preceding delivery for this effect.
        items: usize,
    },
}

/// One recorded exchange: the effect, who served it, and the answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRecord {
    /// Explicitly published tool-result metadata, including on errors. Never
    /// inbound authentication values or runtime scopes. `None` means the tool
    /// did not publish a context before resolving. The wire field is required:
    /// `null` explicitly records no publication; omission cannot establish
    /// whether an older recorder lost output and is rejected when decoding.
    #[serde(deserialize_with = "Option::deserialize")]
    pub tool_output: Option<crate::tool::ToolResultContext>,
    /// The dispatch's id.
    pub id: EffectId,
    /// The handler the effect was routed to.
    pub key: HandlerKey,
    /// The effect.
    pub kind: EffectKind,
    /// The answer.
    pub outcome: Result<Outcome, ErrorReport>,
    /// A streamed dispatch's events, verbatim, when the recorder was asked
    /// to keep them (`EffectLogRecorder::keeping_stream_events`); `None`
    /// otherwise, and the answer is the fold. A replayer re-emits these
    /// when present, so a replayed consumer sees the original delta
    /// boundaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<Vec<StreamEvent>>,
    /// The dispatch this one was made from — a handler serving `parent`
    /// dispatched it through its sink's dispatcher — or `None` for a
    /// dispatch the consumer made itself. Causality as data: a host parents
    /// the effect's entity by it, a replay reads the chain off the record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<EffectId>,
    /// The scope of the program that made the dispatch: a stable serde id
    /// of the dispatching run or agent — never a runtime handle — stamped
    /// by a scoped dispatcher, so one log written by several programs in
    /// one world can be read per program. `None` when the dispatcher had
    /// no scope, which is every record today.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_arc_str")]
    pub scope: Option<std::sync::Arc<str>>,
}

/// The effect row of a program: which keys it can dispatch to, and of
/// which family — the effect type of the program, as a row of
/// `key: family` entries (Leijen's effect rows, with the key as the
/// label). One type, three readers: the log's header stores the program's
/// row and the trace's signature, the agent checks its row against a
/// log's handler table on replay, and a host checks a scene's row against
/// its bus at startup — all with [`EffectRow::is_subset_of`], which names
/// the first gap.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EffectRow(std::collections::BTreeMap<HandlerKey, EffectFamily>);

/// The first entry of a row a set of handlers does not serve: the key,
/// the family the row needs, and what serves the key instead, if anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGap {
    /// The key the row names.
    pub key: HandlerKey,
    /// The family the row needs it as.
    pub needed: EffectFamily,
    /// The family a handler serves it as, or `None` when nothing serves it.
    pub served: Option<EffectFamily>,
}

impl std::fmt::Display for RowGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.served {
            Some(served) => write!(
                f,
                "`{}` is needed as {} but served as {served}",
                self.key, self.needed
            ),
            None => write!(f, "`{}` ({}) is not served", self.key, self.needed),
        }
    }
}

/// One difference between two rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowDiff {
    /// The other row lacks this entry.
    Missing {
        /// The key this row names.
        key: HandlerKey,
        /// The family this row names it as.
        family: EffectFamily,
    },
    /// The other row has an entry this one lacks.
    Extra {
        /// The key the other row names.
        key: HandlerKey,
        /// The family the other row names it as.
        family: EffectFamily,
    },
    /// Both rows name the key, as different families.
    Family {
        /// The key both name.
        key: HandlerKey,
        /// This row's family for it.
        this: EffectFamily,
        other: EffectFamily,
    },
}

impl std::fmt::Display for RowDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { key, family } => write!(f, "`{key}` ({family}) is missing"),
            Self::Extra { key, family } => write!(f, "`{key}` ({family}) is extra"),
            Self::Family { key, this, other } => {
                write!(f, "`{key}` is {this} here and {other} there")
            }
        }
    }
}

impl EffectRow {
    /// An empty row.
    pub fn new() -> Self {
        Self::default()
    }

    /// Name `key` as `family`; a key already named is re-named.
    pub fn insert(&mut self, key: HandlerKey, family: EffectFamily) -> Option<EffectFamily> {
        self.0.insert(key, family)
    }

    /// Name `key` as `family` unless the row already names it.
    pub fn insert_if_absent(&mut self, key: HandlerKey, family: EffectFamily) {
        self.0.entry(key).or_insert(family);
    }

    /// Remove `key` from the row; the family it named, if it was there.
    pub fn remove(&mut self, key: &HandlerKey) -> Option<EffectFamily> {
        self.0.remove(key)
    }

    /// The family the row names `key` as.
    pub fn get(&self, key: &HandlerKey) -> Option<&EffectFamily> {
        self.0.get(key)
    }

    /// Whether the row names `key`.
    pub fn contains_key(&self, key: &HandlerKey) -> bool {
        self.0.contains_key(key)
    }

    /// The keys, in order.
    pub fn keys(&self) -> impl Iterator<Item = &HandlerKey> {
        self.0.keys()
    }

    /// The entries, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&HandlerKey, &EffectFamily)> {
        self.0.iter()
    }

    /// How many keys the row names.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the row names no key.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether every entry of this row is served by `handlers` as the
    /// family the row needs: `Ok` when it is, else the first gap in key
    /// order — a key nothing serves, or one served as another family.
    pub fn is_subset_of(&self, handlers: &[HandlerDescriptor]) -> Result<(), RowGap> {
        for (key, needed) in &self.0 {
            let served = handlers
                .iter()
                .find(|descriptor| &descriptor.key == key)
                .map(|descriptor| descriptor.family.family());
            match served {
                Some(served) if served == *needed => {}
                served => {
                    return Err(RowGap {
                        key: key.clone(),
                        needed: *needed,
                        served,
                    });
                }
            }
        }
        Ok(())
    }

    /// Every difference between this row and `other`, in key order:
    /// entries `other` lacks, entries it has and this row lacks, and keys
    /// both name as different families.
    pub fn diff(&self, other: &EffectRow) -> Vec<RowDiff> {
        let mut diffs = Vec::new();
        for (key, family) in &self.0 {
            match other.0.get(key) {
                None => diffs.push(RowDiff::Missing {
                    key: key.clone(),
                    family: *family,
                }),
                Some(theirs) if theirs != family => diffs.push(RowDiff::Family {
                    key: key.clone(),
                    this: *family,
                    other: *theirs,
                }),
                Some(_) => {}
            }
        }
        for (key, family) in &other.0 {
            if !self.0.contains_key(key) {
                diffs.push(RowDiff::Extra {
                    key: key.clone(),
                    family: *family,
                });
            }
        }
        diffs
    }
}

impl FromIterator<(HandlerKey, EffectFamily)> for EffectRow {
    fn from_iter<I: IntoIterator<Item = (HandlerKey, EffectFamily)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a EffectRow {
    type Item = (&'a HandlerKey, &'a EffectFamily);
    type IntoIter = std::collections::btree_map::Iter<'a, HandlerKey, EffectFamily>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for EffectRow {
    type Item = (HandlerKey, EffectFamily);
    type IntoIter = std::collections::btree_map::IntoIter<HandlerKey, EffectFamily>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// The protocol crosses threads and serializes on every target.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<EffectId>();
    assert_wire::<HandlerKey>();
    assert_wire::<EffectFamily>();
    assert_wire::<HandlerDescriptor>();
    assert_wire::<FamilyDescriptor>();
    assert_wire::<ToolEmbeddingDescriptor>();
    assert_wire::<EmbedModality>();
    assert_wire::<EffectKind>();
    assert_wire::<EmbedInputs>();
    assert_wire::<RerankRequest>();
    assert_wire::<MemoryOp>();
    assert_wire::<RetrieveQuery>();
    assert_wire::<Outcome>();
    assert_wire::<EmbedOutputs>();
    assert_wire::<MemoryOutcome>();
    assert_wire::<RetrievedDocuments>();
    assert_wire::<EffectRecord>();
    assert_wire::<EffectRow>();
    assert_wire::<StreamEvent>();
};

#[cfg(test)]
mod tests;
