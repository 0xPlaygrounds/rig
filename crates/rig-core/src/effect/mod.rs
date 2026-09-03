//! The effect protocol as data.
//!
//! An *effect* is one request an agent (or any host) makes of the outside
//! world — a completion, a tool call, an embedding, a conversation-memory
//! operation, a retrieval — expressed as a value rather than as a call on a
//! trait object. The bus ([`crate::bus`]) carries these values to the
//! handler registered for a [`HandlerKey`] and carries the [`Outcome`] back;
//! the [`EffectLog`] records every exchange so a run can be replayed.
//!
//! Everything in this module is serde, `Clone + Send + Sync + 'static`, with
//! no lifetimes and no `dyn` (asserted at compile time on every target). A
//! host that stores an [`EffectKind`] in a component re-dispatches it by
//! cloning it; a scene stores a [`HandlerDescriptor`] and re-binds a handle at
//! load.
//!
//! # Vocabulary
//!
//! The in-tree kinds are *transcriptions* of the five impl-side traits —
//! [`CompletionModel`](crate::completion::CompletionModel),
//! [`Tool`](crate::tool::Tool), [`EmbeddingModel`](crate::embeddings::EmbeddingModel)
//! (and its image twin), [`ConversationMemory`](crate::memory::ConversationMemory),
//! [`VectorStoreIndex`](crate::vector_store::VectorStoreIndex) — one arm per
//! method, with the rules:
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
    streaming::StreamEvent,
    tool::{ToolContext, ToolResult},
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
/// transcription rule keeps the vocabulary to the five impl-side traits).
pub trait Family: sealed::Sealed + Clone + Copy + Send + Sync + 'static {
    /// The family this marker names.
    const FAMILY: EffectFamily;
    /// What a typed dispatch of this family takes.
    type Request: WasmCompatSend + 'static;
    /// What it resolves to.
    type Answer: WasmCompatSend + 'static;
    /// The wire form of a request.
    fn wrap(request: Self::Request) -> EffectKind;
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

/// A tool call as a typed request: the raw JSON arguments and the
/// dispatch-scoped context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// The tool's name (the name the model calls it by).
    pub name: String,
    /// The arguments as a JSON string.
    pub args: String,
    /// The context the tool runs with.
    pub context: ToolContext,
}

/// A tool call's typed answer: the result and the context the tool
/// published.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAnswer {
    /// The result.
    pub result: ToolResult,
    /// The dispatch context after the tool ran.
    pub context: ToolContext,
}

/// A host's own effect, typed the way a [`ToolContext`] value is: a
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

/// The family markers.
pub mod family {
    use std::marker::PhantomData;

    use super::{
        CustomEffect, EffectFamily, EffectKind, EmbedInputs, EmbedOutputs, Family, MemoryOp,
        MemoryOutcome, Outcome, RetrieveQuery, RetrievedDocuments, ToolAnswer, ToolCallRequest,
        sealed::Sealed,
    };
    use crate::{
        completion::{CompletionRequest, CompletionResponse},
        error::{ErrorKind, ErrorReport},
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

                fn wrap(request: Self::Request) -> EffectKind {
                    let wrap: fn(Self::Request) -> EffectKind = $wrap;
                    wrap(request)
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
        Tool => Tool, ToolCallRequest, ToolAnswer,
            |request| EffectKind::ToolCall { name: request.name, args: request.args, context: request.context },
            |outcome| match outcome {
                Outcome::ToolResult { result, context } => Ok(ToolAnswer { result, context }),
                other => Err(Tool::mismatch(&other)),
            };
        /// The embedding family.
        Embed => Embed, EmbedInputs, EmbedOutputs,
            |inputs| EffectKind::Embed { inputs },
            |outcome| match outcome {
                Outcome::Embeddings(outputs) => Ok(outputs),
                other => Err(Embed::mismatch(&other)),
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

        fn wrap(request: E) -> EffectKind {
            match serde_json::to_value(&request) {
                Ok(payload) => EffectKind::Custom {
                    kind: std::sync::Arc::from(E::KIND),
                    payload,
                },
                // An effect that does not serialize is a defect in `E`; the
                // dispatch carries the error as its payload so the handler
                // (and the log) see it rather than a silent `null`.
                Err(error) => EffectKind::Custom {
                    kind: std::sync::Arc::from(E::KIND),
                    payload: serde_json::json!({ "error": error.to_string() }),
                },
            }
        }

        fn unwrap(outcome: Outcome) -> Result<E::Answer, ErrorReport> {
            match outcome {
                Outcome::Custom(value) => serde_json::from_value(value).map_err(|error| {
                    ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "the answer to the `{}` effect did not deserialize: {error}",
                            E::KIND
                        ),
                    )
                }),
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
    /// A tool call.
    ToolCall {
        /// The tool's name.
        name: String,
        /// The raw JSON arguments as the model produced them.
        args: String,
        /// The dispatch-scoped tool context.
        context: ToolContext,
    },
    /// An embedding request.
    Embed {
        /// The inputs to embed.
        inputs: EmbedInputs,
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
    /// A tool call's result and the context it published.
    ToolResult {
        /// The result.
        result: ToolResult,
        /// The dispatch context after the tool ran.
        context: ToolContext,
    },
    /// Embeddings.
    Embeddings(EmbedOutputs),
    /// A memory operation's answer.
    Memory(MemoryOutcome),
    /// Retrieved documents.
    Documents(RetrievedDocuments),
    /// An out-of-tree answer.
    Custom(serde_json::Value),
}

impl Outcome {
    /// The family this outcome answers.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion(_) => EffectFamily::Completion,
            Self::ToolResult { .. } => EffectFamily::Tool,
            Self::Embeddings(_) => EffectFamily::Embed,
            Self::Memory(_) => EffectFamily::Memory,
            Self::Documents(_) => EffectFamily::Retrieve,
            Self::Custom(_) => EffectFamily::Custom,
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

/// One recorded exchange: the effect, who served it, and the answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRecord {
    /// The dispatch's id.
    pub id: EffectId,
    /// The handler the effect was routed to.
    pub key: HandlerKey,
    /// The effect.
    pub kind: EffectKind,
    /// The answer.
    pub outcome: Result<Outcome, ErrorReport>,
}

/// A recorded run: every exchange in dispatch order.
pub type EffectLog = Vec<EffectRecord>;

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
    assert_wire::<MemoryOp>();
    assert_wire::<RetrieveQuery>();
    assert_wire::<Outcome>();
    assert_wire::<EmbedOutputs>();
    assert_wire::<MemoryOutcome>();
    assert_wire::<RetrievedDocuments>();
    assert_wire::<EffectRecord>();
    assert_wire::<EffectLog>();
    assert_wire::<StreamEvent>();
};

#[cfg(test)]
mod tests;
