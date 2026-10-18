//! Serializable effect requests, handler descriptions, outcomes, and recording data.
//! [`HandlerKey`] identifies a destination; [`EffectKind`] and [`Answered`] describe
//! the exchange. Host-defined operations use [`CustomEffect`].
//!
//! ```
//! use rig_core::effect::{EffectFamily, EffectRow, model_key};
//!
//! let mut row = EffectRow::new();
//! row.insert(model_key("primary"), EffectFamily::Completion);
//! assert_eq!(row.len(), 1);
//! ```

use std::{borrow::Cow, fmt, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    completion::{CompletionRequest, CompletionResponse, ModelRef, ProviderCapabilities},
    embeddings::EmbeddingResponse,
    error::ErrorReport,
    operation::RerankRequest,
    rerank::RerankResponse,
    streaming::StreamRecord,
    tool::ToolResult,
    vector_store::request::{Filter, VectorSearchRequest},
    wasm_compat::WasmCompatSend,
};

/// The identity of one dispatch, minted by the dispatcher.
///
/// It links an [`EffectRecord`] to a host's own bookkeeping (a host driver
/// maps it to its own handle). Ids are unique per dispatcher and strictly increasing.
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

/// String identifier for a registered handler, serializable without a live handle.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
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

    /// Parses the key into [`KeyParts`] without failure. Formatting the parts
    /// reproduces the original key.
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
    /// The name proper: a model label, a tool name, `retrieve`, `note`.
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

/// Effect classification used by typed binding and recording.
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

/// Sealed mapping between typed requests and answers and the wire's
/// [`EffectKind`] and [`Answered`]. Hosts extend the protocol through
/// [`CustomEffect`] and [`family::Custom`], not new family implementations.
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
    /// is pre-failed and never reaches a handler or a log.
    fn wrap(request: Self::Request) -> Result<EffectKind, ErrorReport>;
    /// The typed answer, or the report for an outcome of another family.
    fn unwrap(outcome: Answered) -> Result<Self::Answer, ErrorReport>;
    /// The report [`Family::unwrap`] gives for an outcome of another family.
    fn mismatch(outcome: &Answered) -> ErrorReport {
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

/// Sealed handler-family declaration. Typed families expose `Some(F::FAMILY)`;
/// [`family::Dynamic`] exposes `None` and requires runtime family checks.
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
/// (`Dispatch::scope::<ToolContext>()`), and what the tool publishes
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
/// form is [`EffectKind::Custom`] / [`Answered::Custom`]; the type never
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

/// A key for an embedding handler: `embed:<label>`.
pub fn embed_key(label: &str) -> HandlerKey {
    HandlerKey::from(format!("embed:{label}"))
}

/// A key for a rerank handler: `rerank:<label>`.
pub fn rerank_key(label: &str) -> HandlerKey {
    HandlerKey::from(format!("rerank:{label}"))
}

/// A key for a labelled retrieval handler: `retrieve:<label>`. An agent's
/// own index is registered under the bare `retrieve` key.
pub fn retrieve_key(label: &str) -> HandlerKey {
    HandlerKey::from(format!("retrieve:{label}"))
}

/// Type-level markers for built-in and host-defined effect families.
///
/// ```
/// use rig_core::effect::{family, Family, EffectFamily};
/// assert_eq!(family::Completion::FAMILY, EffectFamily::Completion);
/// ```
pub mod family {
    use std::marker::PhantomData;

    use super::{
        Answered, CustomEffect, EffectFamily, EffectKind, EmbedInputs, EmbedOutputs, Family,
        RetrieveQuery, RetrievedDocuments, ToolCallRequest, sealed::Sealed,
    };
    use crate::{
        completion::{CompletionRequest, CompletionResponse},
        error::{ErrorKind, ErrorReport},
        operation::RerankRequest,
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

                fn unwrap(outcome: Answered) -> Result<Self::Answer, ErrorReport> {
                    let unwrap: fn(Answered) -> Result<Self::Answer, ErrorReport> = $unwrap;
                    unwrap(outcome)
                }
            }
        )+};
    }

    marker! {
        /// The completion family: a unary completion (a streaming dispatch is
        /// `ModelHandle::stream`, not a typed request).
        Completion => Completion, CompletionRequest, CompletionResponse,
            |request| EffectKind::Completion { request: std::sync::Arc::new(request), stream: false },
            |outcome| match outcome {
                Answered::Completion(response) => Ok(std::sync::Arc::unwrap_or_clone(response)),
                other => Err(Completion::mismatch(&other)),
            };
        /// The tool family.
        Tool => Tool, ToolCallRequest, ToolResult,
            |request| EffectKind::ToolCall { name: request.name, args: request.args },
            |outcome| match outcome {
                Answered::ToolResult { result } => Ok(result),
                other => Err(Tool::mismatch(&other)),
            };
        /// The embedding family.
        Embed => Embed, EmbedInputs, EmbedOutputs,
            |inputs| EffectKind::Embed { inputs },
            |outcome| match outcome {
                Answered::Embeddings(outputs) => Ok(outputs),
                other => Err(Embed::mismatch(&other)),
            };
        /// The reranking family.
        Rerank => Rerank, RerankRequest, RerankResponse,
            |request| EffectKind::Rerank { request },
            |outcome| match outcome {
                Answered::Reranked(response) => Ok(response),
                other => Err(Rerank::mismatch(&other)),
            };
        /// The retrieval family.
        Retrieve => Retrieve, RetrieveQuery, RetrievedDocuments,
            |query| EffectKind::Retrieve { query },
            |outcome| match outcome {
                Answered::Documents(documents) => Ok(documents),
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
            // Reject unencodable requests before dispatch so no handler or log
            // receives an effect without a wire representation.
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

        fn unwrap(outcome: Answered) -> Result<E::Answer, ErrorReport> {
            match outcome {
                Answered::Custom { payload: value } => {
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
    /// Interceptor names in outermost-first order, used to validate the layer
    /// stack during replay. Empty for a handler without layers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<String>,
}

impl HandlerDescriptor {
    /// A tool's handler, under [`tool_key`] of its `name`, with no layers
    /// and no embedding.
    pub fn tool(name: &str, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            key: tool_key(name),
            family: FamilyDescriptor::Tool {
                name: name.to_owned(),
                description: description.into(),
                parameters,
                embedding: None,
            },
            layers: Vec::new(),
        }
    }
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
pub enum EffectKind {
    /// A completion request.
    Completion {
        /// The prepared request, shared with the recorder and the log.
        request: Arc<CompletionRequest>,
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
    /// A retrieval.
    Retrieve {
        /// The query.
        query: RetrieveQuery,
    },
    /// An out-of-tree effect.
    Custom {
        /// The host-defined kind label.
        kind: Arc<str>,
        /// The host-defined payload.
        payload: serde_json::Value,
    },
}

/// The effect a model's tool call is: the tool's name and the call's raw
/// arguments, as a tool handler is dispatched it.
///
/// ```
/// use rig_core::effect::EffectKind;
/// use rig_core::message::{ToolCall, ToolFunction, ToolName};
///
/// let read = ToolFunction::new(ToolName::new("read")?, serde_json::json!({ "path": "a.rs" }));
/// let kind = EffectKind::from(&ToolCall::from_wire("call-1", read));
/// assert!(matches!(kind, EffectKind::ToolCall { ref name, .. } if name == "read"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl From<&crate::message::ToolCall> for EffectKind {
    fn from(call: &crate::message::ToolCall) -> Self {
        Self::ToolCall {
            name: call.function.name.as_str().to_owned(),
            args: call.function.raw_arguments(),
        }
    }
}

impl EffectKind {
    /// The family of this effect.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion { .. } => EffectFamily::Completion,
            Self::ToolCall { .. } => EffectFamily::Tool,
            Self::Embed { .. } => EffectFamily::Embed,
            Self::Rerank { .. } => EffectFamily::Rerank,
            Self::Retrieve { .. } => EffectFamily::Retrieve,
            Self::Custom { .. } => EffectFamily::Custom,
        }
    }

    /// Classification label without payload data. Custom effects return their
    /// host-defined kind label.
    pub fn name(&self) -> &str {
        match self {
            Self::Custom { kind, .. } => kind,
            Self::Completion { .. }
            | Self::ToolCall { .. }
            | Self::Embed { .. }
            | Self::Rerank { .. }
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
            | Self::Retrieve { .. }
            | Self::Custom { .. } => false,
        }
    }
}

/// The inputs of an embedding request: the texts or images an embedding
/// model's call embeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "modality", content = "inputs", rename_all = "snake_case")]
pub enum EmbedInputs {
    /// Text documents.
    Texts(Vec<String>),
    /// Image bytes.
    Images(Vec<Vec<u8>>),
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
pub enum Answered {
    /// A unary completion, shared with the records and layers that read it.
    Completion(Arc<CompletionResponse>),
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
    /// Retrieved documents.
    Documents(RetrievedDocuments),
    /// An out-of-tree answer.
    Custom {
        /// The answer, nested so every JSON shape is representable.
        payload: serde_json::Value,
    },
}

impl Answered {
    /// The family this outcome answers.
    pub const fn family(&self) -> EffectFamily {
        match self {
            Self::Completion(_) => EffectFamily::Completion,
            Self::ToolResult { .. } => EffectFamily::Tool,
            Self::Embeddings(_) => EffectFamily::Embed,
            Self::Reranked(_) => EffectFamily::Rerank,
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
    Images(EmbeddingResponse),
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
    /// Explicitly published result metadata, including on errors: a
    /// tool's, or the facts any layer of the dispatch published
    /// ([`PublishedContext`](crate::tool::PublishedContext)). Never
    /// inbound authentication values or runtime scopes. `None` means nothing
    /// was published before resolving; the wire field is required, and
    /// `null` records that nothing was published.
    #[serde(rename = "tool_output", deserialize_with = "Option::deserialize")]
    pub published: Option<crate::tool::ToolResultContext>,
    /// The dispatch's id.
    pub id: EffectId,
    /// The handler the effect was routed to.
    pub key: HandlerKey,
    /// The effect.
    pub kind: EffectKind,
    /// The answer.
    pub outcome: Result<Answered, ErrorReport>,
    /// A streamed dispatch's items, verbatim, when the recorder was asked
    /// to keep them (`EffectLogRecorder::keeping_stream_events`); `None`
    /// otherwise, and the answer is the fold. A replayer re-emits these
    /// when present, so a replayed consumer sees the original fragment
    /// boundaries. They read back through [`StreamRecord::parse_prefix`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<StreamRecord>,
    /// Who a streamed dispatch's reply was from, as its relay stated it
    /// before the first item, when the items are kept. A replayer sends it
    /// first, so a replayed stream cut short knows its origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_origin: Option<crate::message::Origin>,
    /// Parent dispatch for a nested handler call, or `None` for a root dispatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<EffectId>,
    /// Stable identifier of the dispatching program or run, not a runtime handle.
    /// `None` when the dispatcher supplied no scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<std::sync::Arc<str>>,
    /// What the dispatcher made the effect for, such as `"turn"` or
    /// `"compaction"`, from its [`DispatchContext`](crate::serve::DispatchContext). `None` when
    /// it named no purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<std::borrow::Cow<'static, str>>,
    /// The turn of its dispatcher's conversation it was made in, as the
    /// host numbers turns (the first is `0`). `None` when the host keeps no
    /// turns or it belongs to none, such as a call made between turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u64>,
    /// When a handler started serving it, in milliseconds since the Unix
    /// epoch. `None` when no handler started it (it was answered before it
    /// started), or when the recorder keeps no times.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_ms: Option<u64>,
    /// When its outcome was known, in milliseconds since the Unix epoch.
    /// `None` when the recorder keeps no times.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_ms: Option<u64>,
    /// The outcome a reviewer put in place of [`Self::outcome`], which stays
    /// the handler's (or the early answer's) own. `None` when nothing
    /// replaced it. A replayer answers with [`Self::outcome`] and lets the
    /// reviewers run again, as it reruns layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reviewed: Option<Result<Answered, ErrorReport>>,
    /// How the outcome came about.
    #[serde(default, skip_serializing_if = "Disposition::is_ran")]
    pub disposition: Disposition,
    /// Who decided the answer before it ran, when a decision answered it.
    /// `None` when its handler or its owner answered it: an effect no gate
    /// decided ran, and an allow is not a decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decided_by: Option<Decider>,
}

/// Who decided an effect's answer before it ran ([`EffectRecord::decided_by`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decider {
    /// A person, through a front: an answer to a question, a line they typed.
    Person,
    /// A plugin, by its name, such as `print` or `protected-paths`.
    Plugin(Cow<'static, str>),
}

impl Decider {
    /// The plugin called `name`.
    pub fn plugin(name: impl Into<Cow<'static, str>>) -> Self {
        Self::Plugin(name.into())
    }
}

/// How an [`EffectRecord`]'s outcome came about. A recorder of handler
/// dispatches records [`Ran`](Self::Ran); a host that answers, replaces or
/// cancels effects itself records the others.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    /// A handler served it, and the outcome is its answer.
    #[default]
    Ran,
    /// Answered before any handler started it, such as by a policy that
    /// refused it or a replayer.
    AnsweredBeforeStart,
    /// Answered from outside while its handler ran; the handler was
    /// cancelled.
    AnsweredWhileRunning,
    /// Another effect replaced it, and its outcome was never applied.
    Superseded,
    /// Cancelled before it had an outcome.
    Cancelled,
}

impl EffectRecord {
    /// The record of effect `id`, `kind`, routed to `key` and answered with
    /// `outcome`, and nothing else known: no parent, scope, purpose, turn, times,
    /// tool output, kept items, review or decider, and [`Disposition::Ran`]. A host
    /// that answers or cancels an effect itself builds its record from this
    /// and sets what it knows.
    pub fn new(
        id: EffectId,
        key: HandlerKey,
        kind: EffectKind,
        outcome: Result<Answered, ErrorReport>,
    ) -> Self {
        Self {
            published: None,
            id,
            key,
            kind,
            outcome,
            events: None,
            stream_origin: None,
            parent: None,
            scope: None,
            purpose: None,
            turn: None,
            started_ms: None,
            finished_ms: None,
            reviewed: None,
            disposition: Disposition::Ran,
            decided_by: None,
        }
    }
}

impl Disposition {
    /// Whether this is [`Self::Ran`], the default a record leaves out.
    fn is_ran(&self) -> bool {
        *self == Self::Ran
    }
}

/// Required handler keys and their effect families, ordered by key.
/// [`Self::is_subset_of`] validates the requirements against handler descriptors.
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

    /// Validates that every key is served with the required family, returning
    /// the first gap in key order. Uses the first descriptor matching each key.
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

    /// Returns missing and mismatched entries in this row's key order, followed
    /// by extra entries in the other row's key order.
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
    assert_wire::<RetrieveQuery>();
    assert_wire::<Answered>();
    assert_wire::<EmbedOutputs>();
    assert_wire::<RetrievedDocuments>();
    assert_wire::<EffectRecord>();
    assert_wire::<Decider>();
    assert_wire::<EffectRow>();
    assert_wire::<StreamRecord>();
};

#[cfg(test)]
mod tests;
