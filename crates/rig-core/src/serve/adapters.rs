//! Effect handlers adapting model, tool, memory, and retrieval traits.
//! Each adapter translates its effect family into trait calls and returns
//! outcomes or streaming events.
//!
//! ```
//! use rig_core::{memory::InMemoryConversationMemory, serve::{Serve, adapters::MemoryAdapter}};
//!
//! let handler = MemoryAdapter::new(InMemoryConversationMemory::new());
//! assert_eq!(handler.descriptor().key.as_str(), "memory");
//! ```

use crate::{
    completion::ModelRef,
    driver::BoxedModel,
    effect::{
        EffectFamily, EffectKind, EmbedInputs, EmbedModality, EmbedOutputs, FamilyDescriptor,
        HandlerDescriptor, HandlerKey, MemoryOp, MemoryOutcome, Outcome, RetrieveQuery,
        RetrievedDocuments, ToolEmbeddingDescriptor,
    },
    error::{ErrorKind, ErrorReport},
    memory::ConversationMemory,
    operation::{Completion, Embedding, Rerank},
    tool::{ErasedTool, Tool, ToolEmbedding},
    vector_store::{VectorStoreError, VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
    wire::Operation,
};

use super::{Dispatch, Reply, Serve};
use crate::effect::family;

fn wrong_family(handler: EffectFamily, kind: &EffectKind) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!(
            "a {handler} handler cannot serve a `{}` effect",
            kind.name()
        ),
    )
}

/// A model as a handler under `label`. Completion, text-embedding and
/// rerank models serve their effect families; the operation decides how
/// ([`ServeOperation`]). The model is erased to its operation, so one
/// adapter type serves every wire and transport.
pub struct ModelAdapter<Op: Operation> {
    label: ModelRef,
    model: BoxedModel<Op>,
}

impl<Op: Operation> ModelAdapter<Op> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<ModelRef>, model: impl Into<BoxedModel<Op>>) -> Self {
        Self {
            label: label.into(),
            model: model.into(),
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &BoxedModel<Op> {
        &self.model
    }
}

/// How a model performing this operation serves its effect family: the
/// descriptor it advertises, and the call an effect becomes.
pub trait ServeOperation: Operation {
    /// The effect family the model serves.
    type Family: crate::effect::Served;

    /// The handler descriptor for a model labelled `label`.
    fn descriptor(label: &ModelRef, capabilities: Self::Capabilities) -> HandlerDescriptor;

    /// Serve `kind` through `model`, or refuse an effect of another family.
    fn serve(
        model: &BoxedModel<Self>,
        kind: EffectKind,
        dispatch: Dispatch,
    ) -> impl Future<Output = Reply> + WasmCompatSend;
}

impl<Op: ServeOperation> Serve for ModelAdapter<Op> {
    type Family = Op::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        Op::descriptor(&self.label, self.model.capabilities())
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        Op::serve(&self.model, kind, dispatch).await
    }
}

/// Unary and streaming completions both route here; the descriptor carries
/// the model's label and capability snapshot.
impl ServeOperation for Completion {
    type Family = family::Completion;

    fn descriptor(label: &ModelRef, capabilities: Self::Capabilities) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::model_key(label.as_str()),
            family: FamilyDescriptor::Completion {
                model: label.clone(),
                capabilities,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(model: &BoxedModel<Self>, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let context = dispatch.adapter_context();
        match kind {
            EffectKind::Completion {
                request,
                stream: false,
            } => {
                let result = match context {
                    Some(context) => model.call_observed(request, context).await,
                    None => model.call(request).await,
                };
                Reply::Outcome(result.map(Outcome::Completion).map_err(ErrorReport::from))
            }
            EffectKind::Completion {
                request,
                stream: true,
            } => {
                let opened = match context {
                    Some(context) => model.stream_observed(request, context),
                    None => model.stream(request),
                };
                match opened {
                    Ok(stream) => Reply::Stream(Box::pin(stream)),
                    Err(error) => Reply::Outcome(Err(ErrorReport::from(error))),
                }
            }
            other @ (EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Completion, &other)))
            }
        }
    }
}

/// A text embedding model embeds texts and refuses images.
impl ServeOperation for Embedding {
    type Family = family::Embed;

    fn descriptor(label: &ModelRef, capabilities: Self::Capabilities) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::embed_key(label.as_str()),
            family: FamilyDescriptor::Embed {
                model: label.to_string(),
                dims: Some(capabilities.ndims),
                max_documents: capabilities.max_documents,
                modality: EmbedModality::Text,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(model: &BoxedModel<Self>, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Embed {
                inputs: EmbedInputs::Texts(texts),
            } => Reply::Outcome(
                model
                    .call(texts)
                    .await
                    .map(|response| Outcome::Embeddings(EmbedOutputs::Texts(response)))
                    .map_err(ErrorReport::from),
            ),
            EffectKind::Embed {
                inputs: EmbedInputs::Images(_),
            } => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                "a text embedding handler cannot embed images",
            ))),
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Embed, &other)))
            }
        }
    }
}

impl ServeOperation for Rerank {
    type Family = family::Rerank;

    fn descriptor(label: &ModelRef, capabilities: Self::Capabilities) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::rerank_key(label.as_str()),
            family: FamilyDescriptor::Rerank {
                model: label.to_string(),
                max_documents: capabilities,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(model: &BoxedModel<Self>, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Rerank { request } => Reply::Outcome(
                model
                    .call(crate::operation::RerankRequest {
                        query: request.query,
                        documents: request.documents,
                    })
                    .await
                    .map(Outcome::Reranked)
                    .map_err(ErrorReport::from),
            ),
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Rerank, &other)))
            }
        }
    }
}

/// The context a tool call runs with: the driver's inbound values from
/// the dispatch's scope (`ToolContext`, as `for_dispatch`), else empty, with
/// every scope of the dispatch attached so the tool reaches its runtime by
/// type for the length of the call.
fn dispatch_context(dispatch: &Dispatch) -> crate::tool::ToolContext {
    dispatch
        .scope::<crate::tool::ToolContext>()
        .map(|inbound| inbound.for_dispatch())
        .unwrap_or_default()
        .with_scopes(dispatch.scopes())
}

/// Hand what the tool published back beside the dispatch, when the driver
/// asked for it ([`PublishedContext`](crate::tool::PublishedContext) in
/// the dispatch's scope); the result carries data only.
fn publish(dispatch: &Dispatch, context: crate::tool::ToolContext) {
    if let Some(published) = dispatch.scope::<crate::tool::PublishedContext>() {
        published.publish(context);
    }
}

/// A [`Tool`] as a handler, keyed by its name.
pub struct ToolAdapter<T> {
    tool: T,
    embedding: Option<ToolEmbeddingDescriptor>,
}

impl<T: Tool> ToolAdapter<T> {
    /// Wrap a static tool.
    pub fn new(tool: T) -> Self {
        Self {
            tool,
            embedding: None,
        }
    }

    /// Wraps a tool with embedding descriptions and serialized reconstruction
    /// context. Returns an error if context serialization fails.
    pub fn retrievable(tool: T) -> Result<Self, serde_json::Error>
    where
        T: ToolEmbedding,
    {
        let embedding = ToolEmbeddingDescriptor {
            context: serde_json::to_value(tool.context())?,
            embedding_docs: tool.embedding_docs(),
        };
        Ok(Self {
            tool,
            embedding: Some(embedding),
        })
    }

    /// The wrapped tool.
    pub fn tool(&self) -> &T {
        &self.tool
    }
}

impl<T> Serve for ToolAdapter<T>
where
    T: Tool + 'static,
{
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::tool_key(T::NAME),
            family: FamilyDescriptor::Tool {
                name: T::NAME.to_owned(),
                description: self.tool.description(),
                parameters: self.tool.parameters(),
                embedding: self.embedding.clone(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::ToolCall { name, .. } if name != T::NAME => {
                // Key routing must not invoke a different tool than the bound target.
                Reply::Outcome(Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!("tool handler `{}` asked to run `{name}`", T::NAME),
                )))
            }
            EffectKind::ToolCall { args, .. } => {
                let mut context = dispatch_context(&dispatch);
                let result = ErasedTool::execute(&self.tool, args, &mut context).await;
                publish(&dispatch, context);
                Reply::Outcome(Ok(Outcome::ToolResult { result }))
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Tool, &other)))
            }
        }
    }
}

/// Contextual tool callback accepting JSON arguments and returning canonical
/// output or a tool execution error.
pub trait ToolCallback:
    for<'a> Fn(
        &'a mut crate::tool::ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<
        'a,
        Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError>,
    > + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> ToolCallback for F where
    F: for<'a> Fn(
            &'a mut crate::tool::ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<
            'a,
            Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError>,
        > + WasmCompatSend
        + WasmCompatSync
{
}

/// A runtime-defined tool handler with a name, argument schema, and callback.
pub struct ToolFn<F> {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: F,
}

impl<F: ToolCallback> ToolFn<F> {
    /// Build a runtime-defined tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            callback,
        }
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<F> Serve for ToolFn<F>
where
    F: ToolCallback + 'static,
{
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::tool_key(&self.name),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::ToolCall { args, .. } => {
                let mut context = dispatch_context(&dispatch);
                let result =
                    crate::tool::contextual::execute_callback(&self.callback, args, &mut context)
                        .await;
                publish(&dispatch, context);
                Reply::Outcome(Ok(Outcome::ToolResult { result }))
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Tool, &other)))
            }
        }
    }
}

/// A [`ConversationMemory`] as a handler.
pub struct MemoryAdapter<M> {
    memory: M,
    label: Option<String>,
}

impl<M> MemoryAdapter<M> {
    /// Wrap `memory` as an agent's own memory, under the bare `memory` key.
    pub fn new(memory: M) -> Self {
        Self {
            memory,
            label: None,
        }
    }

    /// Wrap `memory` under `memory:<label>`, so a host can serve several
    /// backends (per tenant, per store) on one bus and dispatch to each by
    /// [`memory_key`](crate::effect::memory_key).
    pub fn labelled(label: impl Into<String>, memory: M) -> Self {
        Self {
            memory,
            label: Some(label.into()),
        }
    }

    /// The wrapped backend.
    pub fn memory(&self) -> &M {
        &self.memory
    }
}

impl<M> Serve for MemoryAdapter<M>
where
    M: ConversationMemory + 'static,
{
    type Family = family::Memory;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self
                .label
                .as_deref()
                .map_or_else(|| HandlerKey::from("memory"), crate::effect::memory_key),
            family: FamilyDescriptor::Memory {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Memory { op } => {
                let outcome = match op {
                    MemoryOp::Load { conversation } => self
                        .memory
                        .load(&conversation)
                        .await
                        .map(|messages| Outcome::Memory(MemoryOutcome::Loaded { messages })),
                    MemoryOp::Append {
                        conversation,
                        messages,
                    } => self
                        .memory
                        .append(&conversation, messages)
                        .await
                        .map(|()| Outcome::Memory(MemoryOutcome::Appended)),
                    MemoryOp::Clear { conversation } => self
                        .memory
                        .clear(&conversation)
                        .await
                        .map(|()| Outcome::Memory(MemoryOutcome::Cleared)),
                };
                Reply::Outcome(outcome.map_err(ErrorReport::from))
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Memory, &other)))
            }
        }
    }
}

/// A [`VectorStoreIndex`] as a handler. The index's filter type is rebuilt
/// from the dynamic filter on the wire; documents come back as JSON, and the
/// typed view deserialises on the client side.
pub struct RetrieveAdapter<I> {
    index: I,
    label: Option<String>,
}

impl<I> RetrieveAdapter<I> {
    /// Wrap `index` as an agent's own index, under the bare `retrieve` key.
    pub fn new(index: I) -> Self {
        Self { index, label: None }
    }

    /// Wrap `index` under `retrieve:<label>`, so a host can serve several
    /// indexes on one bus and dispatch to each by
    /// [`retrieve_key`](crate::effect::retrieve_key).
    pub fn labelled(label: impl Into<String>, index: I) -> Self {
        Self {
            index,
            label: Some(label.into()),
        }
    }

    /// The wrapped index.
    pub fn index(&self) -> &I {
        &self.index
    }
}

impl<I, F> Serve for RetrieveAdapter<I>
where
    I: VectorStoreIndex<Filter = F> + 'static,
    F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
{
    type Family = family::Retrieve;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self
                .label
                .as_deref()
                .map_or_else(|| HandlerKey::from("retrieve"), crate::effect::retrieve_key),
            family: FamilyDescriptor::Retrieve {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Retrieve { query } => {
                let outcome = match query {
                    RetrieveQuery::TopN { req } => {
                        match req.try_map_filter(F::from_dynamic_filter) {
                            Ok(req) => self
                                .index
                                .top_n::<serde_json::Value>(req)
                                .await
                                .map(|results| {
                                    Outcome::Documents(RetrievedDocuments::Scored(
                                        results
                                            .into_iter()
                                            .map(|(score, id, doc)| {
                                                (score, id, F::normalize_dynamic_document(doc))
                                            })
                                            .collect(),
                                    ))
                                })
                                .map_err(ErrorReport::from),
                            Err(error) => Err(ErrorReport::from(VectorStoreError::from(error))),
                        }
                    }
                    RetrieveQuery::TopNIds { req } => {
                        match req.try_map_filter(F::from_dynamic_filter) {
                            Ok(req) => self
                                .index
                                .top_n_ids(req)
                                .await
                                .map(|results| Outcome::Documents(RetrievedDocuments::Ids(results)))
                                .map_err(ErrorReport::from),
                            Err(error) => Err(ErrorReport::from(VectorStoreError::from(error))),
                        }
                    }
                };
                Reply::Outcome(outcome)
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Retrieve, &other)))
            }
        }
    }
}
