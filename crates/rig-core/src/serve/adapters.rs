//! Handlers over the impl-side traits.
//!
//! Each adapter translates its family's [`EffectKind`] arm into the trait
//! call and the result into an [`Outcome`] (or
//! [`StreamEvent`](crate::streaming::StreamEvent)s), so a
//! provider or tool author writes the impl-side trait exactly as before and
//! registers the adapter.

use futures::StreamExt;

use crate::{
    completion::{CompletionModel, ModelRef},
    effect::{
        EffectFamily, EffectKind, EmbedInputs, EmbedModality, EmbedOutputs, FamilyDescriptor,
        HandlerDescriptor, HandlerKey, MemoryOp, MemoryOutcome, Outcome, RetrieveQuery,
        RetrievedDocuments, ToolEmbeddingDescriptor,
    },
    embeddings::{EmbeddingModel, ImageEmbeddingModel},
    error::{ErrorKind, ErrorReport},
    memory::ConversationMemory,
    rerank::RerankModel,
    tool::{ErasedTool, Tool, ToolEmbedding},
    vector_store::{VectorStoreError, VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{OutcomeSink, Serve};
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

/// A [`CompletionModel`] as a handler. Unary and streaming completions both
/// route here; the descriptor carries the model's label and capability
/// snapshot.
pub struct CompletionAdapter<M> {
    label: ModelRef,
    model: M,
}

impl<M> CompletionAdapter<M> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<ModelRef>, model: M) -> Self {
        Self {
            label: label.into(),
            model,
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }
}

impl<M> Serve for CompletionAdapter<M>
where
    M: CompletionModel + 'static,
{
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::model_key(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: self.label.clone(),
                capabilities: self.model.capabilities(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, mut sink: OutcomeSink) {
        match kind {
            EffectKind::Completion {
                request,
                stream: false,
            } => {
                let outcome = self
                    .model
                    .completion(request)
                    .await
                    .map(Outcome::Completion)
                    .map_err(ErrorReport::from);
                sink.resolve(outcome).await;
            }
            EffectKind::Completion {
                request,
                stream: true,
            } => {
                let mut stream = match self.model.stream(request).await {
                    Ok(stream) => stream,
                    Err(error) => {
                        sink.resolve(Err(ErrorReport::from(error))).await;
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
                    if sink.send(item).await.is_err() {
                        // The consumer is gone: dropping the provider
                        // stream fires its abort.
                        return;
                    }
                }
            }
            other @ (EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Completion, &other)))
                    .await;
            }
        }
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

    /// Wrap a retrievable tool: the descriptor carries its embedding
    /// context so a catalog can advertise it by similarity.
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
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::ToolCall { args, context, .. } => {
                let mut context = context;
                let result = ErasedTool::execute(&self.tool, args, &mut context).await;
                sink.resolve(Ok(Outcome::ToolResult { result, context }))
                    .await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Tool, &other)))
                    .await;
            }
        }
    }
}

/// The callback shape of a tool defined at runtime (an MCP tool, a closure
/// built from an agent). The callback is the handler; there is no other
/// erasure of it.
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

/// A tool defined by a name, a schema and a callback — the runtime-defined
/// tool as a handler.
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
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::ToolCall { args, context, .. } => {
                let mut context = context;
                let result =
                    crate::tool::contextual::execute_callback(&self.callback, args, &mut context)
                        .await;
                sink.resolve(Ok(Outcome::ToolResult { result, context }))
                    .await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Tool, &other)))
                    .await;
            }
        }
    }
}

/// A text [`EmbeddingModel`] as a handler.
pub struct EmbedAdapter<E> {
    label: String,
    model: E,
}

impl<E> EmbedAdapter<E> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<String>, model: E) -> Self {
        Self {
            label: label.into(),
            model,
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &E {
        &self.model
    }
}

impl<E> Serve for EmbedAdapter<E>
where
    E: EmbeddingModel + 'static,
{
    type Family = family::Embed;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(format!("embed:{}", self.label)),
            family: FamilyDescriptor::Embed {
                model: self.label.clone(),
                dims: Some(self.model.ndims()),
                max_documents: self.model.max_documents(),
                modality: EmbedModality::Text,
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Embed {
                inputs: EmbedInputs::Texts(texts),
            } => {
                let outcome = self
                    .model
                    .embed_texts_response(texts)
                    .await
                    .map(|response| Outcome::Embeddings(EmbedOutputs::Texts(response)))
                    .map_err(ErrorReport::from);
                sink.resolve(outcome).await;
            }
            EffectKind::Embed {
                inputs: EmbedInputs::Images(_),
            } => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    "a text embedding handler cannot embed images",
                )))
                .await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Embed, &other)))
                    .await;
            }
        }
    }
}

/// A [`RerankModel`] as a handler.
pub struct RerankAdapter<M> {
    label: String,
    model: M,
}

impl<M> RerankAdapter<M> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<String>, model: M) -> Self {
        Self {
            label: label.into(),
            model,
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }
}

impl<M> Serve for RerankAdapter<M>
where
    M: RerankModel + 'static,
{
    type Family = family::Rerank;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(format!("rerank:{}", self.label)),
            family: FamilyDescriptor::Rerank {
                model: self.label.clone(),
                max_documents: self.model.max_documents(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Rerank { request } => {
                let outcome = self
                    .model
                    .rerank(&request.query, request.documents)
                    .await
                    .map(Outcome::Reranked)
                    .map_err(ErrorReport::from);
                sink.resolve(outcome).await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Rerank, &other)))
                    .await;
            }
        }
    }
}

/// An [`ImageEmbeddingModel`] as a handler.
pub struct ImageEmbedAdapter<E> {
    label: String,
    model: E,
}

impl<E> ImageEmbedAdapter<E> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<String>, model: E) -> Self {
        Self {
            label: label.into(),
            model,
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &E {
        &self.model
    }
}

impl<E> Serve for ImageEmbedAdapter<E>
where
    E: ImageEmbeddingModel + 'static,
{
    type Family = family::Embed;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(format!("embed:{}", self.label)),
            family: FamilyDescriptor::Embed {
                model: self.label.clone(),
                dims: Some(self.model.ndims()),
                max_documents: self.model.max_documents(),
                modality: EmbedModality::Image,
            },
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Embed {
                inputs: EmbedInputs::Images(images),
            } => {
                let outcome = self
                    .model
                    .embed_images_response(images)
                    .await
                    .map(|response| Outcome::Embeddings(EmbedOutputs::Images(response)))
                    .map_err(ErrorReport::from);
                sink.resolve(outcome).await;
            }
            EffectKind::Embed {
                inputs: EmbedInputs::Texts(_),
            } => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    "an image embedding handler cannot embed text",
                )))
                .await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Embed, &other)))
                    .await;
            }
        }
    }
}

/// A [`ConversationMemory`] as a handler.
pub struct MemoryAdapter<M> {
    memory: M,
}

impl<M> MemoryAdapter<M> {
    /// Wrap `memory`.
    pub fn new(memory: M) -> Self {
        Self { memory }
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
            key: HandlerKey::from("memory"),
            family: FamilyDescriptor::Memory {},
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
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
                sink.resolve(outcome.map_err(ErrorReport::from)).await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Memory, &other)))
                    .await;
            }
        }
    }
}

/// A [`VectorStoreIndex`] as a handler. The index's filter type is rebuilt
/// from the dynamic filter on the wire; documents come back as JSON, and the
/// typed view deserialises on the client side.
pub struct RetrieveAdapter<I> {
    index: I,
}

impl<I> RetrieveAdapter<I> {
    /// Wrap `index`.
    pub fn new(index: I) -> Self {
        Self { index }
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
            key: HandlerKey::from("retrieve"),
            family: FamilyDescriptor::Retrieve {},
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
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
                sink.resolve(outcome).await;
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Memory { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                sink.resolve(Err(wrong_family(EffectFamily::Retrieve, &other)))
                    .await;
            }
        }
    }
}
