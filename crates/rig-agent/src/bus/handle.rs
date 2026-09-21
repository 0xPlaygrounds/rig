//! Typed bus handles and dispatch results for completion, tools, memory,
//! retrieval, embeddings, reranking, and custom effects.
//!
//! Handles are thread-safe on every target but are not serializable. Persist keys
//! and descriptors instead, then rebind them. Descriptor queries observe current
//! registrations and fall back to the bind-time snapshot when a key is absent.
//!
//! ```
//! use rig_agent::bus::{Bus, ModelHandle};
//! use rig_core::effect::HandlerKey;
//! let (dispatcher, registrar, driver) = Bus::channel();
//! let model: Result<ModelHandle, _> = dispatcher.handle(&HandlerKey::from("model"));
//! assert!(model.is_err());
//! ```

use std::{
    fmt,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use serde::de::DeserializeOwned;

use rig_core::{
    completion::{CompletionRequest, ProviderCapabilities},
    effect::{
        CustomEffect, EffectId, EffectKind, EmbedInputs, EmbedModality, EmbedOutputs, Family,
        FamilyDescriptor, HandlerDescriptor, HandlerKey, MemoryOp, MemoryOutcome, RerankRequest,
        RetrieveQuery, RetrievedDocuments, family,
    },
    embeddings::{Embedding, EmbeddingResponse, ImageEmbeddingResponse},
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    message::Message,
    streaming::StreamingCompletionResponse,
    tool::ToolContext,
    vector_store::request::{Filter, VectorSearchRequest},
};

use super::{DispatchOptions, Dispatcher, EffectStream, Pending};
use rig_core::effect::Key;

/// A typed view over the bus for the family `F`.
#[derive(Clone)]
pub struct Handle<F: Family> {
    dispatcher: Dispatcher,
    descriptor: HandlerDescriptor,
    _family: PhantomData<fn() -> F>,
}

/// A completion model: `complete`, `stream`, `capabilities`.
pub type ModelHandle = Handle<family::Completion>;
/// A tool: `call`.
pub type ToolHandle = Handle<family::Tool>;
/// A conversation-memory backend: `load`, `append`, `clear`.
pub type MemoryHandle = Handle<family::Memory>;
/// A vector-store index: `top_n`, `top_n_ids`.
pub type IndexHandle = Handle<family::Retrieve>;
/// An embedding model; the modality is on the descriptor, not the type.
pub type EmbedHandle = Handle<family::Embed>;
/// A reranking model: `rerank`.
pub type RerankHandle = Handle<family::Rerank>;

impl<F: Family> fmt::Debug for Handle<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("family", &F::FAMILY)
            .field("key", &self.descriptor.key)
            .finish_non_exhaustive()
    }
}

impl<F: Family> Handle<F> {
    /// The key this handle dispatches to.
    pub fn key(&self) -> &HandlerKey {
        &self.descriptor.key
    }

    /// The descriptor *now*: re-read from the dispatcher's table, so a
    /// runtime replacement under the same key is visible. Falls back to the
    /// bind-time snapshot when nothing serves the key any more.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.dispatcher
            .descriptor(&self.descriptor.key)
            .unwrap_or_else(|| self.descriptor.clone())
    }

    /// Whether the bus behind this handle has closed.
    pub fn is_closed(&self) -> bool {
        self.dispatcher.is_closed()
    }

    fn dispatch_kind(&self, kind: EffectKind) -> Pending {
        self.dispatcher.dispatch(&self.descriptor.key, kind)
    }

    /// Dispatch a wrapped request, or pre-fail the dispatch when the
    /// request had no wire form.
    fn dispatch_wrapped(&self, kind: Result<EffectKind, ErrorReport>) -> Pending {
        match kind {
            Ok(kind) => self.dispatch_kind(kind),
            Err(report) => self.dispatcher.refused(report),
        }
    }
}

fn family_mismatch(
    key: &HandlerKey,
    wanted: rig_core::effect::EffectFamily,
    found: &FamilyDescriptor,
) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!(
            "handler `{key}` serves the {} family, not {wanted}",
            found.family()
        ),
    )
}

impl Dispatcher {
    /// Bind a typed view to `key`, checking the handler's family against
    /// `F` now rather than at first dispatch: asking for a [`ModelHandle`]
    /// at a tool key is `HandlerUnavailable` here.
    pub fn handle<F: Family>(&self, key: &HandlerKey) -> Result<Handle<F>, ErrorReport> {
        // Driver closure empties the table but must report a lifecycle error,
        // not a missing registration.
        if self.is_closed() {
            return Err(super::dispatcher::bus_closed());
        }
        let descriptor = self
            .descriptor(key)
            .ok_or_else(|| super::dispatcher::handler_unavailable(key))?;
        if descriptor.family.family() != F::FAMILY {
            return Err(family_mismatch(key, F::FAMILY, &descriptor.family));
        }
        Ok(Handle {
            dispatcher: self.clone(),
            descriptor,
            _family: PhantomData,
        })
    }

    /// Bind a typed key, checking existence and current family.
    /// Returns `BusClosed` for closure or `HandlerUnavailable` for missing or
    /// incompatible registrations, including incorrectly asserted unchecked keys.
    pub fn bind<F: Family>(&self, key: &Key<F>) -> Result<Handle<F>, ErrorReport> {
        self.handle(key.raw())
    }

    /// Bind a typed view to a host's custom effect: the handler under `key`
    /// must describe itself as [`FamilyDescriptor::Custom`] with `E::KIND`.
    pub fn custom<E: CustomEffect>(
        &self,
        key: &HandlerKey,
    ) -> Result<Handle<family::Custom<E>>, ErrorReport> {
        let handle = self.handle::<family::Custom<E>>(key)?;
        match &handle.descriptor.family {
            FamilyDescriptor::Custom { kind } if kind == E::KIND => Ok(handle),
            FamilyDescriptor::Custom { kind } => Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!(
                    "handler `{key}` serves the custom kind `{kind}`, not `{}`",
                    E::KIND
                ),
            )),
            other => Err(family_mismatch(key, F_CUSTOM, other)),
        }
    }
}

const F_CUSTOM: rig_core::effect::EffectFamily = rig_core::effect::EffectFamily::Custom;

/// Lazy unary dispatch mapped to a family-specific answer, optionally narrowed
/// to `T`. Propagates dispatch and conversion errors; dropping it cancels work.
/// Executor-neutral and `Unpin`.
#[must_use = "a dispatch does nothing until polled"]
pub struct Typed<F: Family, T = <F as Family>::Answer> {
    pending: Pending,
    map: fn(F::Answer) -> Result<T, ErrorReport>,
}

impl<F: Family, T> Typed<F, T> {
    /// The dispatch this one was made from, if a handler made it.
    pub const fn parent(&self) -> Option<EffectId> {
        self.pending.parent()
    }

    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.pending.id()
    }

    fn narrow(pending: Pending, map: fn(F::Answer) -> Result<T, ErrorReport>) -> Self {
        Self { pending, map }
    }
}

impl<F: Family, T> fmt::Debug for Typed<F, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Typed")
            .field("family", &F::FAMILY)
            .field("id", &self.id())
            .finish()
    }
}

impl<F: Family, T> Unpin for Typed<F, T> {}

impl<F: Family, T> Future for Typed<F, T> {
    type Output = Result<T, ErrorReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match Pin::new(&mut this.pending).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(report)) => Poll::Ready(Err(report)),
            Poll::Ready(Ok(outcome)) => Poll::Ready(F::unwrap(outcome).and_then(this.map)),
        }
    }
}

/// A completion dispatch in flight.
pub type Completion = Typed<family::Completion>;

/// A tool result and published dispatch context. Context is not part of the wire payload.
#[derive(Debug, Clone)]
pub struct ToolAnswer {
    /// The result.
    pub result: rig_core::tool::ToolResult,
    /// The dispatch context after the tool ran: the inbound values it ran
    /// with and the result metadata it published. A handler that published
    /// nothing answers with the inbound values alone.
    pub context: ToolContext,
}

/// A tool call in flight ([`ToolHandle::call`]): the result and the
/// context the tool published. `Unpin`, cancelled by drop.
#[must_use = "a dispatch does nothing until polled"]
pub struct ToolCall {
    pending: Pending,
    published: Option<std::sync::Arc<rig_core::tool::PublishedContext>>,
    /// The dispatch snapshot the call was made under: the answer's context
    /// when the handler published nothing (a handler that is not a tool
    /// adapter), so the inbound values the tool ran with are never lost.
    inbound: ToolContext,
}

impl ToolCall {
    /// The dispatch this one was made from, if a handler made it.
    pub const fn parent(&self) -> Option<EffectId> {
        self.pending.parent()
    }

    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.pending.id()
    }
}

impl fmt::Debug for ToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolCall").field("id", &self.id()).finish()
    }
}

impl Unpin for ToolCall {}

impl Future for ToolCall {
    type Output = Result<ToolAnswer, ErrorReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match Pin::new(&mut this.pending).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(report)) => Poll::Ready(Err(report)),
            Poll::Ready(Ok(outcome)) => Poll::Ready(family::Tool::unwrap(outcome).map(|result| {
                // Returned context must not retain live dispatch scopes.
                let context = this
                    .published
                    .as_ref()
                    .and_then(|published| published.take())
                    .unwrap_or_else(|| {
                        let mut inbound = std::mem::take(&mut this.inbound);
                        inbound.clear_scope();
                        inbound
                    });
                ToolAnswer { result, context }
            })),
        }
    }
}
/// A retrieval in flight, deserialized on this side of the bus.
pub type Retrieval<T> = Typed<family::Retrieve, Vec<(f64, String, T)>>;

impl<F: Family> Handle<F> {
    /// Dispatch a typed request of this family: one implementation for
    /// every family, the shapes coming from [`Family`].
    pub fn dispatch(&self, request: F::Request) -> Typed<F> {
        Typed::narrow(self.dispatch_wrapped(F::wrap(request)), Ok)
    }
}

fn wrong_shape(expected: &'static str, family: rig_core::effect::EffectFamily) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!("expected {expected}, the {family} handler answered another shape"),
    )
}

impl ModelHandle {
    /// The capability snapshot the handler advertises now.
    pub fn capabilities(&self) -> ProviderCapabilities {
        match self.descriptor().family {
            FamilyDescriptor::Completion { capabilities, .. } => capabilities,
            FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => ProviderCapabilities::default(),
        }
    }

    /// The model's label as the handler advertises it now.
    pub fn model_ref(&self) -> rig_core::completion::ModelRef {
        match self.descriptor().family {
            FamilyDescriptor::Completion { model, .. } => model,
            FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => {
                rig_core::completion::ModelRef::new(self.key().as_str())
            }
        }
    }

    /// A unary completion.
    pub fn complete(&self, request: CompletionRequest) -> Completion {
        self.complete_with_context(request, None)
    }

    /// A unary completion with explicit per-invocation observation state.
    /// The context overrides recorder context for this call only.
    pub fn complete_with_context(
        &self,
        request: CompletionRequest,
        context: Option<rig_core::observe::AdapterContext>,
    ) -> Completion {
        let options = DispatchOptions {
            adapter_context: context,
            ..DispatchOptions::default()
        };
        Typed::narrow(
            self.dispatcher.dispatch_with(
                &self.descriptor.key,
                EffectKind::Completion {
                    request,
                    stream: false,
                },
                options,
            ),
            Ok,
        )
    }

    /// Stream a completion through the canonical accumulator, surfacing bus errors
    /// as stream errors. Uses the model label initially and the terminal record's
    /// provider name when available.
    pub fn stream(&self, request: CompletionRequest) -> StreamingCompletionResponse {
        self.stream_with_context(request, None)
    }

    /// A streaming completion with explicit observation state retained by the
    /// invocation, including lazy startup and partial consumption.
    pub fn stream_with_context(
        &self,
        request: CompletionRequest,
        context: Option<rig_core::observe::AdapterContext>,
    ) -> StreamingCompletionResponse {
        let provider = self.model_ref().to_string();
        let options = DispatchOptions {
            adapter_context: context,
            ..DispatchOptions::default()
        };
        let stream: EffectStream = self.dispatcher.dispatch_stream_with(
            &self.descriptor.key,
            EffectKind::Completion {
                request,
                stream: true,
            },
            options,
        );
        wrap_stream(provider, stream)
    }
}

impl ToolHandle {
    /// The tool's name as advertised now.
    pub fn name(&self) -> String {
        match self.descriptor().family {
            FamilyDescriptor::Tool { name, .. } => name,
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => self.key().to_string(),
        }
    }

    /// Call the tool with raw JSON `args` under `context`: the context
    /// travels beside the effect to the tool (never on the wire), and the
    /// answer carries what the tool published.
    pub fn call(
        &self,
        name: impl Into<String>,
        args: impl Into<String>,
        context: ToolContext,
    ) -> ToolCall {
        let kind = EffectKind::ToolCall {
            name: name.into(),
            args: args.into(),
        };
        // The tool runs on a dispatch snapshot (inbound values, no result
        // metadata a previous call left on `context`); a handler that reads
        // the scope directly sees the same snapshot the adapter would.
        let inbound = context.for_dispatch();
        drop(context);
        let pending = self.dispatcher.dispatch_with(
            &self.descriptor.key,
            kind,
            DispatchOptions::default().with_tool_context(inbound.clone()),
        );
        let published = pending.published_context();
        ToolCall {
            pending,
            published,
            inbound,
        }
    }
}

impl MemoryHandle {
    /// Load a conversation's history.
    pub fn load(&self, conversation: ConversationId) -> Typed<family::Memory, Vec<Message>> {
        Typed::narrow(
            self.dispatch_wrapped(family::Memory::wrap(MemoryOp::Load { conversation })),
            |answer| match answer {
                MemoryOutcome::Loaded { messages } => Ok(messages),
                MemoryOutcome::Appended | MemoryOutcome::Cleared => {
                    Err(wrong_shape("loaded messages", family::Memory::FAMILY))
                }
            },
        )
    }

    /// Append messages to a conversation.
    pub fn append(
        &self,
        conversation: ConversationId,
        messages: Vec<Message>,
    ) -> Typed<family::Memory, ()> {
        Typed::narrow(
            self.dispatch_wrapped(family::Memory::wrap(MemoryOp::Append {
                conversation,
                messages,
            })),
            |answer| match answer {
                MemoryOutcome::Appended => Ok(()),
                MemoryOutcome::Loaded { .. } | MemoryOutcome::Cleared => {
                    Err(wrong_shape("an append", family::Memory::FAMILY))
                }
            },
        )
    }

    /// Clear a conversation.
    pub fn clear(&self, conversation: ConversationId) -> Typed<family::Memory, ()> {
        Typed::narrow(
            self.dispatch_wrapped(family::Memory::wrap(MemoryOp::Clear { conversation })),
            |answer| match answer {
                MemoryOutcome::Cleared => Ok(()),
                MemoryOutcome::Loaded { .. } | MemoryOutcome::Appended => {
                    Err(wrong_shape("a clear", family::Memory::FAMILY))
                }
            },
        )
    }
}

/// Deserialize scored documents on this side of the bus: the wire carries
/// JSON, the type parameter never crosses it.
fn deserialize_scored<T: DeserializeOwned>(
    documents: RetrievedDocuments,
) -> Result<Vec<(f64, String, T)>, ErrorReport> {
    match documents {
        RetrievedDocuments::Scored(results) => results
            .into_iter()
            .map(
                |(score, id, document)| match serde_json::from_value::<T>(document) {
                    Ok(document) => Ok((score, id, document)),
                    Err(error) => Err(ErrorReport::new(
                        ErrorKind::Json,
                        format!("retrieved document `{id}` did not deserialize: {error}"),
                    )),
                },
            )
            .collect(),
        RetrievedDocuments::Ids(_) => {
            Err(wrong_shape("scored documents", family::Retrieve::FAMILY))
        }
    }
}

impl IndexHandle {
    /// Scored documents, deserialized on this side of the bus.
    pub fn top_n<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Retrieval<T> {
        Typed::narrow(
            self.dispatch_wrapped(family::Retrieve::wrap(RetrieveQuery::TopN { req })),
            deserialize_scored::<T>,
        )
    }

    /// Scored ids.
    pub fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Typed<family::Retrieve, Vec<(f64, String)>> {
        Typed::narrow(
            self.dispatch_wrapped(family::Retrieve::wrap(RetrieveQuery::TopNIds { req })),
            |documents| match documents {
                RetrievedDocuments::Ids(results) => Ok(results),
                RetrievedDocuments::Scored(_) => {
                    Err(wrong_shape("scored ids", family::Retrieve::FAMILY))
                }
            },
        )
    }
}

impl EmbedHandle {
    /// The modality the handler serves now.
    pub fn modality(&self) -> Option<EmbedModality> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { modality, .. } => Some(modality),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// The vector dimension the handler advertises now.
    pub fn ndims(&self) -> Option<usize> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { dims, .. } => dims,
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// The largest batch the handler advertises now.
    pub fn max_documents(&self) -> Option<usize> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { max_documents, .. } => Some(max_documents),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Rerank { .. }
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// Embed text documents.
    pub fn embed_texts(&self, texts: Vec<String>) -> Typed<family::Embed, EmbeddingResponse> {
        Typed::narrow(
            self.dispatch_wrapped(family::Embed::wrap(EmbedInputs::Texts(texts))),
            |outputs| match outputs {
                EmbedOutputs::Texts(response) => Ok(response),
                EmbedOutputs::Images(_) => {
                    Err(wrong_shape("text embeddings", family::Embed::FAMILY))
                }
            },
        )
    }

    /// Embed one text document.
    pub fn embed_text(&self, text: &str) -> Typed<family::Embed, Embedding> {
        fn first(outputs: EmbedOutputs) -> Result<Embedding, ErrorReport> {
            match outputs {
                EmbedOutputs::Texts(mut response) => response.embeddings.pop().ok_or_else(|| {
                    ErrorReport::new(
                        ErrorKind::Response,
                        "embedding handler returned an empty response for embed_text",
                    )
                }),
                EmbedOutputs::Images(_) => {
                    Err(wrong_shape("text embeddings", family::Embed::FAMILY))
                }
            }
        }
        Typed::narrow(
            self.dispatch_wrapped(family::Embed::wrap(EmbedInputs::Texts(vec![
                text.to_owned(),
            ]))),
            first,
        )
    }

    /// Embed image bytes.
    pub fn embed_images(
        &self,
        images: Vec<Vec<u8>>,
    ) -> Typed<family::Embed, ImageEmbeddingResponse> {
        Typed::narrow(
            self.dispatch_wrapped(family::Embed::wrap(EmbedInputs::Images(images))),
            |outputs| match outputs {
                EmbedOutputs::Images(response) => Ok(response),
                EmbedOutputs::Texts(_) => {
                    Err(wrong_shape("image embeddings", family::Embed::FAMILY))
                }
            },
        )
    }
}

impl RerankHandle {
    /// The model's label as the handler advertises it now.
    pub fn model_label(&self) -> String {
        match self.descriptor().family {
            FamilyDescriptor::Rerank { model, .. } => model,
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => self.key().to_string(),
        }
    }

    /// The largest batch the handler advertises now.
    pub fn max_documents(&self) -> Option<usize> {
        match self.descriptor().family {
            FamilyDescriptor::Rerank { max_documents, .. } => Some(max_documents),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// Rerank `documents` against `query`.
    pub fn rerank(
        &self,
        query: impl Into<String>,
        documents: Vec<String>,
    ) -> Typed<family::Rerank> {
        self.dispatch(RerankRequest {
            query: query.into(),
            documents,
        })
    }
}

/// Wrap bus events in a canonical completion accumulator, preserving stream errors.
pub(crate) fn wrap_stream(
    provider: impl Into<String>,
    stream: EffectStream,
) -> StreamingCompletionResponse {
    StreamingCompletionResponse::from_events(provider, Box::pin(stream))
}

// Bus views must stay thread-safe even where handlers permit local WASM state.
const _: () = {
    const fn assert_view<T: Clone + Send + Sync + 'static>() {}
    assert_view::<ModelHandle>();
    assert_view::<ToolHandle>();
    assert_view::<MemoryHandle>();
    assert_view::<IndexHandle>();
    assert_view::<EmbedHandle>();
    const fn assert_unpin<T: Unpin>() {}
    assert_unpin::<Completion>();
    assert_unpin::<ToolCall>();
};

#[cfg(test)]
mod tests;

/// Access the handler-scoped dispatcher. Nested calls and bound handles retain
/// the served dispatch as their parent for recording and cancellation;
/// reentrant calls onto an active ancestor's serial key are refused.
pub trait DispatchScope {
    /// The scoped dispatcher, or `None` for an inline dispatch.
    fn dispatcher(&self) -> Option<Dispatcher>;
}

impl DispatchScope for rig_core::serve::Dispatch {
    fn dispatcher(&self) -> Option<Dispatcher> {
        self.scope::<Dispatcher>().map(|scoped| (*scoped).clone())
    }
}
