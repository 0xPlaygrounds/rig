//! Typed views over the bus: one generic [`Handle<F>`], five aliases.
//!
//! A handle is a [`Dispatcher`] plus the key it is bound to, checked at bind
//! time against the handler's family. It is `Clone + Send + Sync + 'static`
//! on every target by construction and — deliberately — never serde: the
//! descriptor is the serde half. A scene stores the [`HandlerKey`] and its
//! [`HandlerDescriptor`] and re-binds with [`Dispatcher::handle`] at load.
//!
//! The stored descriptor is the bind-time snapshot. Because a runtime
//! registration can replace what serves a key (model swapping),
//! [`Handle::descriptor`] and [`ModelHandle::capabilities`] re-read the
//! dispatcher's table rather than the field; the field exists for the family
//! check and for hosts that serialize a handle's identity.
//!
//! Handles implement none of the impl-side traits: a consumer calls the
//! inherent methods below, which are dispatches.
//!
//! ```compile_fail
//! use rig_bus::ModelHandle;
//!
//! fn requires_serialize<T: serde::Serialize>() {}
//! requires_serialize::<ModelHandle>();
//! ```
//!
//! ```compile_fail
//! use rig_bus::ToolHandle;
//!
//! fn requires_deserialize<T: for<'de> serde::Deserialize<'de>>() {}
//! requires_deserialize::<ToolHandle>();
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
        RetrieveQuery, RetrievedDocuments, ToolCallRequest, family,
    },
    embeddings::{Embedding, EmbeddingResponse, ImageEmbeddingResponse},
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    message::Message,
    streaming::StreamingCompletionResponse,
    tool::ToolContext,
    vector_store::request::{Filter, VectorSearchRequest},
};

use super::{Dispatcher, EffectStream, Pending};
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

    /// The bind-time snapshot of the descriptor.
    pub fn bound_descriptor(&self) -> &HandlerDescriptor {
        &self.descriptor
    }

    /// Whether the bus behind this handle has closed.
    pub fn is_closed(&self) -> bool {
        self.dispatcher.is_closed()
    }

    /// A view over `dispatcher` from a descriptor a host kept — a scene
    /// loaded before its handlers are re-registered — with **no** table
    /// check: the first dispatch answers `HandlerUnavailable` if nothing
    /// serves the key by then, exactly as for any stale key. The
    /// descriptor's family must be `F`; a mismatch is the host's
    /// programming error and panics here, at the host's line.
    #[track_caller]
    pub fn rebind(dispatcher: Dispatcher, descriptor: HandlerDescriptor) -> Self {
        assert!(
            descriptor.family.family() == F::FAMILY,
            "rebind: descriptor for `{}` serves the {} family, not {}",
            descriptor.key,
            descriptor.family.family(),
            F::FAMILY
        );
        Self {
            dispatcher,
            descriptor,
            _family: PhantomData,
        }
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
        // Lifecycle before wiring: on a closed bus the table is empty
        // because the driver is gone, and that is the answer — not
        // "nothing serves the key".
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

    /// Bind a typed view to a key that carries its family: an existence
    /// check only, the family was proven when the key was minted (a
    /// [`Key::new_unchecked`] that lied fails here as `HandlerUnavailable`).
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

/// A unary dispatch of the family `F`, mapped to its typed answer:
/// `Unpin`, executor-neutral, cancelled by drop — the same value as
/// [`Pending`] with the outcome narrowed by [`Family::unwrap`]. The second
/// parameter is the narrowed answer a convenience method returns
/// (`MemoryHandle::load` narrows `MemoryOutcome` to the messages); by
/// default it is the family's own answer.
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
/// A tool call in flight: the result and the context the tool published.
pub type ToolCall = Typed<family::Tool>;
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
        self.dispatch(request)
    }

    /// A streaming completion, wrapped back into a
    /// [`StreamingCompletionResponse`] over the B2 accumulator. Errors that
    /// cross the bus surface as the stream's error half, [`ErrorReport`].
    /// The stream is opened under the model's label and takes the provider's
    /// name from the terminal record, so `finish().provider` is what the
    /// unary path reports.
    pub fn stream(&self, request: CompletionRequest) -> StreamingCompletionResponse {
        let provider = self.model_ref().to_string();
        let stream: EffectStream = self.dispatcher.dispatch_stream(
            &self.descriptor.key,
            EffectKind::Completion {
                request,
                stream: true,
            },
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

    /// Call the tool with raw JSON `args` and a dispatch-scoped context.
    pub fn call(
        &self,
        name: impl Into<String>,
        args: impl Into<String>,
        context: ToolContext,
    ) -> ToolCall {
        self.dispatch(ToolCallRequest {
            name: name.into(),
            args: args.into(),
            context,
        })
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
    pub fn embed_text(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, ErrorReport>> + Unpin {
        futures::future::FutureExt::map(self.embed_texts(vec![text.to_owned()]), |result| {
            result.and_then(|mut response| {
                response.embeddings.pop().ok_or_else(|| {
                    ErrorReport::new(
                        ErrorKind::Response,
                        "embedding handler returned an empty response for embed_text",
                    )
                })
            })
        })
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

impl<E: CustomEffect> Handle<family::Custom<E>> {
    /// Dispatch the host's own effect.
    pub fn custom(&self, effect: E) -> Typed<family::Custom<E>> {
        self.dispatch(effect)
    }
}

/// Wrap an [`EffectStream`] back into a [`StreamingCompletionResponse`]:
/// the B2 accumulator folds the events on this side of the bus, and errors
/// that crossed it are the stream's own error half, [`ErrorReport`].
pub fn wrap_stream(
    provider: impl Into<String>,
    stream: EffectStream,
) -> StreamingCompletionResponse {
    StreamingCompletionResponse::from_events(provider, Box::pin(stream))
}

// Every alias is `Clone + Send + Sync + 'static` on every target, by
// construction; a compiled assertion keeps it so under wasm32-unknown-unknown
// as well as natively.
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
    // Raised 64 → 80 with `Bus::reopen`: a `Pending` carries its bus
    // generation (8 bytes) and its parked-sender slot (8 bytes).
    assert!(
        size_of::<Typed<family::Completion>>() <= 96,
        "Typed<Completion> budget: 96 bytes (measured 88 natively)"
    );
};

#[cfg(test)]
mod tests;

/// A handler's way back onto the bus that is serving it: the dispatcher the
/// driver attached to the sink, whose every dispatch — and every
/// [`Handle`] bound from it — carries the served dispatch's id as its
/// parent. Causality as data: the record names the chain, a host parents
/// the effect's entity at dispatch, and a nested dispatch that would wait
/// on its own serial key is refused rather than hung.
pub trait SinkDispatch {
    /// The scoped dispatcher, or `None` for a sink no bus driver served.
    fn dispatcher(&self) -> Option<Dispatcher>;
}

impl SinkDispatch for rig_core::serve::OutcomeSink {
    fn dispatcher(&self) -> Option<Dispatcher> {
        self.scope::<Dispatcher>().map(|scoped| (*scoped).clone())
    }
}

impl SinkDispatch for rig_core::serve::DetachedSink {
    fn dispatcher(&self) -> Option<Dispatcher> {
        self.scope::<Dispatcher>().map(|scoped| (*scoped).clone())
    }
}
