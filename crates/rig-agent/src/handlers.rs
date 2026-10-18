#![forbid(unsafe_code)]

//! The handler table an agent dispatches its effects through, served inline.
//!
//! [`Handlers`] maps keys to [`ErasedHandler`]s. A dispatch looks its handler
//! up and runs it in the caller's own future or stream, recorded by the
//! table's [`Recorder`] when one is installed; dropping the future or stream
//! cancels it. A handler dispatches nested effects through the [`Handlers`]
//! it is served with ([`Handlers::serving`]), which records them under its
//! own dispatch.
//!
//! ```
//! use rig_agent::handlers::{Handlers, ModelHandle};
//! use rig_core::effect::HandlerKey;
//! let handlers = Handlers::default();
//! let model: Result<ModelHandle, _> = handlers.handle(&HandlerKey::from("model"));
//! assert!(model.is_err());
//! ```

use std::{
    collections::{BTreeMap, VecDeque},
    marker::PhantomData,
    sync::{
        Arc, PoisonError, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    task::Poll,
};

use futures::StreamExt;
use serde::de::DeserializeOwned;
use tracing::Instrument;

use rig_core::{
    completion::{CompletionRequest, ModelRef, ProviderCapabilities},
    effect::{
        Answered, CustomEffect, EffectFamily, EffectId, EffectKind, Family, FamilyDescriptor,
        HandlerDescriptor, HandlerKey, Key, RetrieveQuery, RetrievedDocuments, family,
    },
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, DispatchContext, ErasedHandler, Recorder, Serve},
    streaming::{CompletionStream, Relayed, StreamEvents},
    tool::{PublishedContext, ToolContext, ToolResult},
    vector_store::{
        VectorSearchIdResult, VectorSearchResult,
        request::{Filter, VectorSearchRequest},
    },
    wasm_compat::WasmCompatSend,
};

/// A keyed table of effect handlers, dispatched inline. Clones share the
/// table, its effect ids and its recorder.
///
/// A dispatch runs in the caller's future, so the caller decides what runs
/// at once, and dropping the future cancels the dispatch. A handler that
/// must bound or serialize its calls does it around its own work, as below.
/// A dispatch looks its handler up once, as it begins on its first polls,
/// and runs on that handler to the end: registering a replacement or
/// deregistering the key changes only dispatches that begin later, and a
/// displaced handler is dropped once no running dispatch holds it. An
/// agent's tool calls run on the registration its turn pinned, so a
/// replacement reaches them from the next turn.
///
/// ```
/// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
/// use rig_agent::handlers::Handlers;
/// use rig_core::effect::{family, Answered, EffectKind, FamilyDescriptor, HandlerDescriptor};
/// use rig_core::serve::{Dispatch, Reply, Serve};
/// use tokio::sync::{Mutex, Semaphore};
///
/// fn describe(key: &str) -> HandlerDescriptor {
///     let family = FamilyDescriptor::Custom { kind: "demo".into() };
///     HandlerDescriptor { key: key.into(), family, layers: Vec::new() }
/// }
///
/// /// At most two lookups run at once; the others wait for a permit.
/// struct Lookup { permits: Semaphore, running: AtomicUsize, peak: Arc<AtomicUsize> }
///
/// impl Serve for Lookup {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { describe("lookup") }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
///         let _permit = self.permits.acquire().await;
///         let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
///         self.peak.fetch_max(now, Ordering::SeqCst);
///         tokio::task::yield_now().await;
///         self.running.fetch_sub(1, Ordering::SeqCst);
///         Reply::Outcome(Ok(Answered::Custom { payload: serde_json::Value::Null }))
///     }
/// }
///
/// /// A counter whose read and write must not interleave across calls.
/// struct Counter { count: Mutex<u64> }
///
/// impl Serve for Counter {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { describe("counter") }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
///         let mut count = self.count.lock().await;
///         let next = *count + 1;
///         tokio::task::yield_now().await;
///         *count = next;
///         Reply::Outcome(Ok(Answered::Custom { payload: next.into() }))
///     }
/// }
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), rig_core::error::ErrorReport> {
/// let handlers = Handlers::default();
/// let peak = Arc::new(AtomicUsize::new(0));
/// let lookup = Lookup { permits: Semaphore::new(2), running: AtomicUsize::new(0), peak: peak.clone() };
/// handlers.register("lookup", lookup)?;
/// handlers.register("counter", Counter { count: Mutex::new(0) })?;
///
/// let call = |key| {
///     let kind = EffectKind::Custom { kind: "demo".into(), payload: serde_json::Value::Null };
///     handlers.dispatch(key, kind)
/// };
/// let (lookup, counter) = ("lookup".into(), "counter".into());
/// futures::future::try_join_all((0..6).map(|_| call(&lookup))).await?;
/// assert_eq!(peak.load(Ordering::SeqCst), 2);
///
/// let counts = futures::future::try_join_all((0..3).map(|_| call(&counter))).await?;
/// let counts: Vec<_> = counts.into_iter().map(|answer| match answer {
///     Answered::Custom { payload } => payload,
///     _ => serde_json::Value::Null,
/// }).collect();
/// assert_eq!(counts, [1, 2, 3]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct Handlers {
    shared: Arc<Shared>,
    /// The dispatch every dispatch made through this value descends from:
    /// `None` for a host's or an agent's own table, the served dispatch for
    /// the one a handler is served with.
    parent: Option<EffectId>,
}

#[derive(Default)]
struct Shared {
    table: RwLock<Table>,
    next_id: AtomicU64,
}

#[derive(Default)]
struct Table {
    handlers: BTreeMap<HandlerKey, (ErasedHandler, HandlerDescriptor)>,
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
}

impl std::fmt::Debug for Handlers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handlers")
            .field("keys", &self.keys())
            .field("parent", &self.parent)
            .finish_non_exhaustive()
    }
}

impl Handlers {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, Table> {
        self.shared
            .table
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Table> {
        self.shared
            .table
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Register (or replace) the handler serving `key`. A replacement must
    /// keep the key's family: a family change is refused with
    /// `HandlerUnavailable` and the table is unchanged.
    pub fn register(
        &self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<(), ErrorReport> {
        self.register_erased(key, ErasedHandler::new(handler))
    }

    /// Register an already-erased handler.
    pub fn register_erased(
        &self,
        key: impl Into<HandlerKey>,
        handler: ErasedHandler,
    ) -> Result<(), ErrorReport> {
        let key = key.into();
        // User code (the descriptor) runs outside the lock.
        let described = handler.descriptor();
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: described.family,
            layers: described.layers,
        };
        let mut table = self.write();
        if let Some((_, current)) = table.handlers.get(&key) {
            let (current, family) = (current.family.family(), descriptor.family.family());
            if current != family {
                return Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "key `{key}` serves the {current:?} family; a {family:?} handler cannot replace it"
                    ),
                )
                .with_retryable(false));
            }
        }
        if let Some(recorder) = &table.recorder {
            recorder.handlers(vec![descriptor.clone()]);
        }
        let displaced = table.handlers.insert(key, (handler, descriptor));
        drop(table);
        // A handler's destructor may reenter the table.
        drop(displaced);
        Ok(())
    }

    /// Remove the handler serving `key`: a later dispatch to it answers
    /// `HandlerUnavailable`. Returns whether one was registered.
    pub fn deregister(&self, key: &HandlerKey) -> bool {
        let removed = self.write().handlers.remove(key);
        removed.is_some()
    }

    /// Record every dispatch into `recorder`, in place of any recorder
    /// installed before: the handlers registered now are handed to it first
    /// ([`Recorder::handlers`]), in key order, and each one registered later
    /// as it is registered.
    ///
    /// Effect ids are unique within one table and its clones only: two
    /// tables recording into one recorder issue the same ids, and the
    /// recorder mixes up their records. Programs that write one log
    /// dispatch through one table, such as agents built over it with
    /// [`AgentBuilder::over_handlers`](crate::AgentBuilder::over_handlers).
    pub fn record_to(&self, recorder: impl Recorder + Send + Sync) {
        self.record_with(Arc::new(recorder));
    }

    pub(crate) fn record_with(&self, recorder: Arc<dyn Recorder + Send + Sync>) {
        let mut table = self.write();
        recorder.handlers(table.handlers.values().map(|(_, d)| d.clone()).collect());
        table.recorder = Some(recorder);
    }

    /// The descriptor of the handler serving `key`, if any.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.read().handlers.get(key).map(|(_, d)| d.clone())
    }

    /// Every registered descriptor, in key order.
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        self.read()
            .handlers
            .values()
            .map(|(_, d)| d.clone())
            .collect()
    }

    /// Every registered key, in key order.
    pub fn keys(&self) -> Vec<HandlerKey> {
        self.read().handlers.keys().cloned().collect()
    }

    /// The dispatch every dispatch made through this value descends from:
    /// `Some` for the handlers a handler is served with.
    pub const fn parent(&self) -> Option<EffectId> {
        self.parent
    }

    /// A reference that does not keep the table alive.
    pub(crate) fn downgrade(&self) -> WeakHandlers {
        WeakHandlers(Arc::downgrade(&self.shared))
    }

    /// The handlers `dispatch` is served with: dispatches made through them
    /// are recorded as its children. `None` for a dispatch made elsewhere.
    /// Native targets only: a dispatch's scopes are thread-safe, and browser
    /// handlers are not.
    #[cfg(not(target_family = "wasm"))]
    pub fn serving(dispatch: &Dispatch) -> Option<Self> {
        dispatch.scope::<Self>().map(|handlers| (*handlers).clone())
    }

    /// Number later dispatches after `last`: a host resuming a recorded run
    /// passes the last id its restored state had issued (a cassette
    /// checkpoint's `last`), so its dispatches take the ids the log holds
    /// for them. Ids never go back: an id this table already issued above
    /// `last` keeps the count.
    pub fn resume_ids_after(&self, last: EffectId) {
        self.shared
            .next_id
            .fetch_max(last.as_u64(), Ordering::SeqCst);
    }

    /// The next effect id of this table.
    pub(crate) fn mint_id(&self) -> EffectId {
        EffectId::from_raw(self.shared.next_id.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// The handler under `key` and the recorder, read under one lock.
    fn lookup(
        &self,
        key: &HandlerKey,
    ) -> Result<(ErasedHandler, Option<Arc<dyn Recorder + Send + Sync>>), ErrorReport> {
        let table = self.read();
        let (handler, _) = table
            .handlers
            .get(key)
            .ok_or_else(|| handler_unavailable(key))?;
        Ok((handler.clone(), table.recorder.clone()))
    }

    /// The dispatch `id` of `kind`, with the scopes it is served with and,
    /// when the table records, its record begun.
    fn open(&self, id: EffectId, key: &HandlerKey, kind: &EffectKind, beside: Beside) -> Opened {
        let (handler, recorder) = match &beside.handler {
            Some(handler) => (handler.clone(), self.read().recorder.clone()),
            None => self.lookup(key)?,
        };
        let mut dispatch = Dispatch::new(id, kind.streams());
        #[cfg(not(target_family = "wasm"))]
        {
            let child = Self {
                shared: Arc::clone(&self.shared),
                parent: Some(id),
            };
            dispatch = dispatch.with_scope(Arc::new(child));
        }
        if let Some(context) = beside.context {
            dispatch = dispatch.with_scope(Arc::new(context));
        }
        if let Some(published) = beside.published {
            dispatch = dispatch.with_scope(published);
        }
        if let Some(recorder) = recorder {
            let origin = DispatchContext {
                parent: self.parent,
                ..DispatchContext::default()
            };
            recorder.begin(id, key.clone(), kind.clone(), origin);
            dispatch = dispatch.recorded_by(recorder);
        }
        Ok((handler, dispatch))
    }

    /// Dispatch a unary effect to the handler under `key`. A streaming
    /// completion dispatched unary resolves to the completion its stream
    /// ends with.
    pub fn dispatch(
        &self,
        key: &HandlerKey,
        kind: EffectKind,
    ) -> impl Future<Output = Result<Answered, ErrorReport>> + WasmCompatSend + 'static {
        let beside = Beside::for_kind(&kind);
        self.serve(self.mint_id(), key.clone(), kind, beside)
    }

    /// Dispatch a streaming effect to the handler under `key`. A unary kind
    /// yields one request error without reaching a handler.
    pub fn dispatch_stream(&self, key: &HandlerKey, kind: EffectKind) -> StreamEvents {
        self.serve_stream(self.mint_id(), key.clone(), kind, None)
    }

    /// Serve `kind` as the effect `id`, in the returned future.
    pub(crate) fn serve(
        &self,
        id: EffectId,
        key: HandlerKey,
        kind: EffectKind,
        beside: Beside,
    ) -> impl Future<Output = Result<Answered, ErrorReport>> + WasmCompatSend + 'static {
        let this = self.clone();
        async move {
            yield_once().await;
            let (handler, dispatch) = this.open(id, &key, &kind, beside)?;
            handler.handle(kind, dispatch).await.into_outcome().await
        }
        // The handler runs under the span the dispatch was made in.
        .instrument(tracing::Span::current())
    }

    /// Serve the streaming `kind` as the effect `id`, on `handler` when
    /// given, else on the one under `key`. A stream that ends before its
    /// response or error ends with `stream_truncated`.
    pub(crate) fn serve_stream(
        &self,
        id: EffectId,
        key: HandlerKey,
        kind: EffectKind,
        handler: Option<ErasedHandler>,
    ) -> StreamEvents {
        if !kind.streams() {
            let report = ErrorReport::new(
                ErrorKind::Request,
                format!(
                    "invalid dispatch: `{}` is a unary effect and cannot be dispatched as a stream",
                    kind.name()
                ),
            );
            return Box::pin(futures::stream::iter([Err(report)]));
        }
        let this = self.clone();
        let beside = Beside {
            handler,
            ..Beside::default()
        };
        let events = async_stream::stream! {
            yield_once().await;
            let (handler, dispatch) = match this.open(id, &key, &kind, beside) {
                Ok(opened) => opened,
                Err(report) => {
                    yield Err(report);
                    return;
                }
            };
            let mut events = handler.handle(kind, dispatch).await.into_stream();
            let (mut ahead, mut drained, mut ended) = (VecDeque::new(), false, false);
            loop {
                // What the handler has ready is read, and recorded, before
                // the next item is handed on, up to `READ_AHEAD` items.
                let next = std::future::poll_fn(|cx| {
                    while !drained && ahead.len() < READ_AHEAD {
                        match events.poll_next_unpin(cx) {
                            Poll::Ready(Some(item)) => ahead.push_back(item),
                            Poll::Ready(None) => drained = true,
                            Poll::Pending => break,
                        }
                    }
                    match ahead.pop_front() {
                        Some(item) => Poll::Ready(Some(item)),
                        None if drained => Poll::Ready(None),
                        None => Poll::Pending,
                    }
                })
                .await;
                let Some(item) = next else { break };
                ended |= matches!(item, Ok(Relayed::Done(_)) | Err(_));
                yield item;
            }
            if !ended {
                yield Err(rig_core::serve::stream_truncated());
            }
        };
        Box::pin(tracing_futures::Instrument::instrument(
            events,
            tracing::Span::current(),
        ))
    }

    /// Bind a typed view to `key`, checking the handler's family against
    /// `F` now rather than at first dispatch.
    pub fn handle<F: Family>(&self, key: &HandlerKey) -> Result<Handle<F>, ErrorReport> {
        let descriptor = self
            .descriptor(key)
            .ok_or_else(|| handler_unavailable(key))?;
        let found = descriptor.family.family();
        if found != F::FAMILY {
            return Err(family_mismatch(key, F::FAMILY, found));
        }
        Ok(Handle {
            handlers: self.clone(),
            descriptor,
            _family: PhantomData,
        })
    }

    /// Bind a typed key, checking that a handler of its family serves it.
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
            other => Err(family_mismatch(key, EffectFamily::Custom, other.family())),
        }
    }
}

/// A [`Handlers`] that does not keep its table alive.
pub(crate) struct WeakHandlers(std::sync::Weak<Shared>);

impl WeakHandlers {
    pub(crate) fn upgrade(&self) -> Option<Handlers> {
        self.0.upgrade().map(|shared| Handlers {
            shared,
            parent: None,
        })
    }
}

type Opened = Result<(ErasedHandler, Dispatch), ErrorReport>;

/// How many items of a streamed reply are read ahead of its consumer.
const READ_AHEAD: usize = 64;

/// Pending once, waking itself: the dispatches a caller makes in one pass
/// all begin, in the order made, before any of them is answered.
async fn yield_once() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if std::mem::replace(&mut yielded, true) {
            return Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    })
    .await;
}

/// What a dispatch is served with beside its effect.
#[derive(Default)]
pub(crate) struct Beside {
    /// The handler to serve it on, in place of the one under its key: a
    /// tool catalog's pinned registration.
    pub(crate) handler: Option<ErasedHandler>,
    /// The context a tool call runs with.
    pub(crate) context: Option<ToolContext>,
    /// Where what the tool publishes lands.
    pub(crate) published: Option<Arc<PublishedContext>>,
}

impl Beside {
    /// A tool call always gets a slot for what it publishes, so the record
    /// keeps it.
    pub(crate) fn for_kind(kind: &EffectKind) -> Self {
        Self {
            published: matches!(kind, EffectKind::ToolCall { .. }).then(PublishedContext::new),
            ..Self::default()
        }
    }
}

fn handler_unavailable(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("no handler serves key `{key}`"),
    )
    .with_retryable(false)
}

fn family_mismatch(key: &HandlerKey, wanted: EffectFamily, found: EffectFamily) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("handler `{key}` serves the {found} family, not {wanted}"),
    )
}

/// A typed view over a [`Handlers`] key for the family `F`.
#[derive(Clone)]
pub struct Handle<F: Family> {
    handlers: Handlers,
    descriptor: HandlerDescriptor,
    _family: PhantomData<fn() -> F>,
}

/// A completion model: `call`, `stream`, `capabilities`, `label`.
pub type ModelHandle = Handle<family::Completion>;
/// A tool: `call`.
pub type ToolHandle = Handle<family::Tool>;
/// A vector-store index: `top_n`, `top_n_ids`.
pub type IndexHandle = Handle<family::Retrieve>;

impl<F: Family> std::fmt::Debug for Handle<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    /// The descriptor now: a runtime replacement under the same key is
    /// visible. Falls back to the bind-time snapshot when nothing serves the
    /// key any more.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.handlers
            .descriptor(&self.descriptor.key)
            .unwrap_or_else(|| self.descriptor.clone())
    }

    /// Dispatch a typed request of this family. A request with no wire form
    /// fails without reaching a handler or the recorder.
    pub fn dispatch(
        &self,
        request: F::Request,
    ) -> impl Future<Output = Result<F::Answer, ErrorReport>> + WasmCompatSend + 'static {
        self.dispatch_mapped(request, Ok)
    }

    fn dispatch_mapped<T: 'static>(
        &self,
        request: F::Request,
        map: fn(F::Answer) -> Result<T, ErrorReport>,
    ) -> impl Future<Output = Result<T, ErrorReport>> + WasmCompatSend + 'static {
        let id = self.handlers.mint_id();
        let served = F::wrap(request).map(|kind| {
            let beside = Beside::for_kind(&kind);
            self.handlers
                .serve(id, self.descriptor.key.clone(), kind, beside)
        });
        async move { F::unwrap(served?.await?).and_then(map) }
    }
}

impl ModelHandle {
    fn family(&self) -> Option<(ModelRef, ProviderCapabilities)> {
        match self.descriptor().family {
            FamilyDescriptor::Completion {
                model,
                capabilities,
            } => Some((model, capabilities)),
            _ => None,
        }
    }

    /// The capability snapshot the handler advertises now.
    pub fn capabilities(&self) -> ProviderCapabilities {
        self.family()
            .map(|(_, capabilities)| capabilities)
            .unwrap_or_default()
    }

    /// The model's label as the handler advertises it now.
    pub fn label(&self) -> ModelRef {
        self.family()
            .map_or_else(|| ModelRef::new(self.key().as_str()), |(model, _)| model)
    }

    /// A unary completion.
    pub fn call(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<<family::Completion as Family>::Answer, ErrorReport>>
    + WasmCompatSend
    + 'static {
        self.dispatch(request)
    }

    /// A streamed completion, folded by the returned stream.
    pub fn stream(&self, request: CompletionRequest) -> CompletionStream {
        let kind = EffectKind::Completion {
            request: request.into(),
            stream: true,
        };
        let events = self.handlers.serve_stream(
            self.handlers.mint_id(),
            self.descriptor.key.clone(),
            kind,
            None,
        );
        CompletionStream::relay(self.label().to_string(), events)
    }
}

/// A tool result and what the tool published into its context.
#[derive(Debug, Clone)]
pub struct ToolAnswer {
    /// The result.
    pub result: ToolResult,
    /// The context after the tool ran: the inbound values it ran with and
    /// the result metadata it published.
    pub context: ToolContext,
}

impl ToolHandle {
    /// Call the tool with raw JSON `args` under `context`: the context
    /// travels beside the effect to the tool (never on the wire), and the
    /// answer carries what the tool published.
    pub fn call(
        &self,
        name: impl Into<String>,
        args: impl Into<String>,
        context: ToolContext,
    ) -> impl Future<Output = Result<ToolAnswer, ErrorReport>> + WasmCompatSend + 'static {
        let kind = EffectKind::ToolCall {
            name: name.into(),
            args: args.into(),
        };
        // The tool runs on a dispatch snapshot: inbound values, no result
        // metadata a previous call left on `context`.
        let mut inbound = context.for_dispatch();
        let published = PublishedContext::new();
        let beside = Beside {
            handler: None,
            context: Some(inbound.clone()),
            published: Some(Arc::clone(&published)),
        };
        let served = self.handlers.serve(
            self.handlers.mint_id(),
            self.descriptor.key.clone(),
            kind,
            beside,
        );
        async move {
            let result = family::Tool::unwrap(served.await?)?;
            let context = published.take().unwrap_or_else(|| {
                inbound.clear_scope();
                inbound
            });
            Ok(ToolAnswer { result, context })
        }
    }
}

impl IndexHandle {
    /// Scored documents, deserialized as `T`.
    pub fn top_n<T: DeserializeOwned + WasmCompatSend + 'static>(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> impl Future<Output = Result<Vec<VectorSearchResult<T>>, ErrorReport>> + WasmCompatSend + 'static
    {
        self.dispatch_mapped(RetrieveQuery::TopN { req }, |documents| match documents {
            RetrievedDocuments::Scored(results) => results
                .into_iter()
                .map(
                    |(score, id, document)| match serde_json::from_value(document) {
                        Ok(document) => Ok(VectorSearchResult {
                            score,
                            id,
                            document,
                        }),
                        Err(error) => Err(ErrorReport::new(
                            ErrorKind::Json,
                            format!("retrieved document `{id}` did not deserialize: {error}"),
                        )),
                    },
                )
                .collect(),
            RetrievedDocuments::Ids(_) => Err(wrong_shape("scored documents")),
        })
    }

    /// Scored ids.
    pub fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> impl Future<Output = Result<Vec<VectorSearchIdResult>, ErrorReport>> + WasmCompatSend + 'static
    {
        self.dispatch_mapped(
            RetrieveQuery::TopNIds { req },
            |documents| match documents {
                RetrievedDocuments::Ids(results) => Ok(results
                    .into_iter()
                    .map(|(score, id)| VectorSearchIdResult { score, id })
                    .collect()),
                RetrievedDocuments::Scored(_) => Err(wrong_shape("scored ids")),
            },
        )
    }
}

fn wrong_shape(expected: &'static str) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!("expected {expected}, the retrieve handler answered another shape"),
    )
}

#[cfg(test)]
mod tests;
