//! Handler dispatch, owned replies, and delivery observation.
//!
//! ```
//! use rig_core::{effect::EffectId, serve::Dispatch};
//!
//! let dispatch = Dispatch::new(EffectId::from_raw(1), false);
//! assert!(!dispatch.is_stream());
//! ```

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
};

use futures::StreamExt;

use crate::{
    completion::UnsupportedOption,
    effect::{Answered, EffectId, EffectKind, HandlerDescriptor, HandlerKey},
    error::{ErrorKind, ErrorReport},
    streaming::{Item, Relayed, StreamEvent, StreamEvents},
    tool::Touches,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

#[cfg(test)]
mod tests;

/// Boxed handler reply future, `Send` on native targets and locally polled on WASM.
pub type HandlerFuture<'a> = WasmBoxedFuture<'a, Reply>;

/// A registered effect handler returning an outcome or an owned stream.
///
/// Provider and tool authors do not implement this directly: the adapters
/// in [`crate::serve::adapters`] wrap models, tools and indexes
/// (`Model`, `Tool`, `VectorStoreIndex`). A
/// host implements it for out-of-tree kinds ([`EffectKind::Custom`], typed
/// through [`crate::effect::CustomEffect`]) or for a replayer.
///
/// A handler returns an outcome or an owned stream. The driver adapts that
/// reply to the requested delivery mode. Stream execution continues after
/// this method returns; dropping the reply cancels that work.
///
/// The returned future must be `Send` natively (it runs inside the driver's
/// task; the bound is the crate's `WasmCompatSend` marker, a no-op on
/// browser wasm). `Self::Family` is what a typed key can be proven against
/// (a typed registration); a handler with no one
/// family names [`crate::effect::family::Dynamic`].
///
/// A handler holding an `Rc` or a `Cell` is refused natively:
///
/// ```compile_fail
/// use std::rc::Rc;
/// use rig_core::{serve::{Dispatch, Reply, Serve}, effect::{EffectKind, HandlerDescriptor, family}};
///
/// struct Local(Rc<u8>);
/// impl Serve for Local {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { unimplemented!() }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply { unimplemented!() }
/// }
/// ```
///
/// ```compile_fail
/// use std::cell::Cell;
/// use rig_core::{serve::{Dispatch, Reply, Serve}, effect::{EffectKind, HandlerDescriptor, family}};
///
/// struct Local(Cell<u8>);
/// impl Serve for Local {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { unimplemented!() }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply { unimplemented!() }
/// }
/// ```
pub trait Serve: WasmCompatSend + WasmCompatSync {
    /// The family this handler serves, or `Dynamic`.
    type Family: crate::effect::Served;

    /// What this handler is: the family-keyed description a typed view
    /// checks at bind time and a scene serializes.
    fn descriptor(&self) -> HandlerDescriptor;

    /// The [descriptor](Self::descriptor)'s key. A handler whose
    /// descriptor is costly to build, such as a tool's argument schema,
    /// says it without building one.
    fn key(&self) -> HandlerKey {
        self.descriptor().key
    }

    /// Prepare an outcome or an owned stream for the driver to consume.
    fn serve(
        &self,
        kind: EffectKind,
        dispatch: Dispatch,
    ) -> impl Future<Output = Reply> + WasmCompatSend + use<'_, Self>;

    /// What dispatching `kind` may read and change, for a tool's handler
    /// ([`Tool::footprint`](crate::tool::Tool::footprint)); `None` when it
    /// cannot say, such as for arguments that do not parse.
    fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        let _ = kind;
        None
    }

    /// The options of `kind` this handler would leave out when it serves
    /// it, with nothing sent: for a model's handler, what its wire or model
    /// cannot honour ([`DynModel::check`](crate::driver::DynModel::check)).
    /// Empty when it leaves nothing out or cannot say, as by default.
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        let _ = kind;
        Vec::new()
    }
}

/// Object-safe handler interface implemented for every [`Serve`].
pub(crate) trait Handler: WasmCompatSend + WasmCompatSync {
    fn descriptor(&self) -> HandlerDescriptor;
    fn handle(&self, kind: EffectKind, dispatch: Dispatch) -> HandlerFuture<'_>;
    fn footprint(&self, kind: &EffectKind) -> Option<Touches>;
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption>;
}

// A type that is not a `Serve` should be told to implement `Serve`, never
// the crate-private `Handler` this blanket impl provides.
#[diagnostic::do_not_recommend]
impl<T: Serve> Handler for T {
    fn descriptor(&self) -> HandlerDescriptor {
        Serve::descriptor(self)
    }

    fn handle(&self, kind: EffectKind, dispatch: Dispatch) -> HandlerFuture<'_> {
        let observer = dispatch.observer.clone();
        let folded = dispatch.folded.clone();
        let streaming = dispatch.is_stream();
        let level = Level::enter(&dispatch.serving);
        Box::pin(async move {
            let reply = self.serve(kind, dispatch).await;
            // A handler that serves the dispatch inside another one, such
            // as a wrapper's inner handler, leaves the observer to the
            // outermost: what the wrapper publishes once its inner handler
            // answered is part of the record.
            if !level.outermost() {
                return reply.observed(streaming, None, None);
            }
            let seen = observer.and_then(|slot| lock(&slot).take());
            reply.observed(streaming, seen, folded)
        })
    }

    fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        Serve::footprint(self, kind)
    }

    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        Serve::check(self, kind)
    }
}

/// A shared handler is a handler: `Arc<H>` forwards, so one handler can be
/// registered under several keys.
impl<H: Serve + ?Sized> Serve for Arc<H> {
    type Family = H::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        (**self).descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        (**self).serve(kind, dispatch).await
    }

    fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        (**self).footprint(kind)
    }

    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        (**self).check(kind)
    }
}

/// One handler serving a dispatch, counted in the dispatch's handlers that
/// serve it one inside another until it answers or is dropped.
struct Level(Arc<AtomicUsize>);

impl Level {
    fn enter(serving: &Arc<AtomicUsize>) -> Self {
        serving.fetch_add(1, Ordering::AcqRel);
        Self(serving.clone())
    }

    /// Whether no handler serves the dispatch around this one: the inner
    /// ones answered or were dropped by now.
    fn outermost(&self) -> bool {
        self.0.load(Ordering::Acquire) == 1
    }
}

impl Drop for Level {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Shared, type-erased handler. Clones retain the same handler allocation.
/// Implements `Send + Sync` on native targets, but not on WASM.
#[derive(Clone)]
pub struct ErasedHandler {
    inner: ErasedInner,
    /// The handler's [key](Serve::key), read once.
    key: HandlerKey,
}

#[cfg(not(target_family = "wasm"))]
type ErasedInner = Arc<dyn Handler + Send + Sync>;
#[cfg(target_family = "wasm")]
type ErasedInner = Arc<dyn Handler>;

impl ErasedHandler {
    /// Erase `handler`.
    pub fn new(handler: impl Serve + 'static) -> Self {
        Self {
            key: handler.key(),
            inner: Arc::new(handler),
        }
    }

    /// A handler that serves nothing: it is `descriptor`, and every effect
    /// it gets is answered [`ErrorKind::HandlerUnavailable`] with `why`. For
    /// a model whose calls a plugin answers itself, such as from a batch,
    /// so a call that reaches the handler by mistake fails plainly while
    /// records still name the model.
    ///
    /// ```
    /// use rig_core::completion::ModelRef;
    /// use rig_core::effect::{FamilyDescriptor, HandlerDescriptor, model_key};
    /// use rig_core::serve::ErasedHandler;
    ///
    /// let descriptor = HandlerDescriptor {
    ///     key: model_key("batch/claude"),
    ///     family: FamilyDescriptor::Completion {
    ///         model: ModelRef::from("batch/claude"),
    ///         capabilities: Default::default(),
    ///     },
    ///     layers: Vec::new(),
    /// };
    /// let handler = ErasedHandler::unavailable(descriptor, "answered from the batch API");
    /// assert_eq!(handler.key(), &model_key("batch/claude"));
    /// ```
    pub fn unavailable(descriptor: HandlerDescriptor, why: impl Into<String>) -> Self {
        Self::new(Unavailable {
            descriptor,
            why: why.into(),
        })
    }

    /// Wrap this handler in a [`Layer`](super::Layer): `intercept` sees
    /// every dispatch before this handler does and every answer after.
    /// `handler.layered(a).layered(b)` calls `b.before` first and `a.after` first.
    pub fn layered(self, intercept: impl super::Intercept) -> Self {
        Self::new(super::Layer::new(self, intercept))
    }

    /// Rewrite this handler's completion streams: every item of each reply
    /// passes through a fresh `make(published)` in order, and the items it
    /// returns go on in its place. It sees the [`Relayed::Done`](crate::streaming::Relayed::Done)
    /// too, so a rewrite of the events also rewrites the response the reply
    /// ends with. A unary completion is rewritten as its events, so both
    /// delivery modes answer alike; other outcomes and errors pass as they
    /// are. `name` joins the descriptor's layers.
    ///
    /// `published` is the dispatch's
    /// [`PublishedContext`](crate::tool::PublishedContext): a fact the
    /// rewrite inserts there before it passes the `Done` on, such as that
    /// it split reasoning out of this reply, comes back to the driver beside
    /// the reply and is recorded with it.
    ///
    /// Unlike a [`Layer`](super::Layer), the rewrite is recorded: the
    /// record of a dispatch is the reply the consumer receives, so a replay
    /// of it does not rewrite twice.
    pub fn map_stream<M, F>(self, name: impl Into<String>, make: M) -> Self
    where
        M: Fn(Arc<crate::tool::PublishedContext>) -> F + WasmCompatSend + WasmCompatSync + 'static,
        F: FnMut(Relayed) -> Vec<Relayed> + WasmCompatSend + 'static,
    {
        Self::new(super::map::MapStream::new(self, name.into(), make))
    }

    /// What the erased handler is.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    /// The key of its [descriptor](Self::descriptor), without building one.
    pub fn key(&self) -> &HandlerKey {
        &self.key
    }

    /// Serve one effect: the driver's call, straight to the boxed handler.
    pub fn handle(&self, kind: EffectKind, dispatch: Dispatch) -> HandlerFuture<'_> {
        self.inner.handle(kind, dispatch)
    }

    /// What dispatching `kind` to this handler may read and change
    /// ([`Serve::footprint`]).
    pub fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        self.inner.footprint(kind)
    }

    /// The options of `kind` this handler would leave out ([`Serve::check`]).
    pub fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        self.inner.check(kind)
    }

    /// Whether two erased handlers are the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::fmt::Debug for ErasedHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErasedHandler")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// What [`ErasedHandler::unavailable`] erases: it answers every effect
/// `HandlerUnavailable` with `why`.
struct Unavailable {
    descriptor: HandlerDescriptor,
    why: String,
}

impl Serve for Unavailable {
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::HandlerUnavailable,
            self.why.clone(),
        )))
    }
}

/// Forwards serving to the wrapped handler; re-erasure adds another boxed future.
impl Serve for ErasedHandler {
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        self.handle(kind, dispatch).await
    }

    fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        self.inner.footprint(kind)
    }

    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        self.inner.check(kind)
    }
}

/// What an observer of one dispatch sees: its outcome and, when wanted, each streamed item.
pub trait Observe: Send + Sync {
    /// Provider witness associated with this dispatch, independently of recording.
    fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        None
    }

    /// The completed handler response, or the fold of a streaming handler's
    /// events. A resolved response retains content that has no stream block,
    /// such as images emitted as unknown events.
    fn outcome(&mut self, outcome: Result<Answered, ErrorReport>);
    /// Whether streamed items are wanted verbatim, with the origin and
    /// error items, beside each [`Self::event`].
    fn keep_events(&self) -> bool;
    /// One item pulled from the original handler stream, kept or not. This
    /// is recording evidence, not acknowledgement that a consumer received
    /// the item.
    fn event(&mut self, item: &Item<StreamEvent>);
    /// An error item in a kept stream.
    fn stream_error(&mut self, _error: &ErrorReport) {}
    /// Who a streamed reply is from, before its first item. Required, so a
    /// recording observer cannot drop it: a replayed stream cut short needs
    /// it to know its origin.
    fn origin(&mut self, origin: &crate::message::Origin);
    /// Observe one stream item together with its first folded outcome, if any.
    /// Drivers that snapshot recording concurrently with cancellation can override
    /// this operation to make the item and its answer one observation boundary.
    fn stream_item(
        &mut self,
        item: &Result<Relayed, ErrorReport>,
        outcome: Option<Result<Answered, ErrorReport>>,
    ) {
        match item {
            Ok(Relayed::Item(item)) => self.event(item),
            Ok(Relayed::Origin(origin)) if self.keep_events() => self.origin(origin),
            Err(error) if self.keep_events() => self.stream_error(error),
            _ => {}
        }
        if let Some(outcome) = outcome {
            self.outcome(outcome);
        }
    }
    /// The dispatch was decided before any handler served it, by the layer
    /// named: the record it opened is forgotten.
    fn discard(&mut self, layer: &str);
    /// The layer named serves `kind` in place of the effect that began
    /// (same family): the record's request is what the innermost handler
    /// served.
    fn patch(&mut self, kind: &EffectKind);
}

/// What a unary consumer, the record, or a layer's verdict takes from a
/// stream: its outcome, the response the stream ends with or the first
/// error item. The origin folded the response; the tap only watches for it.
#[derive(Default)]
pub struct StreamTap {
    /// The tap yielded its outcome.
    finished: bool,
}

impl StreamTap {
    /// A new stream tap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the stream's outcome once: the response it ends with, or the
    /// first error item. Every later call returns `None`.
    pub fn observe(
        &mut self,
        item: &Result<Relayed, ErrorReport>,
    ) -> Option<Result<Answered, ErrorReport>> {
        if self.finished {
            return None;
        }
        let outcome = match item {
            Ok(Relayed::Origin(_) | Relayed::Item(_)) => None,
            Ok(Relayed::Done(response)) => Some(Ok(Answered::Completion(Arc::clone(response)))),
            Err(report) => Some(Err(report.clone())),
        };
        self.finished = outcome.is_some();
        outcome
    }
}

/// The retryable report of a reply that ended before the provider ended it
/// ([`ProviderError::Truncated`](crate::error::ProviderError::Truncated)).
/// Retryability does not guarantee a subsequent attempt will succeed.
pub fn stream_truncated() -> ErrorReport {
    ErrorReport::from(&crate::error::ProviderError::Truncated)
}

/// An answer, or an owned stream whose execution belongs to the driver.
pub enum Reply {
    /// The completed unary answer or a setup error.
    Outcome(Result<Answered, ErrorReport>),
    /// Events, including any frames after the first terminal record.
    Stream(StreamEvents),
}

impl Reply {
    /// Fold a stream to its first outcome, or return the unary answer.
    pub async fn into_outcome(self) -> Result<Answered, ErrorReport> {
        self.folded_outcome(None).await
    }

    pub(crate) async fn folded_outcome(
        self,
        folded: Option<Folded>,
    ) -> Result<Answered, ErrorReport> {
        match self {
            Self::Outcome(outcome) => outcome,
            Self::Stream(mut stream) => {
                let mut fold = StreamTap::new();
                while let Some(item) = stream.next().await {
                    let outcome = match &folded {
                        Some(folded) => lock(folded).take(),
                        None => fold.observe(&item),
                    };
                    if let Some(outcome) = outcome {
                        return outcome;
                    }
                }
                Err(stream_truncated())
            }
        }
    }

    /// Convert a completion into its events, then the response;
    /// incompatible outcomes become errors.
    pub fn into_stream(self) -> StreamEvents {
        match self {
            Self::Stream(stream) => stream,
            Self::Outcome(outcome) => Box::pin(futures::stream::iter(match outcome {
                Ok(Answered::Completion(response)) => {
                    match crate::operation::completion::events_of(&response) {
                        Ok(items) => std::iter::once(Ok(Relayed::Origin(response.origin.clone())))
                            .chain(items.into_iter().map(|item| Ok(Relayed::Item(item))))
                            .chain(std::iter::once(Ok(Relayed::Done(response))))
                            .collect(),
                        Err(error) => vec![Err(ErrorReport::from(&error))],
                    }
                }
                Ok(other) => vec![Err(wrong_stream_answer(&other))],
                Err(report) => vec![Err(report)],
            })),
        }
    }

    fn observed(self, streaming: bool, mut seen: Option<Observed>, folded: Option<Folded>) -> Self {
        if !streaming && let Self::Outcome(outcome) = self {
            if let Some(seen) = &mut seen {
                seen.outcome(outcome.clone());
            }
            return Self::Outcome(outcome);
        }
        if seen.is_none() && folded.is_none() {
            return if streaming {
                Self::Stream(self.into_stream())
            } else {
                self
            };
        }
        let mut original = match &self {
            Self::Outcome(Ok(Answered::Completion(response))) => {
                Some(Ok(Answered::Completion(response.clone())))
            }
            _ => None,
        };
        let mut stream = self.into_stream();
        let mut fold = StreamTap::new();
        let mut finished = false;
        Self::Stream(Box::pin(futures::stream::poll_fn(move |cx| {
            let item = match stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(item) => item,
            };
            if let Some(item) = &item {
                let mut outcome = if finished { None } else { fold.observe(item) };
                finished |= outcome.is_some();
                if let Some(seen) = &mut seen {
                    // A unary answer made a stream is recorded as answered;
                    // the fold's slot, when there is one, keeps the outcome.
                    let recorded = match &outcome {
                        None => None,
                        Some(_) if matches!(item, Ok(Relayed::Done(_))) && original.is_some() => {
                            original.take()
                        }
                        Some(_) if folded.is_none() => outcome.take(),
                        Some(outcome) => Some(outcome.clone()),
                    };
                    if streaming {
                        seen.item(item, recorded);
                    } else if let Some(recorded) = recorded {
                        seen.outcome(recorded);
                    }
                }
                if let (Some(outcome), Some(folded)) = (outcome, &folded) {
                    *lock(folded) = Some(outcome);
                }
            } else if !finished {
                finished = true;
                let outcome = Err(stream_truncated());
                if let Some(seen) = &mut seen {
                    seen.outcome(outcome.clone());
                }
                if let Some(folded) = &folded {
                    *lock(folded) = Some(outcome);
                }
            }
            Poll::Ready(item)
        })))
    }
}

// A layer and the recorder at its immediate inner boundary use the same fold.
// Each outer boundary gets its own slot: a verdict can change that view.
pub(crate) type Folded = Arc<Mutex<Option<Result<Answered, ErrorReport>>>>;

fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// An effect's identity, requested delivery mode, and driver-provided scopes.
/// Reply transport and cancellation remain owned by the driver.
pub struct Dispatch {
    adapter_context: Option<crate::observe::AdapterContext>,
    adapter_context_explicit: bool,
    id: EffectId,
    streaming: bool,
    scopes: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    observer: Option<Arc<Mutex<Option<Observed>>>>,
    /// How many handlers serve it now, one inside another: only the
    /// outermost tells the observer and the fold, once it answered.
    serving: Arc<AtomicUsize>,
    /// The layer whose verdict replaced the answer the consumer receives,
    /// once one did. Shared down the layer chain; a driver reads it after
    /// the reply ([`Dispatch::replaced_by`]).
    replaced_by: Arc<Mutex<Option<String>>>,
    folded: Option<Folded>,
}

/// A layer's handle for attributing its verdict once its inner handler
/// answered: [`Dispatch::attribution`].
#[derive(Clone)]
pub(crate) struct Attribution(Arc<Mutex<Option<String>>>);

impl Attribution {
    /// The layer named replaced the answer the consumer receives. The
    /// outermost layer to say so is the one the consumer's answer is from.
    pub(crate) fn replaced(&self, layer: &str) {
        *lock(&self.0) = Some(layer.to_owned());
    }
}

impl Dispatch {
    /// Construct the context for one effect.
    pub fn new(id: EffectId, streaming: bool) -> Self {
        Self {
            id,
            streaming,
            adapter_context: None,
            adapter_context_explicit: false,
            scopes: Vec::new(),
            observer: None,
            serving: Arc::default(),
            replaced_by: Arc::new(Mutex::new(None)),
            folded: None,
        }
    }

    /// The slot a layer's replacement verdict is named in: `None` until a
    /// layer replaced the answer on its way out, then that layer's name.
    /// The record keeps the handler's answer regardless; this says who the
    /// consumer's answer is from. Shared with every inner dispatch, so a
    /// driver reads it from the dispatch it built.
    pub fn replaced_by(&self) -> Arc<Mutex<Option<String>>> {
        self.replaced_by.clone()
    }

    /// The effect being served.
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// Whether the consumer requested streaming delivery.
    pub const fn is_stream(&self) -> bool {
        self.streaming
    }

    /// Attach a driver scope.
    pub fn with_scope(mut self, scope: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self.scopes.push(scope);
        self
    }

    /// Find a scope by its concrete type.
    pub fn scope<T: std::any::Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.scopes
            .iter()
            .find_map(|scope| Arc::downcast::<T>(scope.clone()).ok())
    }

    /// Copy the scope handles for an inline or tool dispatch.
    pub fn scopes(&self) -> Vec<Arc<dyn std::any::Any + Send + Sync>> {
        self.scopes.clone()
    }

    /// Observe the original handler answer independently of layer verdicts.
    /// A handler that serves the dispatch through an inner one, passing it
    /// on, is observed once it answered, with what it published by then;
    /// a [`Layer`](super::Layer) hands the observer to the handler it wraps.
    pub fn with_observer(mut self, observer: Box<dyn Observe>) -> Self {
        if !self.adapter_context_explicit {
            self.adapter_context = observer.adapter_context();
        }
        self.observer = Some(Arc::new(Mutex::new(Some(Observed {
            observer,
            told: false,
        }))));
        self
    }

    /// This dispatch observed by `recorder` under its id, as a recording
    /// driver installs it after [`begin`](super::Recorder::begin): its
    /// outcome, stream, origin and layer decisions, and the tool output its
    /// handler publishes into a [`PublishedContext`](crate::tool::PublishedContext)
    /// scope. Add the scopes first.
    pub fn recorded_by(self, recorder: Arc<dyn super::Recorder + Send + Sync>) -> Self {
        let observer = super::recorder::RecordingObserver {
            recorder,
            id: self.id,
            published: self.scope::<crate::tool::PublishedContext>(),
        };
        self.with_observer(Box::new(observer))
    }

    /// Supply provider context for this invocation independently of request data.
    ///
    /// Explicit context takes precedence over context supplied by an observer,
    /// regardless of installation order, and survives forwarding through layers.
    /// Reuse an operation context only for attempts of that same logical call.
    pub fn with_adapter_context(mut self, context: crate::observe::AdapterContext) -> Self {
        self.adapter_context = Some(context);
        self.adapter_context_explicit = true;
        self
    }

    /// Provider observation context forwarded across handler layers.
    pub fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        self.adapter_context.clone()
    }

    pub(crate) fn patched(&mut self, kind: &EffectKind) {
        if let Some(slot) = &self.observer
            && let Some(seen) = lock(slot).as_mut()
        {
            seen.observer.patch(kind);
        }
    }

    pub(crate) fn discard(&mut self, layer: &str) {
        if let Some(slot) = &self.observer
            && let Some(mut seen) = lock(slot).take()
        {
            seen.told = true;
            seen.observer.discard(layer);
        }
    }

    /// The handle a layer keeps to attribute its verdict once its inner
    /// handler answered (the observer itself moves inward with
    /// [`Self::inner`]).
    pub(crate) fn attribution(&self) -> Attribution {
        Attribution(self.replaced_by.clone())
    }

    pub(crate) fn inner(&mut self, folded: Option<Folded>) -> Self {
        let observer = self.observer.as_ref().and_then(|slot| lock(slot).take());
        Self {
            id: self.id,
            streaming: self.streaming,
            adapter_context: self.adapter_context.clone(),
            adapter_context_explicit: self.adapter_context_explicit,
            scopes: self.scopes.clone(),
            observer: observer.map(|seen| Arc::new(Mutex::new(Some(seen)))),
            serving: Arc::default(),
            replaced_by: self.replaced_by.clone(),
            folded,
        }
    }
}

struct Observed {
    observer: Box<dyn Observe>,
    told: bool,
}

impl Observed {
    fn outcome(&mut self, outcome: Result<Answered, ErrorReport>) {
        if !self.told {
            self.told = true;
            self.observer.outcome(outcome);
        }
    }

    fn item(
        &mut self,
        item: &Result<Relayed, ErrorReport>,
        outcome: Option<Result<Answered, ErrorReport>>,
    ) {
        let outcome = outcome.filter(|_| !self.told);
        self.told |= outcome.is_some();
        self.observer.stream_item(item, outcome);
    }
}

impl Drop for Observed {
    fn drop(&mut self) {
        self.outcome(Err(cancelled()));
    }
}

fn wrong_stream_answer(other: &Answered) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "a streaming dispatch was answered with a {} outcome",
            other.family()
        ),
    )
}

/// The consumer stopped listening before an answer was observed.
pub fn cancelled() -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Cancelled,
        "the consumer cancelled the dispatch before it was answered",
    )
    .with_retryable(false)
}

/// The receiver of an external answer or writer has been dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the dispatch's consumer is gone")]
pub struct SinkClosed;

/// Serve an effect inline, using the same reply conversions as a driver.
pub async fn serve_inline(
    handler: &ErasedHandler,
    kind: EffectKind,
) -> Result<Answered, ErrorReport> {
    serve_inline_with(handler, kind, Vec::new()).await
}

/// Serve inline with driver scopes, including tool context and publication.
pub async fn serve_inline_with(
    handler: &ErasedHandler,
    kind: EffectKind,
    scopes: Vec<Arc<dyn std::any::Any + Send + Sync>>,
) -> Result<Answered, ErrorReport> {
    let mut dispatch = Dispatch::new(EffectId::from_raw(0), false);
    for scope in scopes {
        dispatch = dispatch.with_scope(scope);
    }
    handler.handle(kind, dispatch).await.into_outcome().await
}
