//! Handler dispatch, owned replies, and delivery observation.
//!
//! ```
//! use rig_core::{effect::EffectId, serve::Dispatch};
//!
//! let dispatch = Dispatch::new(EffectId::from_raw(1), false);
//! assert!(!dispatch.is_stream());
//! ```

use std::{
    sync::{Arc, Mutex},
    task::Poll,
};

use futures::{StreamExt, channel::oneshot};

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, Outcome},
    error::{ErrorKind, ErrorReport},
    operation::CompletionFold,
    streaming::{StreamEvent, StreamEvents},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
    wire::Fold,
};

#[cfg(test)]
mod tests;

/// Boxed handler reply future, `Send` on native targets and locally polled on WASM.
pub type HandlerFuture<'a> = WasmBoxedFuture<'a, Reply>;

/// A registered effect handler returning an outcome or an owned stream.
///
/// Provider and tool authors do not implement this directly: the adapters
/// in [`crate::serve::adapters`] wrap models, tools, memories and indexes
/// (`Model`, `Tool`, `ConversationMemory`, `VectorStoreIndex`). A
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
/// (a typed registration on the bus); a handler with no one
/// family names [`crate::effect::family::Dynamic`].
pub trait Serve: WasmCompatSend + WasmCompatSync {
    /// The family this handler serves, or `Dynamic`.
    type Family: crate::effect::Served;

    /// What this handler is: the family-keyed description a typed view
    /// checks at bind time and a scene serializes.
    fn descriptor(&self) -> HandlerDescriptor;

    /// Prepare an outcome or an owned stream for the driver to consume.
    fn serve(
        &self,
        kind: EffectKind,
        dispatch: Dispatch,
    ) -> impl Future<Output = Reply> + WasmCompatSend + use<'_, Self>;
}

/// Object-safe handler interface implemented for every [`Serve`].
pub(crate) trait Handler: WasmCompatSend + WasmCompatSync {
    fn descriptor(&self) -> HandlerDescriptor;
    fn handle(&self, kind: EffectKind, dispatch: Dispatch) -> HandlerFuture<'_>;
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
        Box::pin(async move {
            let reply = self.serve(kind, dispatch).await;
            let seen = observer.and_then(|slot| lock(&slot).take());
            reply.observed(streaming, seen, folded)
        })
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
}

/// Shared, type-erased handler. Clones retain the same handler allocation.
/// Implements `Send + Sync` on native targets, but not on WASM.
#[derive(Clone)]
pub struct ErasedHandler(ErasedInner);

#[cfg(not(target_family = "wasm"))]
type ErasedInner = Arc<dyn Handler + Send + Sync>;
#[cfg(target_family = "wasm")]
type ErasedInner = Arc<dyn Handler>;

impl ErasedHandler {
    /// Erase `handler`.
    pub fn new(handler: impl Serve + 'static) -> Self {
        Self(Arc::new(handler))
    }

    /// Wrap this handler in a [`Layer`](super::Layer): `intercept` sees
    /// every dispatch before this handler does and every answer after.
    /// `handler.layered(a).layered(b)` calls `b.before` first and `a.after` first.
    pub fn layered(self, intercept: impl super::Intercept) -> Self {
        Self::new(super::Layer::new(self, intercept))
    }

    /// What the erased handler is.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    /// Serve one effect: the driver's call, straight to the boxed handler.
    pub fn handle(&self, kind: EffectKind, dispatch: Dispatch) -> HandlerFuture<'_> {
        self.0.handle(kind, dispatch)
    }

    /// Whether two erased handlers are the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ErasedHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErasedHandler")
            .field("key", &self.0.descriptor().key)
            .finish_non_exhaustive()
    }
}

/// Forwards serving to the wrapped handler; re-erasure adds another boxed future.
impl Serve for ErasedHandler {
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        self.handle(kind, dispatch).await
    }
}

pub trait Observe: Send + Sync {
    /// Provider witness associated with this dispatch, independently of recording.
    fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        None
    }

    /// The completed handler response, or the fold of a streaming handler's
    /// events. A resolved response retains content that has no stream block,
    /// such as images emitted as unknown events.
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>);
    /// Whether streamed events are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One event pulled from the original handler stream. This is recording
    /// evidence, not acknowledgement that a consumer received the item.
    fn event(&mut self, event: &StreamEvent);
    /// An error item in a kept stream, including errors after `Final`.
    fn stream_error(&mut self, _error: &ErrorReport) {}
    /// Observe one stream item together with its first folded outcome, if any.
    /// Drivers that snapshot recording concurrently with cancellation can override
    /// this operation to make the item and its answer one observation boundary.
    fn stream_item(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
        outcome: Option<&Result<Outcome, ErrorReport>>,
    ) {
        if self.keep_events() {
            match item {
                Ok(event) => self.event(event),
                Err(error) => self.stream_error(error),
            }
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

/// The one fold of a stream into the completion a unary consumer, or the
/// record, holds: what a unary consumer runs over a streaming handler's
/// events, what the driver's observer runs over a streaming dispatch, what
/// a layer runs for its verdict. The events pass the completion sink first,
/// so a handler's stream folds the same whether its handler wrote it through
/// the sink or not.
pub struct StreamTap {
    sink: crate::operation::AdapterOutput,
    fold: CompletionFold,
}

impl Default for StreamTap {
    fn default() -> Self {
        Self {
            sink: crate::operation::AdapterOutput::new(),
            fold: CompletionFold::relayed(""),
        }
    }
}

impl StreamTap {
    /// An empty fold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds an event, returning an outcome on `Final` or an error item,
    /// including a defect the sink finds in the event. Callers decide
    /// whether to stop after an outcome.
    pub fn observe(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
    ) -> Option<Result<Outcome, ErrorReport>> {
        let event = match item {
            Err(report) => return Some(Err(report.clone())),
            Ok(event) => event.clone(),
        };
        self.sink.push(Ok(event));
        let canonical: Vec<_> = self.sink.drain().collect();
        for item in canonical {
            let event = match item {
                Ok(event) => event,
                Err(error) => return Some(Err(ErrorReport::from(&error))),
            };
            if let Err(error) = self.fold.absorb(&event) {
                return Some(Err(ErrorReport::from(&error)));
            }
            if matches!(event, StreamEvent::Final(_)) {
                // A relayed stream's reply is its terminal record; the tap saw
                // no transport.
                let reply = crate::wire::Reply {
                    provider: String::new(),
                    raw: serde_json::Value::Null,
                    provider_request_id: None,
                };
                return Some(
                    std::mem::take(self)
                        .fold
                        .finish(reply)
                        .map(Outcome::Completion)
                        .map_err(|error| ErrorReport::from(&error)),
                );
            }
        }
        None
    }
}

/// Returns a retryable response error for a stream that ended before its terminal
/// record. Retryability does not guarantee a subsequent attempt will succeed.
pub fn stream_truncated() -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Response,
        "the stream ended before its terminal record",
    )
    .with_retryable(true)
}

/// An answer, or an owned stream whose execution belongs to the driver.
pub enum Reply {
    /// The completed unary answer or a setup error.
    Outcome(Result<Outcome, ErrorReport>),
    /// Events, including any frames after the first terminal record.
    Stream(StreamEvents),
}

impl Reply {
    /// Fold a stream to its first outcome, or return the unary answer.
    pub async fn into_outcome(self) -> Result<Outcome, ErrorReport> {
        self.folded_outcome(None).await
    }

    pub(crate) async fn folded_outcome(
        self,
        folded: Option<Folded>,
    ) -> Result<Outcome, ErrorReport> {
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

    /// Convert a completion into events; incompatible outcomes become errors.
    pub fn into_stream(self) -> StreamEvents {
        match self {
            Self::Stream(stream) => stream,
            Self::Outcome(outcome) => Box::pin(futures::stream::iter(match outcome {
                Ok(Outcome::Completion(response)) => {
                    let mut out = crate::operation::AdapterOutput::new();
                    out.response(&response, crate::operation::ImagePart::Unknown);
                    // An item that failed to re-emit (an image that did not
                    // serialize) is delivered as the error it is, not dropped.
                    out.drain()
                        .map(|item| item.map_err(|error| ErrorReport::from(&error)))
                        .collect()
                }
                Ok(other) => vec![Err(wrong_stream_answer(&other))],
                Err(report) => vec![Err(report)],
            })),
        }
    }

    fn observed(self, streaming: bool, mut seen: Option<Observed>, folded: Option<Folded>) -> Self {
        if !streaming && let Self::Outcome(outcome) = self {
            if let Some(seen) = &mut seen {
                seen.outcome(&outcome);
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
        let original = match &self {
            Self::Outcome(Ok(Outcome::Completion(response))) => {
                Some(Ok(Outcome::Completion(response.clone())))
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
                let outcome = if finished { None } else { fold.observe(item) };
                if let Some(seen) = &mut seen {
                    let recorded = outcome.as_ref().map(|outcome| {
                        if matches!(item, Ok(StreamEvent::Final(_))) {
                            original.as_ref().unwrap_or(outcome)
                        } else {
                            outcome
                        }
                    });
                    if streaming {
                        seen.item(item, recorded);
                    } else if let Some(recorded) = recorded {
                        seen.outcome(recorded);
                    }
                }
                if let Some(outcome) = outcome {
                    finished = true;
                    if let Some(folded) = &folded {
                        *lock(folded) = Some(outcome);
                    }
                }
            } else if !finished {
                finished = true;
                let outcome = Err(stream_truncated());
                if let Some(seen) = &mut seen {
                    seen.outcome(&outcome);
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
pub(crate) type Folded = Arc<Mutex<Option<Result<Outcome, ErrorReport>>>>;

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
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if !self.told {
            self.told = true;
            self.observer.outcome(outcome);
        }
    }

    fn item(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
        outcome: Option<&Result<Outcome, ErrorReport>>,
    ) {
        let outcome = outcome.filter(|_| !self.told);
        self.told |= outcome.is_some();
        self.observer.stream_item(item, outcome);
    }
}

impl Drop for Observed {
    fn drop(&mut self) {
        self.outcome(&Err(cancelled()));
    }
}

fn wrong_stream_answer(other: &Outcome) -> ErrorReport {
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

/// A single-use answer owned by an external responder.
pub struct Resolver(oneshot::Sender<Result<Outcome, ErrorReport>>);

/// Return a resolver and the answer future a handler must await.
/// Dropping the resolver without answering reports the established
/// unanswered-handler error; dropping the future closes the resolver.
pub fn deferred() -> (
    Resolver,
    impl Future<Output = Result<Outcome, ErrorReport>> + Send + 'static,
) {
    let (sender, receiver) = oneshot::channel();
    (Resolver(sender), async move {
        receiver.await.unwrap_or_else(|_| {
            Err(ErrorReport::new(
                ErrorKind::Internal,
                "the handler dropped its outcome sink without answering",
            ))
        })
    })
}

impl Resolver {
    /// Answer once. A late answer is discarded and reports closure.
    pub fn resolve(self, outcome: Result<Outcome, ErrorReport>) -> Result<(), SinkClosed> {
        self.0.send(outcome).map_err(|_| SinkClosed)
    }

    /// Whether the handler stopped waiting for this answer.
    pub fn is_closed(&self) -> bool {
        self.0.is_canceled()
    }
}

/// Serve an effect inline, using the same reply conversions as a driver.
pub async fn serve_inline(
    handler: &ErasedHandler,
    kind: EffectKind,
) -> Result<Outcome, ErrorReport> {
    serve_inline_with(handler, kind, Vec::new()).await
}

/// Serve inline with driver scopes, including tool context and publication.
pub async fn serve_inline_with(
    handler: &ErasedHandler,
    kind: EffectKind,
    scopes: Vec<Arc<dyn std::any::Any + Send + Sync>>,
) -> Result<Outcome, ErrorReport> {
    let mut dispatch = Dispatch::new(EffectId::from_raw(0), false);
    for scope in scopes {
        dispatch = dispatch.with_scope(scope);
    }
    handler.handle(kind, dispatch).await.into_outcome().await
}
