//! The impl side of the bus: what a handler is and how it answers.

use std::{
    sync::Arc,
    task::{Context, Poll},
};

use futures::{SinkExt, channel::mpsc, channel::oneshot};

use crate::{
    completion::CompletionResponse,
    effect::{EffectId, EffectKind, HandlerDescriptor, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::{BlockAccumulator, StreamEvent, StreamFinal},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// The future the bus stores for a handler: boxed, because the driver's
/// table holds handlers as `Arc<dyn Handler>` (an in-flight task holds its
/// handler while the table is replaced) and the stored trait must be
/// dyn-compatible. `Send` on native (the `WasmBoxedFuture` fork), which is
/// what makes `BusDriver: Send`. Authors never see it: they implement
/// [`Serve`] with an `async fn`, and the one `Box::pin` is in the blanket
/// impl below.
pub type HandlerFuture<'a> = WasmBoxedFuture<'a, ()>;

/// Something registered on the bus that serves effects — the trait
/// handler authors implement, with an `async fn`.
///
/// Provider and tool authors do not implement this directly: the adapters
/// in [`crate::serve::adapters`] wrap the impl-side traits (`CompletionModel`,
/// `Tool`, `EmbeddingModel`, `ConversationMemory`, `VectorStoreIndex`). A
/// host implements it for out-of-tree kinds ([`EffectKind::Custom`], typed
/// through [`crate::effect::CustomEffect`]) or for a replayer.
///
/// A handler answers through the [`OutcomeSink`] it is given: a unary effect
/// resolves it once, a streaming effect feeds it [`StreamEvent`]s ending in
/// [`StreamEvent::Final`]. There is one sink type so a handler body cannot
/// answer on the wrong channel — the sink adapts the shape it receives to
/// the shape the dispatch asked for.
///
/// ```ignore
/// impl Serve for AskUser {
///     type Family = family::Custom<AskUserEffect>;
///     fn descriptor(&self) -> HandlerDescriptor {
///         self.descriptor.clone()
///     }
///     async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
///         let answer = self.ask(kind).await;
///         sink.resolve(Ok(Outcome::Custom(answer))).await;
///     }
/// }
/// ```
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

    /// Serve one effect. The future completes when the answer has been
    /// delivered (or the consumer went away — see [`OutcomeSink::send`]).
    fn serve(
        &self,
        kind: EffectKind,
        sink: OutcomeSink,
    ) -> impl Future<Output = ()> + WasmCompatSend + use<'_, Self>;
}

/// The dyn-compatible form the bus stores: the one erasure. Every [`Serve`]
/// is a `Handler` through the blanket impl, which is where the boxing
/// happens — once, here.
pub(crate) trait Handler: WasmCompatSend + WasmCompatSync {
    fn descriptor(&self) -> HandlerDescriptor;
    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_>;
}

// A type that is not a `Serve` should be told to implement `Serve`, never
// the crate-private `Handler` this blanket impl provides.
#[diagnostic::do_not_recommend]
impl<T: Serve> Handler for T {
    fn descriptor(&self) -> HandlerDescriptor {
        Serve::descriptor(self)
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(self.serve(kind, sink))
    }
}

/// A shared handler is a handler: `Arc<H>` forwards, so one handler can be
/// registered under several keys.
impl<H: Serve + ?Sized> Serve for Arc<H> {
    type Family = H::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        (**self).descriptor()
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        (**self).serve(kind, sink).await;
    }
}

/// A handler behind the bus's one erasure: what a registry stages until a
/// bus takes it, what a registrar carries to the
/// driver, what the driver's handler table holds.
///
/// On native this is `Arc<dyn Handler + Send + Sync>` (every handler is,
/// through the `WasmCompat*` supertraits), so it is `Clone + Send + Sync +
/// 'static`. On browser wasm the supertraits are no-op markers — a provider
/// client there is `!Send` — and so is this: `Arc<dyn Handler>`, `!Send`,
/// honestly. Nothing that must be `Send + Sync` on every target (the
/// dispatcher, the typed views) holds one.
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
    /// `handler.layered(a).layered(b)` puts `b` outermost — `b.before`
    /// first, `a.after` first.
    pub fn layered(self, intercept: impl super::Intercept) -> Self {
        Self::new(super::Layer::new(self, intercept))
    }

    /// What the erased handler is.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    /// Serve one effect: the driver's call, straight to the boxed handler.
    pub fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        self.0.handle(kind, sink)
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

/// An erased handler serves whatever it wraps: re-erasing one (nothing in
/// the tree does) forwards through one more box.
impl Serve for ErasedHandler {
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        self.0.handle(kind, sink).await;
    }
}

/// Serve one effect on `handler` right here, without a bus: the inline
/// path a standalone tool set or catalog uses. The bus is still the only
/// erasure — this is a direct call on the erased handler.
pub async fn serve_inline(
    handler: &ErasedHandler,
    kind: EffectKind,
) -> Result<Outcome, ErrorReport> {
    serve_inline_with(handler, kind, Vec::new()).await
}

/// [`serve_inline`] with `scopes` attached to the sink: the way an inline
/// tool call hands the tool its [`ToolContext`](crate::tool::ToolContext)
/// and the [`PublishedContext`](crate::tool::PublishedContext) it
/// publishes into, exactly as a driver would.
pub async fn serve_inline_with(
    handler: &ErasedHandler,
    kind: EffectKind,
    scopes: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
) -> Result<Outcome, ErrorReport> {
    let id = EffectId::from_raw(0);
    let (reply, receiver) = oneshot::channel();
    let mut sink = OutcomeSink::unary(id, reply);
    for scope in scopes {
        sink = sink.with_scope(scope);
    }
    handler.handle(kind, sink).await;
    match receiver.await {
        Ok(outcome) => outcome,
        Err(oneshot::Canceled) => Err(ErrorReport::new(
            ErrorKind::Internal,
            "the handler dropped its outcome sink without answering",
        )),
    }
}

/// The consumer dropped its pending dispatch or its effect stream: nobody
/// is listening. A streaming
/// handler stops on it — that is how cancellation reaches a provider stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkClosed;

impl std::fmt::Display for SinkClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the dispatch's consumer is gone")
    }
}

impl std::error::Error for SinkClosed {}

/// The reply half of one dispatch, handed to the handler by the driver.
///
/// `Send + Sync + 'static` on every target (asserted below): it holds the
/// reply channel, the fold state and the driver's observer, never the
/// handler.
/// A handler may therefore hand it out of its own future — see
/// [`OutcomeSink::detach`] — and answer from somewhere else: a Bevy system
/// with `World` access, a human at a console, a queue.
pub struct OutcomeSink {
    id: EffectId,
    inner: SinkInner,
    observer: Option<Observed>,
    /// Held until the sink answers or is dropped, whichever first; the
    /// driver's receiver resolves then. This is what keeps a detached
    /// sink's dispatch in flight — its serial slot, its `in_flight` count
    /// — after the handler future that detached it has returned.
    done: Option<oneshot::Sender<()>>,
    /// The driver's scopes for this dispatch: opaque values the driver
    /// attaches ([`with_scope`]) and a runtime crate or an adapter reads
    /// back by type — rig-agent's bus hands a `Dispatcher` whose dispatches carry
    /// this dispatch's id as their parent; a tool call's driver hands the
    /// `ToolContext` the tool runs with and the [`PublishedContext`] it
    /// publishes into. rig-core names no runtime, so the slots are `Any`;
    /// a handler served inline or by a driver that attached none has none.
    ///
    /// [`with_scope`]: OutcomeSink::with_scope
    /// [`PublishedContext`]: crate::tool::PublishedContext
    scopes: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// Set by the driver when the dispatch is cancelled from above — its
    /// parent's consumer went away — so the sink is closed to the handler
    /// (`is_closed`) and a drop reports a cancellation, exactly as when the
    /// dispatch's own consumer left. Attached with [`with_cancel`].
    ///
    /// [`with_cancel`]: OutcomeSink::with_cancel
    cancelled: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// An [`OutcomeSink`] that has left its handler: the external-resolver seam.
///
/// A [`Serve`] impl that cannot answer inside its own future — the answer
/// needs `&mut World`, a person, another schedule — calls
/// [`OutcomeSink::detach`], hands the result to whoever will answer, and
/// returns. The driver keeps the dispatch in flight (serial slot, in-flight
/// count, recorder slot) until the detached sink answers or is dropped, so
/// a serial key is not served twice concurrently and the log's order is
/// the serve order. Cancellation reaches the resolver through
/// [`DetachedSink::is_closed`]: the consumer dropped its `Pending`, and an
/// answer will go nowhere.
///
/// ```ignore
/// impl Serve for WorldTool {
///     type Family = family::Tool;
///     fn descriptor(&self) -> HandlerDescriptor { self.descriptor.clone() }
///     async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
///         // Not answered here: a system with `Query` access answers next tick.
///         self.mailbox.lock().push((kind, sink.detach()));
///     }
/// }
/// ```
pub struct DetachedSink(OutcomeSink);

impl std::fmt::Debug for DetachedSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedSink")
            .field("id", &self.0.id)
            .field("stream", &self.0.is_stream())
            .field("closed", &self.0.is_closed())
            .finish()
    }
}

impl DetachedSink {
    /// The driver's scope of type `T`; see [`OutcomeSink::scope`].
    pub fn scope<T: std::any::Any + Send + Sync>(&self) -> Option<std::sync::Arc<T>> {
        self.0.scope::<T>()
    }

    /// Every scope the driver attached; see [`OutcomeSink::scopes`].
    pub fn scopes(&self) -> Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.0.scopes()
    }

    /// The dispatch this sink answers.
    pub const fn id(&self) -> EffectId {
        self.0.id()
    }

    /// Whether the dispatch asked for a stream.
    pub const fn is_stream(&self) -> bool {
        self.0.is_stream()
    }

    /// Whether the consumer is gone (its `Pending`/`EffectStream` dropped):
    /// an answer would be discarded, and the record says cancelled.
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Answer the dispatch: [`OutcomeSink::resolve`].
    pub fn resolve(self, outcome: Result<Outcome, ErrorReport>) -> HandlerFuture<'static> {
        self.0.resolve(outcome)
    }

    /// Feed one stream item: [`OutcomeSink::send`].
    pub async fn send(&mut self, item: Result<StreamEvent, ErrorReport>) -> Result<(), SinkClosed> {
        self.0.send(item).await
    }

    /// Stream through a writer: [`OutcomeSink::writer`].
    pub fn writer(self) -> super::StreamWriter {
        self.0.writer()
    }

    /// The sink back, for a resolver that has the handler-side API in hand.
    pub fn into_sink(self) -> OutcomeSink {
        self.0
    }
}

/// What a driver sees of one dispatch through the sink it handed out: the
/// record's view. A driver that is on the reply path itself (an ECS
/// schedule reading an outcome component) installs none; a driver that is
/// not (rig-agent's bus, whose consumer holds the reply channel) installs one per
/// dispatch with [`OutcomeSink::with_observer`], and it is told the
/// outcome exactly once — a stream's folded, at its terminal.
pub trait Observe: Send + Sync {
    /// The outcome, as the consumer receives it (a stream's folded).
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>);
    /// Whether streamed events are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One streamed event, as it is sent.
    fn event(&mut self, event: &StreamEvent);
    /// The dispatch was decided before any handler served it: the record
    /// it opened is forgotten.
    fn discard(&mut self);
    /// A layer serves `kind` in place of the effect that began (same
    /// family): the record's request is what the innermost handler served.
    fn patch(&mut self, kind: &EffectKind);
}

/// The driver's observer with the fold a streaming dispatch needs to tell
/// it the outcome once.
pub(crate) struct Observed {
    observer: Box<dyn Observe>,
    /// The fold of a streaming dispatch's events into the outcome the
    /// observer is told. A unary dispatch folds in its own arm and tells
    /// the observer what it resolved, so this stays empty there.
    stream: StreamTap,
    told: bool,
}

impl Observed {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if !self.told {
            self.told = true;
            self.observer.outcome(outcome);
        }
    }
}

/// A streaming dispatch answered with a non-completion outcome: what the
/// consumer receives, and what the observer records.
fn wrong_stream_answer(other: &Outcome) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "a streaming dispatch was answered with a {} outcome",
            other.family()
        ),
    )
}

impl Drop for OutcomeSink {
    fn drop(&mut self) {
        // Answered or not, the dispatch is over for the driver: dropping
        // the sender resolves the driver's receiver.
        self.done = None;
        // A sink dropped before it answered is a dispatch the consumer sees
        // fail — a stream cut short before its `Final`, a unary handler
        // that never resolved — and the log records the same failure the
        // consumer receives rather than losing the dispatch. One case is
        // not a failure of the handler: the consumer dropped its `Pending`
        // or `EffectStream`, the driver dropped the handler future, and the
        // sink went with it. That is a cancellation, and the record says so
        // — a replay of the log must not answer it as a provider failure.
        let unanswered = match &self.inner {
            SinkInner::Unary { reply, .. } => reply.is_some(),
            SinkInner::Stream { finished, .. } => !*finished,
        };
        if unanswered && self.observer.as_ref().is_some_and(|seen| !seen.told) {
            let report = if self.is_closed() {
                cancelled()
            } else {
                match &self.inner {
                    SinkInner::Unary { .. } => ErrorReport::new(
                        ErrorKind::Internal,
                        "the handler dropped its outcome sink without answering",
                    ),
                    SinkInner::Stream { .. } => stream_truncated(),
                }
            };
            self.tell_outcome(&Err(report));
        }
        // Cancelled from above with a consumer still listening (a child whose
        // `Pending` outlived its parent's handler): that consumer is told the
        // dispatch was cancelled, not that a handler misbehaved.
        let cancelled_from_above = self
            .cancelled
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::SeqCst));
        if unanswered && cancelled_from_above {
            match &mut self.inner {
                SinkInner::Unary { reply, .. } => {
                    if let Some(reply) = reply.take() {
                        let _ = reply.send(Err(cancelled()));
                    }
                }
                SinkInner::Stream { events, .. } => {
                    let _ = events.try_send(Err(cancelled()));
                }
            }
        }
    }
}

/// The report a dispatch resolves to in the record when its consumer went
/// away before the handler answered.
pub fn cancelled() -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Cancelled,
        "the consumer cancelled the dispatch before it was answered",
    )
    .with_retryable(false)
}

#[allow(
    clippy::large_enum_variant,
    reason = "one sink per dispatch, moved into the handler once; the unary arm carries the fold state"
)]
enum SinkInner {
    /// A unary dispatch. A streaming handler answering here is folded —
    /// the one fold, [`StreamTap`] — and resolved at `Final`.
    Unary {
        reply: Option<oneshot::Sender<Result<Outcome, ErrorReport>>>,
        fold: StreamTap,
    },
    /// A streaming dispatch. A unary handler answering here has its
    /// completion re-emitted as events. `finished` is set once a unary
    /// answer was re-emitted, so a later `send` is refused.
    Stream {
        events: mpsc::Sender<Result<StreamEvent, ErrorReport>>,
        finished: bool,
    },
}

impl OutcomeSink {
    pub fn unary(id: EffectId, reply: oneshot::Sender<Result<Outcome, ErrorReport>>) -> Self {
        Self {
            id,
            inner: SinkInner::Unary {
                reply: Some(reply),
                fold: StreamTap::new(),
            },
            observer: None,
            done: None,
            scopes: Vec::new(),
            cancelled: None,
        }
    }

    pub fn stream(id: EffectId, events: mpsc::Sender<Result<StreamEvent, ErrorReport>>) -> Self {
        Self {
            id,
            inner: SinkInner::Stream {
                events,
                finished: false,
            },
            observer: None,
            done: None,
            scopes: Vec::new(),
            cancelled: None,
        }
    }

    /// Part of the driver seam: `done`'s receiver resolves when this sink
    /// has answered or been dropped — after the handler future that held
    /// it, if the sink was [detached](Self::detach). A driver keys the
    /// dispatch's lifetime on it, not on the handler future.
    pub fn with_done(mut self, done: oneshot::Sender<()>) -> Self {
        self.done = Some(done);
        self
    }

    /// Attach the driver's cancel marker (see the field): once set, the
    /// sink is closed.
    pub fn with_cancel(mut self, cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.cancelled = Some(cancelled);
        self
    }

    /// Attach one of the driver's scopes (see the field).
    pub fn with_scope(mut self, scope: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        self.scopes.push(scope);
        self
    }

    /// The driver's scope of type `T`, read back as the type the driver
    /// attached; `None` when no driver attached one or it is another
    /// runtime's.
    pub fn scope<T: std::any::Any + Send + Sync>(&self) -> Option<std::sync::Arc<T>> {
        self.scopes
            .iter()
            .find_map(|scope| std::sync::Arc::downcast::<T>(scope.clone()).ok())
    }

    /// Every scope the driver attached, untyped, for an adapter that passes
    /// them on (to a tool's [`ToolContext`](crate::tool::ToolContext)).
    pub fn scopes(&self) -> Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.scopes.clone()
    }

    /// Leave the handler: the dispatch stays in flight until the returned
    /// sink answers or is dropped. See [`DetachedSink`].
    pub fn detach(self) -> DetachedSink {
        DetachedSink(self)
    }

    /// The driver seam: install the driver's [`Observe`]r — the record's
    /// view of this dispatch, for a driver that is not itself on the reply
    /// path.
    pub fn with_observer(mut self, observer: Box<dyn Observe>) -> Self {
        self.observer = Some(Observed {
            observer,
            stream: StreamTap::new(),
            told: false,
        });
        self
    }

    /// The layer seam: a layer serves `kind` in place of what began, so the
    /// record's request is what the handler beneath will see.
    pub(crate) fn patched(&mut self, kind: &EffectKind) {
        if let Some(seen) = &mut self.observer {
            seen.observer.patch(kind);
        }
    }

    /// The layer seam: the observer moves to the innermost hop, so the
    /// record holds what the handler answered, never a layer's verdict.
    pub(crate) fn take_observer(&mut self) -> Option<Observed> {
        self.observer.take()
    }

    pub(crate) fn with_observer_slot(mut self, observer: Option<Observed>) -> Self {
        self.observer = observer;
        self
    }

    /// The layer seam: an inner sink carries the outer's scope and cancel
    /// marker, so a nested dispatch from the inner handler descends from the
    /// same dispatch and a cancel from above reaches it.
    pub(crate) fn inheriting(mut self, outer: &Self) -> Self {
        self.scopes = outer.scopes.clone();
        self.cancelled = outer.cancelled.clone();
        self
    }

    /// The layer seam: the dispatch was decided before any handler served
    /// it, so it is no record — the observer is told and dropped; what the
    /// sink then resolves reaches the consumer only.
    pub(crate) fn discard(&mut self) {
        if let Some(mut seen) = self.observer.take() {
            seen.observer.discard();
        }
    }

    fn tell_outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if let Some(seen) = &mut self.observer {
            seen.outcome(outcome);
        }
    }

    /// A streaming dispatch's item, seen by the observer: the event
    /// verbatim when it keeps them, and the fold's outcome at the terminal.
    fn tell_item(&mut self, item: &Result<StreamEvent, ErrorReport>) {
        let Some(seen) = &mut self.observer else {
            return;
        };
        if let Ok(event) = item
            && seen.observer.keep_events()
        {
            seen.observer.event(event);
        }
        if let Some(outcome) = seen.stream.observe(item) {
            seen.outcome(&outcome);
        }
    }

    /// The dispatch this sink answers.
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// Whether the dispatch asked for a stream (`dispatch_stream`).
    pub const fn is_stream(&self) -> bool {
        matches!(self.inner, SinkInner::Stream { .. })
    }

    /// Whether the consumer is still listening.
    pub fn is_closed(&self) -> bool {
        let cancelled_from_above = self
            .cancelled
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::SeqCst));
        cancelled_from_above
            || match &self.inner {
                SinkInner::Unary { reply, .. } => reply.as_ref().is_none_or(|r| r.is_canceled()),
                SinkInner::Stream { events, finished } => *finished || events.is_closed(),
            }
    }

    /// Resolve a unary dispatch. On a streaming dispatch a completion is
    /// re-emitted as its events followed by `Final`; any other outcome, or an
    /// error, is delivered as the stream's one item.
    pub fn resolve(mut self, outcome: Result<Outcome, ErrorReport>) -> HandlerFuture<'static> {
        Box::pin(async move {
            // What the observer records is what the consumer receives: a stream
            // dispatch answered with a non-completion outcome is delivered
            // as an error, and recorded as that error.
            let delivered: Result<Outcome, ErrorReport> = match (&self.inner, &outcome) {
                (SinkInner::Stream { .. }, Ok(other))
                    if !matches!(other, Outcome::Completion(_)) =>
                {
                    Err(ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "a streaming dispatch was answered with a {} outcome",
                            other.family()
                        ),
                    ))
                }
                _ => outcome.clone(),
            };
            self.tell_outcome(&delivered);
            match &mut self.inner {
                SinkInner::Unary { reply, .. } => {
                    if let Some(reply) = reply.take() {
                        // A dropped receiver is the consumer cancelling; nothing to do.
                        let _ = reply.send(outcome);
                    }
                }
                SinkInner::Stream { events, finished } => {
                    if *finished {
                        return;
                    }
                    match delivered {
                        Ok(Outcome::Completion(response)) => {
                            for item in events_from_response(&response) {
                                if events.send(item).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Ok(other) => {
                            let _ = events.send(Err(wrong_stream_answer(&other))).await;
                        }
                        Err(report) => {
                            let _ = events.send(Err(report)).await;
                        }
                    }
                    *finished = true;
                }
            }
        })
    }

    /// Feed one stream item. On a unary dispatch the item is folded into the
    /// accumulator and `Final` resolves the dispatch with the aggregated
    /// [`CompletionResponse`]; an error resolves it with the report.
    ///
    /// `Err(SinkClosed)` means the consumer is gone: stop producing.
    pub async fn send(&mut self, item: Result<StreamEvent, ErrorReport>) -> Result<(), SinkClosed> {
        // A finished stream sink takes nothing more, and the observer sees only
        // what the consumer could receive.
        if let SinkInner::Stream { finished: true, .. } = &self.inner {
            return Err(SinkClosed);
        }
        if self.is_stream() {
            self.tell_item(&item);
        }
        match &mut self.inner {
            SinkInner::Stream { events, .. } => {
                // `Final` is not the end of the channel: a wire may still
                // deliver frames after its terminal record (a late message
                // id, a provider error), and the consumer's post-final rules
                // are its own. The stream ends when the handler drops the
                // sink.
                events.send(item).await.map_err(|_| SinkClosed)
            }
            SinkInner::Unary { reply, fold } => {
                let Some(sender) = reply.as_ref() else {
                    return Err(SinkClosed);
                };
                if sender.is_canceled() {
                    *reply = None;
                    return Err(SinkClosed);
                }
                // The one fold: the terminal, or an error, or a fold failure
                // resolves the dispatch; anything else is folded and taken.
                let Some(outcome) = fold.observe(&item) else {
                    return Ok(());
                };
                if let Some(seen) = &mut self.observer {
                    seen.outcome(&outcome);
                }
                if let SinkInner::Unary { reply, .. } = &mut self.inner
                    && let Some(reply) = reply.take()
                {
                    let _ = reply.send(outcome);
                }
                Ok(())
            }
        }
    }

    /// Wait until the consumer can take another item without the driver
    /// stalling — the back-pressure point a streaming handler may poll
    /// explicitly. Resolves immediately on a unary dispatch.
    pub fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), SinkClosed>> {
        match &mut self.inner {
            SinkInner::Stream { events, .. } => events.poll_ready(cx).map_err(|_| SinkClosed),
            SinkInner::Unary { .. } => Poll::Ready(Ok(())),
        }
    }
}

fn finish_unary(
    accumulator: &mut BlockAccumulator,
    message_id: Option<String>,
    terminal: StreamFinal,
) -> Result<Outcome, ErrorReport> {
    let choice = std::mem::replace(accumulator, BlockAccumulator::new()).finish();
    let mut response = CompletionResponse::new(choice, terminal.usage, terminal.provider.clone())
        .with_optional_finish_reason(terminal.finish_reason.clone());
    response.message_id = message_id.or(terminal.message_id.clone());
    response.response_id = terminal.response_id.clone();
    response.provider_request_id = terminal.provider_request_id.clone();
    response.model = terminal.model.clone();
    response.raw = terminal.raw;
    Ok(Outcome::Completion(response))
}

/// Re-emit a completed response as the events a stream consumer expects:
/// one block per content item, then `Final`. Used when a unary answer meets
/// a streaming dispatch (a replayed log, a unary-only custom handler).
pub(crate) fn events_from_response(
    response: &CompletionResponse,
) -> Vec<Result<StreamEvent, ErrorReport>> {
    use crate::{
        message::AssistantContent,
        providers::internal::adapter::AdapterOutput,
        streaming::{BlockId, MintKind, ToolCallEnd},
    };

    let mut out = AdapterOutput::new();
    if let Some(message_id) = &response.message_id {
        out.message_id(message_id.clone());
    }
    for (index, content) in response.choice.iter().enumerate() {
        let index = index as u64;
        match content {
            AssistantContent::Text(text) => {
                let id = BlockId::minted(MintKind::Text, index);
                out.text_start(id.clone(), text.additional_params.clone());
                out.text(text.text.clone());
                out.text_end(id);
            }
            AssistantContent::Reasoning(reasoning) => {
                let id = reasoning
                    .id
                    .as_deref()
                    .map(BlockId::wire)
                    .unwrap_or_else(|| BlockId::minted(MintKind::Reasoning, index));
                out.reasoning_end(id, Some(reasoning.clone()), None, true);
            }
            // Images never stream (no adapter emits one, the accumulator has
            // no block for one); a unary answer carrying an image reaches a
            // stream consumer as an unmodeled item, verbatim.
            AssistantContent::Image(image) => match serde_json::to_value(image) {
                Ok(value) => out.unknown(crate::streaming::UnknownPayload::new(value)),
                Err(error) => out.error(crate::completion::CompletionError::JsonError(error)),
            },
            AssistantContent::ToolCall(call) => {
                // The call's ids travel verbatim, so the accumulator publishes
                // the same durable id and the same `provider` the folded
                // response held: a provider's single id as the wire's tool
                // id, a dual (call id, item id) as both, and a minted
                // `tool-<n>` as the minted block it names — no wire id at
                // all, so no provider id is derived from it.
                let mut end =
                    ToolCallEnd::whole(call.function.name.clone(), call.function.arguments.clone())
                        .with_signature(call.signature.clone())
                        .with_additional_params(call.additional_params.clone());
                let id = match &call.provider {
                    Some(provider) => {
                        end = match &provider.item_id {
                            Some(item_id) => end
                                .with_call_id(provider.call_id.clone())
                                .with_tool_id(item_id.clone()),
                            None => end.with_tool_id(provider.call_id.clone()),
                        };
                        BlockId::wire(call.id.as_str())
                    }
                    None => BlockId::from_minted_name(call.id.as_str())
                        .unwrap_or_else(|| BlockId::wire(call.id.as_str())),
                };
                out.tool_call(id, end);
            }
        }
    }
    let mut terminal = StreamFinal::new(response.provider.clone(), response.usage)
        .with_optional_finish_reason(response.finish_reason());
    terminal.message_id = response.message_id.clone();
    terminal.response_id = response.response_id.clone();
    terminal.provider_request_id = response.provider_request_id.clone();
    terminal.model = response.model.clone();
    terminal.raw = response.raw.clone();
    out.final_record(terminal);
    // An item that failed to re-emit (an image that did not serialize) is
    // delivered as the error it is, not dropped.
    out.drain()
        .map(|item| item.map_err(|error| ErrorReport::from(&error)))
        .collect()
}

/// The one fold of a stream into the completion a unary consumer, or the
/// record, holds: what a unary sink runs over a streaming handler's
/// events, what the driver's observer runs over a streaming dispatch, what
/// a layer runs for its verdict.
#[derive(Default)]
pub struct StreamTap {
    accumulator: BlockAccumulator,
    message_id: Option<String>,
}

impl StreamTap {
    /// An empty fold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one event; returns the recorded outcome at the terminal.
    pub fn observe(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
    ) -> Option<Result<Outcome, ErrorReport>> {
        match item {
            Err(report) => Some(Err(report.clone())),
            Ok(StreamEvent::Final(terminal)) => Some(finish_unary(
                &mut self.accumulator,
                self.message_id.take(),
                terminal.clone(),
            )),
            Ok(event) => {
                if let StreamEvent::BlockStart {
                    id,
                    kind: crate::streaming::BlockKind::Message,
                } = event
                    && let Some(wire) = id.wire_str()
                {
                    self.message_id = Some(wire.to_owned());
                }
                if let Err(report) = self.accumulator.apply(event) {
                    return Some(Err(report));
                }
                None
            }
        }
    }
}

/// The report a stream that ended before its terminal record resolves to.
pub fn stream_truncated() -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Response,
        "the stream ended before its terminal record",
    )
}

// The sink crosses out of its handler (`detach`) and into whatever answers
// it — a Bevy system on another thread, natively — so it is `Send + Sync`
// on every target: reply channel, fold state and observer, never a handler.
const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<OutcomeSink>();
    assert_send_sync::<DetachedSink>();
};
