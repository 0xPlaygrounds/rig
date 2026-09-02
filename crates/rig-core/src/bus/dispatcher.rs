//! The client half of the bus: `Dispatcher`, and the `Pending`/`EffectStream`
//! values a dispatch returns.

use std::{
    collections::BTreeMap,
    fmt,
    pin::Pin,
    sync::{
        Arc, PoisonError, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::{Context, Poll},
};

use futures::{
    Stream,
    channel::{mpsc, oneshot},
};

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::StreamEvent,
};

use super::{ErasedHandler, Handler, OutcomeSink};

/// State shared between every `Dispatcher` clone and the driver.
pub(super) struct Shared {
    next_id: AtomicU64,
    /// The handler table. Registration writes it synchronously from either
    /// side — no control message, so a registration made while nobody is
    /// driving (an MCP reconcile, a sync `add_tool`) never waits on the
    /// driver — and the driver reads it when it serves a command.
    handlers: RwLock<BTreeMap<HandlerKey, ErasedHandler>>,
    /// Set by the driver's drop guard: every reply that comes back
    /// `Canceled` after this is `BusClosed`, not a handler defect.
    closed: AtomicBool,
}

impl Shared {
    pub(super) fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            handlers: RwLock::new(BTreeMap::new()),
            closed: AtomicBool::new(false),
        }
    }

    pub(super) fn mark_closed(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub(super) fn register(&self, key: HandlerKey, handler: ErasedHandler) {
        self.handlers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, handler);
    }

    pub(super) fn deregister(&self, key: &HandlerKey) -> bool {
        self.handlers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(key)
            .is_some()
    }

    pub(super) fn handler(&self, key: &HandlerKey) -> Option<ErasedHandler> {
        self.handlers
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    /// The descriptor of the handler under `key`, stamped with the key it
    /// is registered under: the registration is authoritative, a handler's
    /// self-declared key is only a default.
    pub(super) fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.handler(key).map(|handler| HandlerDescriptor {
            key: key.clone(),
            family: handler.descriptor().family,
        })
    }

    pub(super) fn keys(&self) -> Vec<HandlerKey> {
        self.handlers
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .cloned()
            .collect()
    }
}

/// One command on the channel: a dispatch and its reply half.
pub(super) struct Command {
    pub(super) id: EffectId,
    pub(super) key: HandlerKey,
    pub(super) kind: EffectKind,
    pub(super) reply: Reply,
    /// The tracing span current at dispatch: the handler runs inside it,
    /// so a provider's telemetry parents under the caller's span exactly
    /// as a direct call would.
    pub(super) span: tracing::Span,
    /// Resolves `Canceled` when the consumer drops its `Pending` /
    /// `EffectStream`: the driver races the handler against it, so a
    /// dropped dispatch drops its handler future (and the provider call or
    /// stream inside) the next time the driver is polled.
    pub(super) cancel: oneshot::Receiver<()>,
}

pub(super) enum Reply {
    Unary(oneshot::Sender<Result<Outcome, ErrorReport>>),
    Stream(mpsc::Sender<Result<StreamEvent, ErrorReport>>),
}

impl Reply {
    pub(super) fn into_sink(self, id: EffectId) -> OutcomeSink {
        match self {
            Self::Unary(sender) => OutcomeSink::unary(id, sender),
            Self::Stream(sender) => OutcomeSink::stream(id, sender),
        }
    }

    /// Answer without a handler (unknown key, closed bus).
    pub(super) fn fail(self, report: ErrorReport) {
        match self {
            Self::Unary(sender) => {
                let _ = sender.send(Err(report));
            }
            Self::Stream(mut sender) => {
                let _ = sender.try_send(Err(report));
            }
        }
    }
}

/// The erased half of the bus: sends effects, reads descriptors, registers
/// handlers on a live bus. `Clone + Send + Sync + 'static` on every target.
///
/// A dispatcher never blocks and never awaits: [`Dispatcher::dispatch`] and
/// [`Dispatcher::dispatch_stream`] return immediately, and the *first poll*
/// of the returned [`Pending`]/[`EffectStream`] performs the (possibly
/// back-pressured) send. A full command channel therefore lands its pressure
/// on the value being polled, never on the caller — a system that dispatches
/// from inside a frame cannot deadlock the app.
#[derive(Clone)]
pub struct Dispatcher {
    pub(super) tx: mpsc::Sender<Command>,
    pub(super) shared: Arc<Shared>,
    pub(super) stream_capacity: usize,
}

impl fmt::Debug for Dispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dispatcher")
            .field("closed", &self.shared.is_closed())
            .field("handlers", &self.shared.keys())
            .finish_non_exhaustive()
    }
}

impl Dispatcher {
    fn mint(&self) -> EffectId {
        EffectId::from_raw(self.shared.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Mint the id a later [`Dispatcher::dispatch_with_id`] will carry, so a
    /// hook can see the effect's identity before it is sent.
    pub fn mint_id(&self) -> EffectId {
        self.mint()
    }

    /// Dispatch a unary effect. The returned [`Pending`] resolves to the
    /// handler's outcome, or to `BusClosed` / `HandlerUnavailable`.
    ///
    /// A streaming kind (`Completion { stream: true }`) may be dispatched
    /// unary: the driver folds the handler's events and resolves the
    /// aggregated completion at `Final`.
    pub fn dispatch(&self, key: &HandlerKey, kind: EffectKind) -> Pending {
        self.dispatch_with_id(self.mint(), key, kind)
    }

    /// [`Dispatcher::dispatch`] under an id minted earlier with
    /// [`Dispatcher::mint_id`].
    pub fn dispatch_with_id(&self, id: EffectId, key: &HandlerKey, kind: EffectKind) -> Pending {
        let (reply, receiver) = oneshot::channel();
        let (cancel_guard, cancel) = oneshot::channel();
        Pending {
            id,
            state: PendingState::Sending {
                tx: self.tx.clone(),
                command: Some(Box::new(Command {
                    id,
                    key: key.clone(),
                    kind,
                    reply: Reply::Unary(reply),
                    span: tracing::Span::current(),
                    cancel,
                })),
            },
            receiver,
            shared: self.shared.clone(),
            _cancel_guard: cancel_guard,
        }
    }

    /// Dispatch a streaming effect. Legal only for kinds whose family
    /// streams — today `Completion { stream: true }` alone; a stream
    /// dispatch of a unary kind resolves as one failed item with an
    /// invalid-dispatch report and never reaches a handler.
    pub fn dispatch_stream(&self, key: &HandlerKey, kind: EffectKind) -> EffectStream {
        self.dispatch_stream_with_id(self.mint(), key, kind)
    }

    /// [`Dispatcher::dispatch_stream`] under an id minted earlier with
    /// [`Dispatcher::mint_id`].
    pub fn dispatch_stream_with_id(
        &self,
        id: EffectId,
        key: &HandlerKey,
        kind: EffectKind,
    ) -> EffectStream {
        if !kind.streams() {
            return EffectStream {
                _cancel_guard: None,
                id,
                state: StreamState::Failed(Some(ErrorReport::new(
                    ErrorKind::Request,
                    format!(
                        "invalid dispatch: `{}` is a unary effect and cannot be dispatched as a stream",
                        kind.name()
                    ),
                ))),
                shared: self.shared.clone(),
            };
        }
        let (events, receiver) = mpsc::channel(self.stream_capacity);
        let (cancel_guard, cancel) = oneshot::channel();
        EffectStream {
            id,
            state: StreamState::Sending {
                tx: self.tx.clone(),
                command: Some(Box::new(Command {
                    id,
                    key: key.clone(),
                    kind,
                    reply: Reply::Stream(events),
                    span: tracing::Span::current(),
                    cancel,
                })),
                receiver: Some(receiver),
            },
            shared: self.shared.clone(),
            _cancel_guard: Some(cancel_guard),
        }
    }

    /// The descriptor of the handler serving `key` — a snapshot of the
    /// handler table, no round trip. `None` when nothing serves the key.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.shared.descriptor(key)
    }

    /// Every registered key, in key order.
    pub fn keys(&self) -> Vec<HandlerKey> {
        self.shared.keys()
    }

    /// Register (or replace) the handler serving `key` on a live bus. Takes
    /// effect for the next dispatch; an in-flight dispatch keeps the handler
    /// it started with.
    pub fn register(&self, key: impl Into<HandlerKey>, handler: impl Handler + 'static) {
        self.shared
            .register(key.into(), ErasedHandler::new(handler));
    }

    /// Register an already-erased handler on a live bus.
    pub fn register_erased(&self, key: impl Into<HandlerKey>, handler: ErasedHandler) {
        self.shared.register(key.into(), handler);
    }

    /// Remove the handler serving `key`; later dispatches answer
    /// `HandlerUnavailable`. Returns whether a handler was registered.
    pub fn deregister(&self, key: &HandlerKey) -> bool {
        self.shared.deregister(key)
    }

    /// Whether the driver has been dropped. A dispatch on a closed bus
    /// resolves `BusClosed` on first poll.
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed() || self.tx.is_closed()
    }
}

fn bus_closed() -> ErrorReport {
    ErrorReport::new(ErrorKind::BusClosed, "the bus driver is gone").with_retryable(false)
}

pub(super) fn handler_unavailable(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("no handler serves key `{key}`"),
    )
    .with_retryable(false)
}

fn reply_dropped(shared: &Shared) -> ErrorReport {
    if shared.is_closed() {
        bus_closed()
    } else {
        ErrorReport::new(
            ErrorKind::Internal,
            "the handler dropped its outcome sink without answering",
        )
    }
}

enum PendingState {
    Sending {
        tx: mpsc::Sender<Command>,
        command: Option<Box<Command>>,
    },
    Waiting,
}

/// A unary dispatch in flight: a plain `Unpin` future with no executor
/// affinity, resolving to the outcome or a report. Dropping it cancels the
/// dispatch (the handler's sink reports closed).
pub struct Pending {
    id: EffectId,
    state: PendingState,
    receiver: oneshot::Receiver<Result<Outcome, ErrorReport>>,
    shared: Arc<Shared>,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: oneshot::Sender<()>,
}

impl Pending {
    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.id
    }
}

impl fmt::Debug for Pending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pending").field("id", &self.id).finish()
    }
}

impl Future for Pending {
    type Output = Result<Outcome, ErrorReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                PendingState::Sending { tx, command } => {
                    if this.shared.is_closed() {
                        return Poll::Ready(Err(bus_closed()));
                    }
                    match tx.poll_ready(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(_)) => return Poll::Ready(Err(bus_closed())),
                        Poll::Ready(Ok(())) => {}
                    }
                    let Some(command) = command.take() else {
                        return Poll::Ready(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a dispatch was sent twice",
                        )));
                    };
                    if tx.start_send(*command).is_err() {
                        return Poll::Ready(Err(bus_closed()));
                    }
                    this.state = PendingState::Waiting;
                }
                PendingState::Waiting => {
                    return match Pin::new(&mut this.receiver).poll(cx) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(Ok(outcome)) => Poll::Ready(outcome),
                        Poll::Ready(Err(oneshot::Canceled)) => {
                            Poll::Ready(Err(reply_dropped(&this.shared)))
                        }
                    };
                }
            }
        }
    }
}

enum StreamState {
    Sending {
        tx: mpsc::Sender<Command>,
        command: Option<Box<Command>>,
        receiver: Option<mpsc::Receiver<Result<StreamEvent, ErrorReport>>>,
    },
    Receiving {
        receiver: mpsc::Receiver<Result<StreamEvent, ErrorReport>>,
        saw_terminal: bool,
    },
    /// Rejected before any send (an invalid dispatch): yields the report once.
    Failed(Option<ErrorReport>),
    Done,
}

/// A streaming dispatch in flight: a plain `Unpin` stream of
/// `Result<StreamEvent, ErrorReport>`, `Final`-terminated. Dropping it
/// cancels the dispatch: the handler's next send fails and the provider
/// stream is dropped. Pause is client-side back-pressure — stop polling and
/// the bounded channel stalls the handler.
pub struct EffectStream {
    id: EffectId,
    state: StreamState,
    shared: Arc<Shared>,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: Option<oneshot::Sender<()>>,
}

impl EffectStream {
    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.id
    }
}

impl fmt::Debug for EffectStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectStream")
            .field("id", &self.id)
            .finish()
    }
}

impl Stream for EffectStream {
    type Item = Result<StreamEvent, ErrorReport>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                StreamState::Failed(report) => {
                    let report = report.take();
                    this.state = StreamState::Done;
                    return Poll::Ready(report.map(Err));
                }
                StreamState::Done => return Poll::Ready(None),
                StreamState::Sending {
                    tx,
                    command,
                    receiver,
                } => {
                    if this.shared.is_closed() {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(bus_closed())));
                    }
                    match tx.poll_ready(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Err(_)) => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(bus_closed())));
                        }
                        Poll::Ready(Ok(())) => {}
                    }
                    let (Some(command), Some(receiver)) = (command.take(), receiver.take()) else {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a stream dispatch was sent twice",
                        ))));
                    };
                    if tx.start_send(*command).is_err() {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(bus_closed())));
                    }
                    this.state = StreamState::Receiving {
                        receiver,
                        saw_terminal: false,
                    };
                }
                StreamState::Receiving {
                    receiver,
                    saw_terminal,
                } => {
                    return match Pin::new(receiver).poll_next(cx) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(Some(item)) => {
                            if matches!(item, Ok(StreamEvent::Final(_)) | Err(_)) {
                                *saw_terminal = true;
                            }
                            Poll::Ready(Some(item))
                        }
                        Poll::Ready(None) => {
                            // The handler dropped the sink. A provider stream
                            // that ends without its terminal record is the
                            // consumer's truncation rule to apply; only a bus
                            // that closed under the dispatch is reported here.
                            let terminated = *saw_terminal;
                            this.state = StreamState::Done;
                            if !terminated && this.shared.is_closed() {
                                Poll::Ready(Some(Err(bus_closed())))
                            } else {
                                Poll::Ready(None)
                            }
                        }
                    };
                }
            }
        }
    }
}

// The client half crosses threads on every target and polls anywhere.
const _: fn() = || {
    fn assert_dispatcher<T: Clone + Send + Sync + 'static>() {}
    fn assert_unpin<T: Unpin + 'static>() {}
    assert_dispatcher::<Dispatcher>();
    assert_unpin::<Pending>();
    assert_unpin::<EffectStream>();
};

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send<T: Send + 'static>() {}
    assert_send::<Pending>();
    assert_send::<EffectStream>();
};
