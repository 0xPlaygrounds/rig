//! The client half of the bus: `Dispatcher`, and the `Pending`/`EffectStream`
//! values a dispatch returns.

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    pin::Pin,
    sync::{
        Arc, Mutex, PoisonError, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
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

use super::OutcomeSink;

/// State shared between every `Dispatcher` clone, every `Registrar` and the
/// driver. Holds only `Send + Sync` data — serde descriptors, the command
/// queue, atomics — which is what makes `Dispatcher: Send + Sync` on every
/// target by construction; handlers never pass through here.
pub(super) struct Shared {
    next_id: AtomicU64,
    /// The descriptor table: what serves each key, as data. Registration
    /// writes it synchronously from either side — so a descriptor read or
    /// a typed bind made while nobody is driving (an MCP reconcile, a sync
    /// `add_tool`) never waits on the driver — while the handler itself
    /// travels to the driver, which owns the only handler table.
    descriptors: RwLock<BTreeMap<HandlerKey, HandlerDescriptor>>,
    /// The command queue: one bounded buffer for the whole bus. The bound is
    /// bus-wide on purpose — a per-sender channel would hand every
    /// `Dispatcher` clone (and every dispatch, if each cloned a sender) a
    /// guaranteed slot of its own, and `command_capacity` would bound
    /// nothing.
    queue: Mutex<CommandQueue>,
    /// Live `Dispatcher` clones. The driver ends when this reaches zero with
    /// nothing queued or in flight.
    dispatchers: AtomicUsize,
    /// Serial serving (one command in flight per key), copied from the
    /// config so a dispatch can refuse to queue behind itself.
    serial_per_handler: bool,
    /// The key whose handler the driver is polling *right now*, on which
    /// thread. A dispatch to that key made during that poll, on that thread,
    /// comes from inside the handler (a tool running a nested prompt); under
    /// serial serving it would queue behind the very command that waits on
    /// it, so it is refused instead of hung.
    serving: Mutex<Option<(HandlerKey, std::thread::ThreadId)>>,
    /// Set by the driver's drop guard: every reply that comes back
    /// `Canceled` after this is `BusClosed`, not a handler defect.
    closed: AtomicBool,
}

/// What became of an offered command.
pub(super) enum Enqueue {
    Sent,
    Parked(Box<Command>),
    Refused(Box<Command>),
}

/// The bounded command buffer and the wakers on either side of it.
struct CommandQueue {
    commands: VecDeque<Box<Command>>,
    capacity: usize,
    /// The driver's waker, refreshed on every driver poll; woken when a
    /// command is enqueued or the last dispatcher drops.
    driver: Option<Waker>,
    /// The wakers of `Pending`/`EffectStream` values parked at the send
    /// stage because the buffer was full; all woken when the driver drains.
    senders: Vec<Waker>,
}

impl Shared {
    pub(super) fn new(command_capacity: usize, serial_per_handler: bool) -> Self {
        Self {
            serial_per_handler,
            serving: Mutex::new(None),
            next_id: AtomicU64::new(1),
            descriptors: RwLock::new(BTreeMap::new()),
            queue: Mutex::new(CommandQueue {
                commands: VecDeque::new(),
                capacity: command_capacity.max(1),
                driver: None,
                senders: Vec::new(),
            }),
            dispatchers: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    pub(super) fn mark_closed(&self) {
        self.closed.store(true, Ordering::SeqCst);
        // Commands the driver never took fail now — their reply halves live
        // in this buffer, not in the driver, so nothing else would close
        // them — and parked senders wake to observe the close.
        let (commands, senders) = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            (
                std::mem::take(&mut queue.commands),
                std::mem::take(&mut queue.senders),
            )
        };
        for command in commands {
            command.reply.fail(bus_closed());
        }
        for waker in senders {
            waker.wake();
        }
    }

    /// Offer `command` to the buffer. A full buffer hands the command back
    /// (`Parked`) and parks `cx`'s waker until the driver drains; the caller
    /// keeps the command and retries when woken. A dispatch that would queue
    /// behind the handler that is making it is `Refused`.
    pub(super) fn enqueue(&self, command: Box<Command>, cx: &Context<'_>) -> Enqueue {
        if self.is_reentrant(&command.key) {
            return Enqueue::Refused(command);
        }
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if queue.commands.len() >= queue.capacity {
            let waker = cx.waker();
            if !queue.senders.iter().any(|parked| parked.will_wake(waker)) {
                queue.senders.push(waker.clone());
            }
            return Enqueue::Parked(command);
        }
        queue.commands.push_back(command);
        if let Some(driver) = queue.driver.take() {
            driver.wake();
        }
        Enqueue::Sent
    }

    /// Take every buffered command (the driver's side), registering `cx` as
    /// the waker to wake on the next enqueue, and release any parked sender.
    pub(super) fn drain(&self, cx: &Context<'_>) -> VecDeque<Box<Command>> {
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        let commands = std::mem::take(&mut queue.commands);
        match &mut queue.driver {
            Some(driver) if driver.will_wake(cx.waker()) => {}
            slot => *slot = Some(cx.waker().clone()),
        }
        let senders = if commands.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut queue.senders)
        };
        drop(queue);
        for waker in senders {
            waker.wake();
        }
        commands
    }

    /// Commands buffered and not yet taken by the driver.
    pub(super) fn buffered(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .commands
            .len()
    }

    pub(super) fn dispatcher_opened(&self) {
        self.dispatchers.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn dispatcher_closed(&self) {
        if self.dispatchers.fetch_sub(1, Ordering::SeqCst) == 1 {
            // The driver may be waiting for exactly this to end.
            self.wake_driver();
        }
    }

    pub(super) fn dispatchers(&self) -> usize {
        self.dispatchers.load(Ordering::SeqCst)
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Publish the descriptor of the handler that will serve `key`, stamped
    /// with the key it is registered under (the registration is
    /// authoritative; a handler's self-declared key is only a default). A
    /// replacement must keep the key's family: a bound handle checked its
    /// family at bind time, and that check stays true for its lifetime.
    pub(super) fn publish_descriptor(
        &self,
        key: HandlerKey,
        descriptor: HandlerDescriptor,
    ) -> Result<(), ErrorReport> {
        let family = descriptor.family.family();
        let mut descriptors = self
            .descriptors
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(current) = descriptors.get(&key) {
            let current_family = current.family.family();
            if current_family != family {
                return Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "key `{key}` serves the {current_family:?} family; a {family:?} handler cannot replace it"
                    ),
                )
                .with_retryable(false));
            }
        }
        descriptors.insert(
            key.clone(),
            HandlerDescriptor {
                key,
                family: descriptor.family,
            },
        );
        Ok(())
    }

    /// Retract the descriptor under `key`: later dispatches answer
    /// `HandlerUnavailable`. Returns whether one was published.
    pub(super) fn retract_descriptor(&self, key: &HandlerKey) -> bool {
        let removed = self
            .descriptors
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(key)
            .is_some();
        // Under serial serving the driver may hold commands queued for this
        // key; it drains them with `HandlerUnavailable` on its next poll.
        self.wake_driver();
        removed
    }

    fn wake_driver(&self) {
        let driver = self
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .driver
            .take();
        if let Some(driver) = driver {
            driver.wake();
        }
    }

    /// Mark (or clear) the key whose handler the driver is polling.
    pub(super) fn set_serving(&self, key: Option<HandlerKey>) {
        *self.serving.lock().unwrap_or_else(PoisonError::into_inner) =
            key.map(|key| (key, std::thread::current().id()));
    }

    fn is_reentrant(&self, key: &HandlerKey) -> bool {
        self.serial_per_handler
            && matches!(
                &*self.serving.lock().unwrap_or_else(PoisonError::into_inner),
                Some((serving, thread)) if serving == key && *thread == std::thread::current().id()
            )
    }

    /// The descriptor published under `key`.
    pub(super) fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.descriptors
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    pub(super) fn keys(&self) -> Vec<HandlerKey> {
        self.descriptors
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

/// The client half of the bus: sends effects, reads descriptors, binds typed
/// views. `Clone + Send + Sync + 'static` on every target **by
/// construction** — it holds serde data, channels and atomics, never a
/// handler; handlers are the [`Registrar`](super::Registrar)'s business.
///
/// A dispatcher never blocks and never awaits: [`Dispatcher::dispatch`] and
/// [`Dispatcher::dispatch_stream`] return immediately, and the *first poll*
/// of the returned [`Pending`]/[`EffectStream`] performs the (possibly
/// back-pressured) send. The command buffer is bounded **bus-wide** by
/// [`BusConfig::command_capacity`](super::BusConfig::command_capacity):
/// a full buffer lands its pressure on the value being polled, never on the
/// caller — a system that dispatches from inside a frame cannot deadlock the
/// app, and a burst of dispatches cannot grow the buffer past the bound.
pub struct Dispatcher {
    pub(super) shared: Arc<Shared>,
    pub(super) stream_capacity: usize,
}

impl Dispatcher {
    pub(super) fn open(shared: Arc<Shared>, stream_capacity: usize) -> Self {
        shared.dispatcher_opened();
        Self {
            shared,
            stream_capacity,
        }
    }
}

impl Clone for Dispatcher {
    fn clone(&self) -> Self {
        Self::open(Arc::clone(&self.shared), self.stream_capacity)
    }
}

impl Drop for Dispatcher {
    fn drop(&mut self) {
        self.shared.dispatcher_closed();
    }
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
    /// descriptor table, no round trip. `None` when nothing serves the key.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.shared.descriptor(key)
    }

    /// Every registered key, in key order.
    pub fn keys(&self) -> Vec<HandlerKey> {
        self.shared.keys()
    }

    /// Whether the driver has been dropped. A dispatch on a closed bus
    /// resolves `BusClosed` on first poll.
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }

    /// Commands buffered on the bus and not yet taken by the driver — at
    /// most [`BusConfig::command_capacity`](super::BusConfig::command_capacity).
    /// A dispatch that finds the buffer full parks at its send stage (its
    /// poll stays `Pending`) until the driver drains; the pressure is on the
    /// `Pending`/`EffectStream`, never on the caller.
    pub fn buffered(&self) -> usize {
        self.shared.buffered()
    }
}

fn bus_closed() -> ErrorReport {
    ErrorReport::new(ErrorKind::BusClosed, "the bus driver is gone").with_retryable(false)
}

fn reentrant(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Request,
        format!(
            "re-entrant dispatch: the handler serving `{key}` dispatched to its own key under serial serving and would wait on itself"
        ),
    )
    .with_retryable(false)
}

/// A stream that ended before its `Final`: the handler dropped its sink
/// mid-stream (the provider stream ended early, or the handler failed
/// without reporting).
pub(super) fn stream_truncated() -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Response,
        "the stream ended before its terminal record",
    )
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
    Sending { command: Option<Box<Command>> },
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
                PendingState::Sending { command } => {
                    if this.shared.is_closed() {
                        return Poll::Ready(Err(bus_closed()));
                    }
                    let Some(taken) = command.take() else {
                        return Poll::Ready(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a dispatch was sent twice",
                        )));
                    };
                    match this.shared.enqueue(taken, cx) {
                        Enqueue::Sent => this.state = PendingState::Waiting,
                        Enqueue::Parked(kept) => {
                            *command = Some(kept);
                            return Poll::Pending;
                        }
                        Enqueue::Refused(refused) => {
                            return Poll::Ready(Err(reentrant(&refused.key)));
                        }
                    }
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
                StreamState::Sending { command, receiver } => {
                    if this.shared.is_closed() {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(bus_closed())));
                    }
                    let Some(taken) = command.take() else {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a stream dispatch was sent twice",
                        ))));
                    };
                    match this.shared.enqueue(taken, cx) {
                        Enqueue::Sent => {}
                        Enqueue::Parked(kept) => {
                            *command = Some(kept);
                            return Poll::Pending;
                        }
                        Enqueue::Refused(refused) => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(reentrant(&refused.key))));
                        }
                    }
                    let Some(receiver) = receiver.take() else {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a stream dispatch was sent twice",
                        ))));
                    };
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
                            // The handler dropped the sink. After the terminal
                            // that is the normal end; before it, the stream
                            // was cut short — by the bus closing, or by a
                            // handler that ended without its `Final` — and
                            // the consumer is told so as one last item rather
                            // than left to infer it from silence.
                            let terminated = *saw_terminal;
                            this.state = StreamState::Done;
                            if terminated {
                                Poll::Ready(None)
                            } else if this.shared.is_closed() {
                                Poll::Ready(Some(Err(bus_closed())))
                            } else {
                                Poll::Ready(Some(Err(stream_truncated())))
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
