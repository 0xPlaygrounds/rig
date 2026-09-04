//! The client half of the bus: `Dispatcher`, and the `Pending`/`EffectStream`
//! values a dispatch returns.

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    pin::Pin,
    sync::{Arc, PoisonError, Weak},
    task::{Context, Poll, Waker},
};

use super::sync::{
    Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use futures::{
    Stream,
    channel::{mpsc, oneshot},
    task::{AtomicWaker, noop_waker_ref},
};

use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::StreamEvent,
};

use rig_core::serve::OutcomeSink;

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
    /// `Canceled` after this is `BusClosed`, not a handler defect. Cleared
    /// by [`Shared::reopen`].
    closed: AtomicBool,
    /// Whether a driver currently owns this bus. A driver's construction
    /// sets it, its drop clears it; `reopen` needs it clear.
    driver_alive: AtomicBool,
    /// Set by the driver once every `Dispatcher` has dropped and the buffer
    /// is empty: the driver will not drain again, so a `Pending` created
    /// before its dispatcher went and polled after answers `BusClosed` at
    /// once instead of waiting for the driver's last in-flight work to end.
    commands_closed: AtomicBool,
    /// Bumped by every `reopen`. A `Pending`/`EffectStream` remembers the
    /// generation it was dispatched under: one minted against a driver that
    /// has since died answers `BusClosed` even after a new driver took over
    /// — no in-flight work is resurrected across a restart.
    generation: AtomicU64,
    /// The config the bus was opened with; a reopened driver keeps it.
    config: rig_core::serve::ServingPolicy,
}

/// A bus's identity while it lives (see [`Dispatcher::id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BusId(u64);

impl BusId {
    /// The raw value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for BusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bus#{:x}", self.0)
    }
}

/// What became of an offered command.
pub(super) enum Enqueue {
    Sent,
    Parked(Box<Command>),
    Refused(Box<Command>),
    /// The driver is gone. Decided under the queue lock, so a command can
    /// never slip into the buffer after the close emptied it; the command is
    /// dropped (its reply half with it — the caller answers `BusClosed`).
    Closed,
}

/// The bounded command buffer and the wakers on either side of it.
struct CommandQueue {
    commands: VecDeque<Box<Command>>,
    capacity: usize,
    /// The driver's waker, refreshed on every driver poll; woken when a
    /// command is enqueued or the last dispatcher drops.
    driver: Option<Waker>,
    /// The `Pending`/`EffectStream` values parked at the send stage because
    /// the buffer was full — **one slot per value**, however many times it
    /// is polled while parked (a frame-ticked host polls it once per frame
    /// with a fresh waker each time; the value's `AtomicWaker` keeps only
    /// the latest). All woken when the driver drains; a slot whose value
    /// was dropped is skipped.
    senders: Vec<Weak<AtomicWaker>>,
}

impl Shared {
    pub(super) fn new(config: rig_core::serve::ServingPolicy) -> Self {
        Self {
            serial_per_handler: config.serial_per_handler,
            serving: Mutex::new(None),
            next_id: AtomicU64::new(1),
            descriptors: RwLock::new(BTreeMap::new()),
            queue: Mutex::new(CommandQueue {
                commands: VecDeque::new(),
                capacity: config.command_capacity.max(1),
                driver: None,
                senders: Vec::new(),
            }),
            dispatchers: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            driver_alive: AtomicBool::new(false),
            commands_closed: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            config,
        }
    }

    /// Close the bus for commands if nothing can send one any more: no
    /// dispatcher is open and nothing is buffered. Decided and stored
    /// **under the queue lock**, as `mark_closed` stores `closed`: `enqueue`
    /// reads the flag under the same lock, so a `Pending` that outlived its
    /// dispatcher cannot land a command in the buffer between this check and
    /// this store — which would have been a command the driver, already
    /// `Ready`, never takes. Returns whether the bus is now closed for
    /// commands.
    pub(super) fn try_close_commands(&self) -> bool {
        let queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if queue.commands.is_empty() && self.dispatchers() == 0 {
            self.commands_closed.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Whether the bus is closed for commands (the loom models' probe).
    #[cfg(rig_loom)]
    pub(super) fn commands_closed(&self) -> bool {
        self.commands_closed.load(Ordering::SeqCst)
    }

    /// A bus's identity while it lives: two buses in one process never
    /// share one, so a host keying its bookkeeping by `(BusId, EffectId)`
    /// never confuses two buses' effects.
    pub(super) fn id(self: &Arc<Self>) -> BusId {
        BusId(Arc::as_ptr(self) as usize as u64)
    }

    pub(super) fn descriptors(&self) -> Vec<HandlerDescriptor> {
        self.descriptors
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }

    pub(super) fn config(&self) -> rig_core::serve::ServingPolicy {
        self.config
    }

    /// A driver took (or is taking) the bus.
    pub(super) fn driver_born(&self) {
        self.driver_alive.store(true, Ordering::SeqCst);
    }

    /// The driver is gone: its handlers with it, so the descriptor table
    /// describes nothing any more and is cleared. Called after
    /// [`mark_closed`](Self::mark_closed).
    pub(super) fn driver_died(&self) {
        self.descriptors
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        self.driver_alive.store(false, Ordering::SeqCst);
    }

    /// Take the bus for a new driver, if no driver is alive: clears the
    /// close and moves to the next generation. `false` when a driver holds
    /// the bus.
    pub(super) fn reopen(&self) -> bool {
        if self
            .driver_alive
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        // Under the queue lock, as the close was: a dispatch's first poll
        // reads `closed` there.
        let _queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.closed.store(false, Ordering::SeqCst);
        self.commands_closed.store(false, Ordering::SeqCst);
        true
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    pub(super) fn mark_closed(&self) {
        // Commands the driver never took fail now — their reply halves live
        // in this buffer, not in the driver, so nothing else would close
        // them — and parked senders wake to observe the close. The flag is
        // set *under the queue lock*: `enqueue` reads it under the same
        // lock, so a dispatch whose first poll saw the bus open cannot land
        // its command in the buffer after this emptied it — which would
        // have been a dispatch nobody ever answers.
        let (commands, senders) = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            self.closed.store(true, Ordering::SeqCst);
            (
                std::mem::take(&mut queue.commands),
                std::mem::take(&mut queue.senders),
            )
        };
        for command in commands {
            command.reply.fail(bus_closed());
        }
        wake_parked(senders);
    }

    /// Offer `command` to the buffer. A full buffer hands the command back
    /// (`Parked`) and parks the caller — `parked` is the caller's one slot,
    /// which `cx`'s waker is stored in — until the driver drains; the
    /// caller keeps the command and retries when woken. A dispatch that
    /// would queue behind the handler that is making it is `Refused`.
    pub(super) fn enqueue(
        &self,
        command: Box<Command>,
        parked: &Arc<AtomicWaker>,
        cx: &Context<'_>,
    ) -> Enqueue {
        if self.is_reentrant(&command.key) {
            return Enqueue::Refused(command);
        }
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if self.closed.load(Ordering::SeqCst) || self.commands_closed.load(Ordering::SeqCst) {
            drop(command);
            return Enqueue::Closed;
        }
        if queue.commands.len() >= queue.capacity {
            parked.register(cx.waker());
            if !queue
                .senders
                .iter()
                .any(|slot| Weak::ptr_eq(slot, &Arc::downgrade(parked)))
            {
                queue.senders.push(Arc::downgrade(parked));
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
        wake_parked(senders);
        commands
    }

    /// Values parked at the send stage (test seam).
    #[cfg(test)]
    pub(super) fn parked_senders(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .senders
            .len()
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

/// Wake every parked value that is still alive.
fn wake_parked(senders: Vec<Weak<AtomicWaker>>) {
    for slot in senders {
        if let Some(parked) = slot.upgrade() {
            parked.wake();
        }
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
/// [`ServingPolicy::command_capacity`](rig_core::serve::ServingPolicy::command_capacity):
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
            parked: Arc::new(AtomicWaker::new()),
            generation: self.shared.generation(),
            _cancel_guard: cancel_guard,
        }
    }

    /// Dispatch a streaming effect. Legal only for kinds whose family
    /// streams — today `Completion { stream: true }` alone; a stream
    /// dispatch of a unary kind resolves as one failed item with an
    /// invalid-dispatch report and never reaches a handler.
    /// A dispatch refused before any send — a typed request with no wire
    /// form — as a [`Pending`] that resolves `report` on its first poll. It
    /// never reaches the buffer, a handler or a recorder; the id is minted so
    /// a host's bookkeeping keys it like any dispatch.
    pub(crate) fn refused(&self, report: ErrorReport) -> Pending {
        let (_reply, receiver) = oneshot::channel();
        let (cancel_guard, _cancel) = oneshot::channel();
        Pending {
            id: self.mint_id(),
            state: PendingState::Failed(Some(Box::new(report))),
            receiver,
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            generation: self.shared.generation(),
            _cancel_guard: cancel_guard,
        }
    }

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
                parked: Arc::new(AtomicWaker::new()),
                generation: self.shared.generation(),
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
            parked: Arc::new(AtomicWaker::new()),
            generation: self.shared.generation(),
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

    /// Every registered descriptor, in key order, as one snapshot under one
    /// lock — a registration made while a host iterates cannot tear it.
    /// The scene half of a bus: what a save stores and a load re-binds
    /// ([`Handle::rebind`](super::Handle::rebind)).
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        self.shared.descriptors()
    }

    /// This bus's identity for as long as it lives: distinct from every
    /// other live bus in the process, the same for every clone and handle
    /// over this bus, and stable across [`Bus::reopen`](super::Bus::reopen).
    /// Derived from the bus's allocation, so it is **not** a persistent
    /// identifier: a scene stores keys and descriptors, never a `BusId`. Its
    /// use is in-memory bookkeeping — `EffectId`s are minted per bus, so a
    /// host with two buses keys its map by `(BusId, EffectId)`.
    pub fn id(&self) -> BusId {
        self.shared.id()
    }

    /// Whether the driver has been dropped. A dispatch on a closed bus
    /// resolves `BusClosed` on first poll — until [`Bus::reopen`](super::Bus::reopen)
    /// gives the bus a new driver.
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }

    /// Commands buffered on the bus and not yet taken by the driver — at
    /// most [`ServingPolicy::command_capacity`](rig_core::serve::ServingPolicy::command_capacity).
    /// A dispatch that finds the buffer full parks at its send stage (its
    /// poll stays `Pending`) until the driver drains; the pressure is on the
    /// `Pending`/`EffectStream`, never on the caller.
    pub fn buffered(&self) -> usize {
        self.shared.buffered()
    }
}

pub(super) fn bus_closed() -> ErrorReport {
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
    rig_core::serve::stream_truncated()
}

pub(super) fn handler_unavailable(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("no handler serves key `{key}`"),
    )
    .with_retryable(false)
}

fn reply_dropped(shared: &Shared, generation: u64) -> ErrorReport {
    if shared.is_closed() || shared.generation() != generation {
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
        command: Option<Box<Command>>,
    },
    Waiting,
    /// Refused before any send: the request had no wire form
    /// ([`rig_core::effect::Family::wrap`] failed). Resolves the report on
    /// the first poll; nothing reaches a handler or a recorder. Boxed so
    /// the rare arm costs the common ones nothing (`Pending`'s budget).
    Failed(Option<Box<ErrorReport>>),
}

/// A unary dispatch in flight: a plain `Unpin` future with no executor
/// affinity, resolving to the outcome or a report. Dropping it cancels the
/// dispatch (the handler's sink reports closed).
///
/// A host that ticks rather than awaits — a Bevy system, once per frame —
/// probes it with [`Pending::poll_outcome`] instead of `block_on`: no
/// executor, no waker minted per frame.
pub struct Pending {
    id: EffectId,
    state: PendingState,
    receiver: oneshot::Receiver<Result<Outcome, ErrorReport>>,
    shared: Arc<Shared>,
    /// This value's one slot in the bus's parked-sender list while it waits
    /// on a full buffer; holds the latest waker it was polled with.
    parked: Arc<AtomicWaker>,
    /// The bus generation this dispatch was minted under (see
    /// [`Shared::reopen`]).
    generation: u64,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: oneshot::Sender<()>,
}

impl Pending {
    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// One poll, no executor: the outcome if the dispatch has resolved,
    /// `None` if not yet. The first call performs the send (or parks on a
    /// full buffer), exactly as the first `poll` would; a host calls it
    /// once per tick. Polling with a real waker and probing may be mixed —
    /// the bus keeps only the latest waker, and a probe's is a no-op, so a
    /// host that probes must keep probing.
    pub fn poll_outcome(&mut self) -> Option<Result<Outcome, ErrorReport>> {
        let mut cx = Context::from_waker(noop_waker_ref());
        match Pin::new(self).poll(&mut cx) {
            Poll::Ready(outcome) => Some(outcome),
            Poll::Pending => None,
        }
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
                    if this.shared.is_closed() || this.shared.generation() != this.generation {
                        return Poll::Ready(Err(bus_closed()));
                    }
                    let Some(taken) = command.take() else {
                        return Poll::Ready(Err(ErrorReport::new(
                            ErrorKind::Internal,
                            "a dispatch was sent twice",
                        )));
                    };
                    match this.shared.enqueue(taken, &this.parked, cx) {
                        Enqueue::Sent => this.state = PendingState::Waiting,
                        Enqueue::Parked(kept) => {
                            *command = Some(kept);
                            return Poll::Pending;
                        }
                        Enqueue::Refused(refused) => {
                            return Poll::Ready(Err(reentrant(&refused.key)));
                        }
                        Enqueue::Closed => return Poll::Ready(Err(bus_closed())),
                    }
                }
                PendingState::Failed(report) => {
                    return Poll::Ready(Err(report.take().map_or_else(
                        || {
                            ErrorReport::new(
                                ErrorKind::Internal,
                                "a refused dispatch was polled twice",
                            )
                        },
                        |report| *report,
                    )));
                }
                PendingState::Waiting => {
                    return match Pin::new(&mut this.receiver).poll(cx) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(Ok(outcome)) => Poll::Ready(outcome),
                        Poll::Ready(Err(oneshot::Canceled)) => {
                            Poll::Ready(Err(reply_dropped(&this.shared, this.generation)))
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
/// the bounded channel stalls the handler. A ticking host probes it with
/// [`EffectStream::poll_item`].
pub struct EffectStream {
    id: EffectId,
    state: StreamState,
    shared: Arc<Shared>,
    /// This value's one slot in the parked-sender list (see [`Pending`]).
    parked: Arc<AtomicWaker>,
    /// The bus generation this dispatch was minted under.
    generation: u64,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: Option<oneshot::Sender<()>>,
}

impl EffectStream {
    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// One poll, no executor: `Some(Some(item))` for the next item,
    /// `Some(None)` once the stream has ended, `None` if nothing is ready
    /// yet. The first call performs the send; a host calls it once per
    /// tick (or in a loop until `None`, to drain what a tick delivered).
    pub fn poll_item(&mut self) -> Option<Option<Result<StreamEvent, ErrorReport>>> {
        let mut cx = Context::from_waker(noop_waker_ref());
        match Pin::new(self).poll_next(&mut cx) {
            Poll::Ready(item) => Some(item),
            Poll::Pending => None,
        }
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
                    if this.shared.is_closed() || this.shared.generation() != this.generation {
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
                    match this.shared.enqueue(taken, &this.parked, cx) {
                        Enqueue::Sent => {}
                        Enqueue::Parked(kept) => {
                            *command = Some(kept);
                            return Poll::Pending;
                        }
                        Enqueue::Refused(refused) => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(reentrant(&refused.key))));
                        }
                        Enqueue::Closed => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(bus_closed())));
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
                            } else if this.shared.is_closed()
                                || this.shared.generation() != this.generation
                            {
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

// The client half crosses threads on every target and polls anywhere; the
// values a dispatch returns are small, plain futures — budgeted here so a
// field that grows one past its budget fails to compile with the budget in
// the message (raise a budget deliberately, with the reason in the commit).
const _: () = {
    const fn assert_dispatcher<T: Clone + Send + Sync + 'static>() {}
    const fn assert_unpin<T: Unpin + 'static>() {}
    // The values a dispatch returns are `Send` on every target — a Bevy
    // `Component` in the browser too — not only natively: they hold serde
    // data, channels and atomics, never a handler.
    const fn assert_send<T: Send + 'static>() {}
    assert_dispatcher::<Dispatcher>();
    assert_unpin::<Pending>();
    assert_unpin::<EffectStream>();
    assert_send::<Pending>();
    assert_send::<EffectStream>();
    assert!(
        size_of::<Dispatcher>() <= 32,
        "Dispatcher budget: 32 bytes (measured 16 natively)"
    );
    assert!(
        size_of::<Pending>() <= 64,
        "Pending budget: 64 bytes (measured 64 natively: one parked-sender slot, one generation)"
    );
    assert!(
        size_of::<EffectStream>() <= 160,
        "EffectStream budget: 160 bytes (measured 160 natively: one parked-sender slot, one generation)"
    );
};
