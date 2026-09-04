//! The client half of the bus: `Dispatcher`, and the `Pending`/`EffectStream`
//! values a dispatch returns.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    future::Future,
    pin::Pin,
    sync::{Arc, PoisonError, Weak},
    task::{Context, Poll, Waker},
};

use crate::sync::{
    Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use futures::{
    Stream,
    channel::{mpsc, oneshot},
    task::AtomicWaker,
};

use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::StreamEvent,
    tool::{PublishedContext, ToolContext},
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
    /// Causality: every dispatch in flight, by id, with the key it is served
    /// on and the dispatch it was made from. A nested dispatch carries its
    /// parent on the command, so a dispatch that would queue behind an
    /// ancestor on the same serial key is refused here (`is_reentrant`), and
    /// a cancelled dispatch reaches its descendants (`cancel_descendants`).
    /// What a thread id used to approximate, as data.
    causality: Mutex<Causality>,
    /// Set by the driver's drop guard: every reply that comes back
    /// `Canceled` after this is `BusClosed`, not a handler defect.
    closed: AtomicBool,
    /// Set by the driver once every `Dispatcher` has dropped and the buffer
    /// is empty: the driver will not drain again, so a `Pending` created
    /// before its dispatcher went and polled after answers `BusClosed` at
    /// once instead of waiting for the driver's last in-flight work to end.
    commands_closed: AtomicBool,
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
            causality: Mutex::new(Causality::default()),
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
            commands_closed: AtomicBool::new(false),
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

    /// The driver is gone: its handlers with it, so the descriptor table
    /// describes nothing any more and is cleared. Called after
    /// [`mark_closed`](Self::mark_closed).
    pub(super) fn driver_died(&self) {
        self.descriptors
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
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
        if self.is_reentrant(&command) {
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
                layers: descriptor.layers,
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

    /// A dispatch begins under the driver: its id, key and parent enter the
    /// causality table and the flag it returns is set when an ancestor is
    /// cancelled. `Err` when an ancestor was cancelled already: the check
    /// and the insert are one critical section, so a child beginning while
    /// its parent's cancel runs is either flagged by the cancel or refused
    /// here — never in flight unflagged.
    pub(super) fn begin_in_flight(
        &self,
        id: EffectId,
        key: HandlerKey,
        parent: Option<EffectId>,
    ) -> Result<Arc<CancelFlag>, ChainCancelled> {
        let mut causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if causality.chain_cancelled(parent) {
            return Err(ChainCancelled);
        }
        let flag = Arc::new(CancelFlag::default());
        causality.in_flight.insert(
            id,
            InFlightEntry {
                key,
                parent,
                cancelled: Arc::clone(&flag),
            },
        );
        Ok(flag)
    }

    /// A dispatch left the driver. Returns whether it had been cancelled
    /// from above or below — its consumer's, or an ancestor's, departure —
    /// in which case the driver sweeps the children it still holds
    /// (queued, or buffered: [`Shared::fail_buffered_children`]); those in
    /// flight were flagged by the cancel itself.
    pub(super) fn end_in_flight(&self, id: EffectId) -> bool {
        let mut causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        causality.in_flight.remove(&id);
        causality.cancelled.remove(&id)
    }

    /// Fail, as cancelled, every buffered command made from `id`: a child
    /// dispatched by a handler whose dispatch was cancelled before the
    /// driver took the child. The queue lock is held for the sweep only.
    pub(super) fn fail_buffered_children(&self, id: EffectId) {
        let orphans: Vec<Box<Command>> = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            let (orphans, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut queue.commands)
                .into_iter()
                .partition(|command| command.parent == Some(id));
            queue.commands = kept.into();
            orphans
        };
        for orphan in orphans {
            orphan.reply.fail(rig_core::serve::cancelled());
        }
    }

    /// The consumer of `id` is gone: every descendant in flight is flagged
    /// (and woken), and `id` and each of them are remembered as cancelled
    /// until they leave the driver, so a child that is still queued or
    /// buffered, or begins meanwhile, is refused by its parent's id.
    pub(super) fn cancel_descendants(&self, id: EffectId) {
        let mut causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let descendants: Vec<(EffectId, Arc<CancelFlag>)> = causality
            .in_flight
            .iter()
            .filter(|(child, _)| **child != id && causality.descends_from(**child, id))
            .map(|(child, entry)| (*child, Arc::clone(&entry.cancelled)))
            .collect();
        causality.cancelled.insert(id);
        causality
            .cancelled
            .extend(descendants.iter().map(|(child, _)| *child));
        let flags: Vec<Arc<CancelFlag>> = descendants.into_iter().map(|(_, flag)| flag).collect();
        drop(causality);
        for flag in flags {
            flag.set();
        }
    }

    /// Under serial serving, a dispatch that descends from a dispatch in
    /// flight on its own key would queue behind that ancestor and wait on
    /// itself; it is refused instead. The chain is walked by parent ids,
    /// so a nested dispatch made from a spawned task — invisible to a
    /// thread check — is refused too, not hung.
    fn is_reentrant(&self, command: &Command) -> bool {
        if !self.serial_per_handler {
            return false;
        }
        let causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut next = command.parent;
        let mut hops = 0usize;
        while let Some(id) = next {
            let Some(entry) = causality.in_flight.get(&id) else {
                break;
            };
            if entry.key == command.key {
                return true;
            }
            next = entry.parent;
            hops += 1;
            if hops > causality.in_flight.len() {
                break;
            }
        }
        false
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

/// The dispatches in flight and the cancels they descend from.
#[derive(Default)]
pub(super) struct Causality {
    in_flight: BTreeMap<EffectId, InFlightEntry>,
    /// Dispatches whose consumer is gone, kept while a descendant may still
    /// be queued behind them; cleared when the dispatch itself leaves.
    cancelled: BTreeSet<EffectId>,
}

/// The dispatch descends from one whose consumer is gone.
pub(super) struct ChainCancelled;

impl Causality {
    /// Whether a dispatch made from `parent` was cancelled from above. The
    /// parent's id suffices: a cancel remembers every descendant in flight
    /// along with the root, so a grandchild's parent is in the set too.
    fn chain_cancelled(&self, parent: Option<EffectId>) -> bool {
        parent.is_some_and(|parent| self.cancelled.contains(&parent))
    }

    fn descends_from(&self, mut id: EffectId, ancestor: EffectId) -> bool {
        let mut hops = 0usize;
        while let Some(entry) = self.in_flight.get(&id) {
            match entry.parent {
                Some(parent) if parent == ancestor => return true,
                Some(parent) => id = parent,
                None => return false,
            }
            hops += 1;
            if hops > self.in_flight.len() {
                return false;
            }
        }
        false
    }
}

struct InFlightEntry {
    key: HandlerKey,
    parent: Option<EffectId>,
    cancelled: Arc<CancelFlag>,
}

/// Set when an ancestor of the dispatch is cancelled; the serving future
/// polls it and drops the handler when it is, and the sink shares the
/// marker so the handler sees a closed sink and the record a cancellation.
/// (`std` atomics on purpose: the marker crosses into rig-core's sink, and
/// under loom the flag is data, not a protocol under test.)
#[derive(Default)]
pub(super) struct CancelFlag {
    set: std::sync::Arc<std::sync::atomic::AtomicBool>,
    waker: AtomicWaker,
}

impl CancelFlag {
    fn set(&self) {
        self.set.store(true, std::sync::atomic::Ordering::SeqCst);
        self.waker.wake();
    }

    pub(super) fn is_set(&self) -> bool {
        self.set.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The marker the sink shares.
    pub(super) fn marker(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        std::sync::Arc::clone(&self.set)
    }

    /// Resolves when the flag is set.
    pub(super) fn wait(self: &Arc<Self>) -> CancelWait {
        CancelWait(Arc::clone(self))
    }
}

pub(super) struct CancelWait(Arc<CancelFlag>);

impl Future for CancelWait {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0.is_set() {
            return Poll::Ready(());
        }
        self.0.waker.register(cx.waker());
        if self.0.is_set() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// One command on the channel: a dispatch and its reply half.
pub(super) struct Command {
    pub(super) id: EffectId,
    pub(super) key: HandlerKey,
    pub(super) kind: EffectKind,
    /// The dispatch this one was made from: a handler dispatching through
    /// its sink's dispatcher, or `None` for a consumer's own dispatch.
    pub(super) parent: Option<EffectId>,
    /// The scope of the program that made the dispatch, if its dispatcher
    /// was scoped ([`Dispatcher::scoped`]).
    pub(super) scope: Option<Arc<str>>,
    /// The context a tool call runs with, carried beside the effect (never
    /// in it) to the handler's sink ([`Dispatcher::dispatch_tool_with_id`]).
    pub(super) context: Option<ToolContext>,
    /// Where the tool's published values come back, beside the sink.
    pub(super) published: Option<Arc<PublishedContext>>,
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
    /// The dispatch every dispatch made through this value descends from:
    /// `None` for a consumer's dispatcher, the served dispatch's id for the
    /// one a handler reads off its sink ([`crate::SinkDispatch`]).
    pub(super) parent: Option<EffectId>,
    /// The scope every dispatch made through this value carries: a stable
    /// serde id of the run or agent dispatching (never a runtime handle),
    /// `None` until [`Dispatcher::scoped`] sets it. A handler's scoped
    /// dispatcher inherits the scope of the dispatch it serves.
    pub(super) scope: Option<Arc<str>>,
}

impl Dispatcher {
    pub(super) fn open(shared: Arc<Shared>, stream_capacity: usize) -> Self {
        shared.dispatcher_opened();
        Self {
            shared,
            stream_capacity,
            parent: None,
            scope: None,
        }
    }

    /// A dispatcher whose dispatches descend from `parent`: what a handler
    /// serving `parent` dispatches through.
    ///
    /// A consumer's dispatcher holds the bus open for commands; a handler's
    /// scoped one does not — the dispatch it serves does, while it is in
    /// flight. So the count of open dispatchers is the count of consumers',
    /// and a bus whose consumers are all gone closes for commands even with
    /// handlers in flight, exactly as before.
    pub(super) fn parented(
        shared: Arc<Shared>,
        stream_capacity: usize,
        parent: EffectId,
        scope: Option<Arc<str>>,
    ) -> Self {
        Self {
            shared,
            stream_capacity,
            parent: Some(parent),
            scope,
        }
    }

    /// The dispatch every dispatch made through this value descends from,
    /// if any.
    pub const fn parent(&self) -> Option<EffectId> {
        self.parent
    }

    /// A dispatcher whose every dispatch — and every handle bound from it,
    /// and every nested dispatch a handler makes while serving one —
    /// carries `scope`: the record's `scope`, a stable id of the program
    /// dispatching, so a log several programs write in one world reads per
    /// program. The id is the caller's (a run id, an agent name), never a
    /// runtime handle.
    pub fn scoped(&self, scope: impl Into<Arc<str>>) -> Self {
        let mut dispatcher = self.clone();
        dispatcher.scope = Some(scope.into());
        dispatcher
    }

    /// The scope every dispatch made through this value carries, if any.
    pub fn scope(&self) -> Option<&Arc<str>> {
        self.scope.as_ref()
    }
}

impl Clone for Dispatcher {
    fn clone(&self) -> Self {
        let mut dispatcher = match self.parent {
            None => Self::open(Arc::clone(&self.shared), self.stream_capacity),
            Some(parent) => {
                Self::parented(Arc::clone(&self.shared), self.stream_capacity, parent, None)
            }
        };
        dispatcher.scope = self.scope.clone();
        dispatcher
    }
}

impl Drop for Dispatcher {
    fn drop(&mut self) {
        // Only a consumer's dispatcher was counted (see `parented`).
        if self.parent.is_none() {
            self.shared.dispatcher_closed();
        }
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
        self.dispatch_in(id, key, kind, None)
    }

    /// A tool call under `context`: the context travels beside the effect
    /// to the handler's sink (never on the wire), and what the tool
    /// publishes comes back through [`Pending::published_context`] once
    /// the dispatch resolved.
    pub fn dispatch_tool_with_id(
        &self,
        id: EffectId,
        key: &HandlerKey,
        kind: EffectKind,
        context: ToolContext,
    ) -> Pending {
        self.dispatch_in(id, key, kind, Some(context))
    }

    fn dispatch_in(
        &self,
        id: EffectId,
        key: &HandlerKey,
        kind: EffectKind,
        context: Option<ToolContext>,
    ) -> Pending {
        let (reply, receiver) = oneshot::channel();
        let (cancel_guard, cancel) = oneshot::channel();
        let published = context.as_ref().map(|_| PublishedContext::new());
        Pending {
            id,
            parent: self.parent,
            state: PendingState::Sending {
                command: Some(Box::new(Command {
                    id,
                    key: key.clone(),
                    kind,
                    parent: self.parent,
                    scope: self.scope.clone(),
                    context,
                    published: published.clone(),
                    reply: Reply::Unary(reply),
                    span: tracing::Span::current(),
                    cancel,
                })),
            },
            receiver,
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            _cancel_guard: cancel_guard,
            published,
        }
    }

    /// A dispatch refused before any send — a typed request with no wire
    /// form — as a [`Pending`] that resolves `report` on its first poll. It
    /// never reaches the buffer, a handler or a recorder; the id is minted so
    /// a host's bookkeeping keys it like any dispatch.
    pub(crate) fn refused(&self, report: ErrorReport) -> Pending {
        let (_reply, receiver) = oneshot::channel();
        let (cancel_guard, _cancel) = oneshot::channel();
        Pending {
            id: self.mint_id(),
            parent: self.parent,
            state: PendingState::Failed(Some(Box::new(report))),
            receiver,
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            _cancel_guard: cancel_guard,
            published: None,
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
                parent: self.parent,
                state: StreamState::Failed(Some(ErrorReport::new(
                    ErrorKind::Request,
                    format!(
                        "invalid dispatch: `{}` is a unary effect and cannot be dispatched as a stream",
                        kind.name()
                    ),
                ))),
                shared: self.shared.clone(),
                parked: Arc::new(AtomicWaker::new()),
            };
        }
        let (events, receiver) = mpsc::channel(self.stream_capacity);
        let (cancel_guard, cancel) = oneshot::channel();
        EffectStream {
            id,
            parent: self.parent,
            state: StreamState::Sending {
                command: Some(Box::new(Command {
                    id,
                    key: key.clone(),
                    kind,
                    parent: self.parent,
                    scope: self.scope.clone(),
                    context: None,
                    published: None,
                    reply: Reply::Stream(events),
                    span: tracing::Span::current(),
                    cancel,
                })),
                receiver: Some(receiver),
            },
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
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
    /// The scene half of a bus: what a save stores.
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        self.shared.descriptors()
    }

    /// This bus's identity for as long as it lives: distinct from every
    /// other live bus in the process, the same for every clone and handle
    /// over this bus. Derived from the bus's allocation, so it is **not** a persistent
    /// identifier: a scene stores keys and descriptors, never a `BusId`. Its
    /// use is in-memory bookkeeping — `EffectId`s are minted per bus, so a
    /// host with two buses keys its map by `(BusId, EffectId)`.
    pub fn id(&self) -> BusId {
        self.shared.id()
    }

    /// Whether the driver has been dropped. A dispatch on a closed bus
    /// resolves `BusClosed` on first poll.
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
/// dispatch (the handler's sink reports closed). A host that ticks rather
/// than awaits does not hold one: it holds effects as entities
/// (`rig_ecs::bus`).
pub struct Pending {
    id: EffectId,
    /// The dispatch this one was made from, if a handler made it.
    parent: Option<EffectId>,
    state: PendingState,
    receiver: oneshot::Receiver<Result<Outcome, ErrorReport>>,
    shared: Arc<Shared>,
    /// This value's one slot in the bus's parked-sender list while it waits
    /// on a full buffer; holds the latest waker it was polled with.
    parked: Arc<AtomicWaker>,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: oneshot::Sender<()>,
    /// Where a tool call's published context comes back
    /// ([`Dispatcher::dispatch_tool_with_id`]); `None` for any other
    /// dispatch.
    published: Option<Arc<PublishedContext>>,
}

impl Pending {
    /// Where the tool's published context lands once this dispatch resolved
    /// (clone it before awaiting the dispatch): `Some` for a
    /// [`Dispatcher::dispatch_tool_with_id`], `None` otherwise.
    pub fn published_context(&self) -> Option<Arc<PublishedContext>> {
        self.published.clone()
    }

    /// The dispatch this one was made from: `Some` when a handler dispatched
    /// it through its sink's dispatcher, `None` for a consumer's own.
    pub const fn parent(&self) -> Option<EffectId> {
        self.parent
    }

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
    /// The dispatch this one was made from, if a handler made it.
    parent: Option<EffectId>,
    state: StreamState,
    shared: Arc<Shared>,
    /// This value's one slot in the parked-sender list (see [`Pending`]).
    parked: Arc<AtomicWaker>,
    /// Dropped with the value: the driver's cancel signal.
    _cancel_guard: Option<oneshot::Sender<()>>,
}

impl EffectStream {
    /// The dispatch this one was made from, if a handler made it.
    pub const fn parent(&self) -> Option<EffectId> {
        self.parent
    }

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
        size_of::<Dispatcher>() <= 48,
        "Dispatcher budget: 48 bytes (measured 48 natively: the shared half, the stream capacity, the parent, the scope)"
    );
    assert!(
        size_of::<Pending>() <= 80,
        "Pending budget: 80 bytes (measured 80 natively: one parked-sender slot, one parent, the published-context slot of a tool call)"
    );
    assert!(
        size_of::<EffectStream>() <= 168,
        "EffectStream budget: 168 bytes (measured 168 natively: one parked-sender slot, one parent)"
    );
};
