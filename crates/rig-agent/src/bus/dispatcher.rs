//! Bus dispatch, descriptor snapshots, and lazy unary or streaming replies.
//!
//! ```
//! use rig_agent::bus::Bus;
//! let (dispatcher, _registrar, _driver) = Bus::channel();
//! assert!(dispatcher.keys().is_empty());
//! ```

use std::{
    collections::{BTreeMap, VecDeque},
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

/// Shared descriptors, queue, and cancellation state. Contains no handlers so
/// dispatchers remain `Send + Sync` on every target.
pub(super) struct Shared {
    next_id: AtomicU64,
    /// Synchronously published descriptors; reads and typed binds do not require
    /// driver polling. Handlers reside only in the driver.
    descriptors: RwLock<BTreeMap<HandlerKey, HandlerDescriptor>>,
    /// Bus-wide bounded buffer. Cloning a dispatcher must not increase capacity.
    queue: Mutex<CommandQueue>,
    /// Live `Dispatcher` clones. The driver ends when this reaches zero with
    /// nothing queued or in flight.
    dispatchers: AtomicUsize,
    /// Serial serving (one command in flight per key), copied from the
    /// config so a dispatch can refuse to queue behind itself.
    serial_per_handler: bool,
    /// Active handler occupancy indexed by id. Each entry retains immutable
    /// ancestry; completed intermediates leave this table without severing
    /// cancellation or serial-reentrancy relationships.
    causality: Mutex<Causality>,
    /// Set by the driver's drop guard: every reply that comes back
    /// `Canceled` after this is `BusClosed`, not a handler defect.
    closed: AtomicBool,
    /// Prevents sends after the last consumer dispatcher drops and the queue
    /// empties, even while handlers remain in flight.
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
    Parked(Command),
    Refused(Command),
    Cancelled(Command),
    /// Driver is gone; the command was dropped under the queue's closure protocol.
    Closed,
}

/// The bounded command buffer and the wakers on either side of it.
struct CommandQueue {
    commands: VecDeque<Command>,
    capacity: usize,
    /// The driver's waker, refreshed on every driver poll; woken when a
    /// command is enqueued or the last dispatcher drops.
    driver: Option<Waker>,
    /// One weak waker slot per parked reply, updated on repoll. Live senders wake
    /// on every drain, closure, or chain cancellation.
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

    /// Close command submission when no consumer dispatcher or queued command
    /// remains. The queue lock prevents late sends from replies that outlived
    /// their dispatcher. Returns whether submission is now closed.
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
        // Close atomically with draining so no late send can escape failure.
        // Fail replies and wake senders outside the lock to allow reentrant callbacks.
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

    /// Offer a command, returning ownership when parked, cancelled, or refused.
    /// Parked callers retry after waking; serial dispatches behind an active
    /// ancestor serving the same key are refused.
    pub(super) fn enqueue(
        &self,
        command: Command,
        parked: &Arc<AtomicWaker>,
        cx: &Context<'_>,
    ) -> Enqueue {
        if command.lineage.is_cancelled() {
            return Enqueue::Cancelled(command);
        }
        if self.is_reentrant(&command) {
            return Enqueue::Refused(command);
        }
        // Registration can invoke an executor callback. Do it before locking;
        // the subsequent capacity check and sender insertion remain atomic.
        parked.register(cx.waker());
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        if self.closed.load(Ordering::SeqCst) || self.commands_closed.load(Ordering::SeqCst) {
            drop(queue);
            drop(command);
            return Enqueue::Closed;
        }
        if command.lineage.is_cancelled() {
            return Enqueue::Cancelled(command);
        }
        if queue.commands.len() >= queue.capacity {
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
        let driver = queue.driver.take();
        drop(queue);
        if let Some(driver) = driver {
            driver.wake();
        }
        Enqueue::Sent
    }

    /// Take all commands, register the driver waker, and wake parked senders.
    /// Senders wake even when cancellation already emptied the queue.
    pub(super) fn drain(&self, cx: &Context<'_>) -> VecDeque<Command> {
        // Raw waker clone/drop callbacks may reenter the dispatcher too.
        let next_driver = cx.waker().clone();
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        let commands = std::mem::take(&mut queue.commands);
        let previous_driver = queue.driver.replace(next_driver);
        let senders = std::mem::take(&mut queue.senders);
        drop(queue);
        drop(previous_driver);
        wake_parked(senders);
        commands
    }

    /// Values parked at the send stage (test seam).
    #[cfg(all(test, not(rig_loom)))]
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

    /// Publish a descriptor under the registration key, overriding its declared
    /// key. Returns `HandlerUnavailable` if an existing registration has a
    /// different family. Removal permits a subsequent change of family.
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
    /// `HandlerUnavailable`. Returns whether one was published. The mailbox
    /// wakes the driver after releasing the registration lock.
    pub(super) fn retract_descriptor(&self, key: &HandlerKey) -> bool {
        self.descriptors
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(key)
            .is_some()
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

    /// Register occupancy atomically with checking retained cancellation ancestry.
    pub(super) fn begin_lineage(
        &self,
        lineage: Arc<Lineage>,
        key: HandlerKey,
    ) -> Result<Arc<CancelFlag>, ChainCancelled> {
        let mut causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if lineage.is_cancelled() {
            return Err(ChainCancelled);
        }
        let flag = Arc::clone(&lineage.cancelled);
        causality
            .in_flight
            .insert(lineage.id, InFlightEntry { key, lineage });
        Ok(flag)
    }

    #[cfg(all(test, rig_loom))]
    pub(super) fn retained_lineage(&self, id: EffectId) -> Arc<Lineage> {
        self.causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .in_flight
            .get(&id)
            .expect("test dispatch is active")
            .lineage
            .clone()
    }

    #[cfg(test)]
    pub(super) fn begin_in_flight(
        &self,
        id: EffectId,
        key: HandlerKey,
        parent: Option<EffectId>,
    ) -> Result<Arc<CancelFlag>, ChainCancelled> {
        let parent = parent.map(|id| {
            self.causality
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .in_flight
                .get(&id)
                .expect("test parent is active")
                .lineage
                .clone()
        });
        self.begin_lineage(Lineage::new(id, parent), key)
    }

    /// Retiring occupancy does not retire ancestry held by descendants or dispatchers.
    pub(super) fn end_in_flight(&self, id: EffectId) -> bool {
        self.causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .in_flight
            .remove(&id)
            .is_some_and(|entry| entry.lineage.is_cancelled())
    }

    /// Refuse buffered descendants even when intermediate dispatches have completed.
    pub(super) fn fail_cancelled_buffered(&self) {
        let orphans = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            let (orphans, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut queue.commands)
                .into_iter()
                .partition(|command| command.lineage.is_cancelled());
            queue.commands = kept.into();
            orphans
        };
        for orphan in orphans {
            orphan.reply.fail(rig_core::serve::cancelled());
        }
    }

    /// Publish cancellation under the same lock as starting work. Wake outside locks.
    pub(super) fn cancel_descendants(&self, id: EffectId) {
        let flags = {
            let causality = self
                .causality
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let flags: Vec<_> = causality
                .in_flight
                .values()
                .filter(|entry| entry.lineage.contains(id))
                .map(|entry| entry.lineage.cancelled.clone())
                .collect();
            for flag in &flags {
                flag.set.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            flags
        };
        for flag in flags {
            flag.waker.wake();
        }
        // enqueue checks the markers while holding this lock, so a sender either
        // observes cancellation or registers before this wake takes its slot.
        let (senders, driver) = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            (std::mem::take(&mut queue.senders), queue.driver.take())
        };
        wake_parked(senders);
        if let Some(driver) = driver {
            driver.wake();
        }
    }

    fn is_reentrant(&self, command: &Command) -> bool {
        if !self.serial_per_handler {
            return false;
        }
        let causality = self
            .causality
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut next = command.lineage.parent.as_deref();
        while let Some(node) = next {
            if causality
                .in_flight
                .get(&node.id)
                .is_some_and(|entry| entry.key == command.key)
            {
                return true;
            }
            next = node.parent.as_deref();
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

/// Only active occupancy is indexed; ancestry ownership follows live work.
#[derive(Default)]
pub(super) struct Causality {
    in_flight: BTreeMap<EffectId, InFlightEntry>,
}

pub(super) struct ChainCancelled;

/// Immutable ancestry survives completed intermediates without a historical index.
pub(super) struct Lineage {
    id: EffectId,
    parent: Option<Arc<Lineage>>,
    cancelled: Arc<CancelFlag>,
}

impl Lineage {
    pub(super) fn new(id: EffectId, parent: Option<Arc<Self>>) -> Arc<Self> {
        Arc::new(Self {
            id,
            parent,
            cancelled: Arc::new(CancelFlag::default()),
        })
    }

    pub(super) fn is_cancelled(&self) -> bool {
        let mut next = Some(self);
        while let Some(node) = next {
            if node.cancelled.is_set() {
                return true;
            }
            next = node.parent.as_deref();
        }
        false
    }

    fn contains(&self, id: EffectId) -> bool {
        let mut next = Some(self);
        while let Some(node) = next {
            if node.id == id {
                return true;
            }
            next = node.parent.as_deref();
        }
        false
    }
}

impl Drop for Lineage {
    fn drop(&mut self) {
        // A retained dispatcher can outlive thousands of completed ancestors.
        // Release unique ancestry iteratively instead of recursively on the stack.
        let mut next = self.parent.take();
        while let Some(parent) = next {
            match Arc::try_unwrap(parent) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

struct InFlightEntry {
    key: HandlerKey,
    lineage: Arc<Lineage>,
}

/// An ancestor's cancellation wakes the serving future, which drops the
/// owned handler task or stream before notifying its consumer.
#[derive(Default)]
pub(super) struct CancelFlag {
    set: std::sync::atomic::AtomicBool,
    waker: AtomicWaker,
}

impl CancelFlag {
    pub(super) fn is_set(&self) -> bool {
        self.set.load(std::sync::atomic::Ordering::SeqCst)
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
    pub(super) lineage: Arc<Lineage>,
    pub(super) id: EffectId,
    pub(super) key: HandlerKey,
    pub(super) kind: EffectKind,
    /// The dispatch this one was made from: a handler dispatching through
    /// its dispatch context, or `None` for a consumer's own dispatch.
    pub(super) parent: Option<EffectId>,
    /// The scope of the program that made the dispatch, if its dispatcher
    /// was scoped ([`Dispatcher::scoped`]).
    pub(super) scope: Option<Arc<str>>,
    /// The context a tool call runs with, carried beside the effect (never
    /// in it) to the handler's dispatch context ([`DispatchOptions::with_tool_context`]).
    pub(super) context: Option<ToolContext>,
    /// Observation state for this invocation only; never inherited by child dispatches.
    pub(super) adapter_context: Option<rig_core::observe::AdapterContext>,
    /// Where the tool's published values come back, in dispatch context.
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
    /// Answer without a handler (unknown key, closed bus).
    pub(super) fn fail(self, report: ErrorReport) {
        match self {
            Self::Unary(sender) => {
                let _ = sender.send(Err(report));
            }
            Self::Stream(sender) => {
                // A cancellation terminal needs a reserved slot even when
                // ordinary delivery filled the original sender's slot.
                let _ = sender.clone().try_send(Err(report));
            }
        }
    }
}

/// Cloneable bus client for dispatching effects and reading descriptors.
/// It is `Send + Sync` on every target and owns no handlers.
///
/// Dispatch methods create lazy replies; polling those replies attempts the
/// send. A full bus-wide command queue parks the reply until capacity is available.
pub struct Dispatcher {
    lineage: Option<Arc<Lineage>>,
    pub(super) shared: Arc<Shared>,
    pub(super) stream_capacity: usize,
    /// The dispatch every dispatch made through this value descends from:
    /// `None` for a consumer's dispatcher, the served dispatch's id for the
    /// one a handler reads from its dispatch context ([`crate::DispatchScope`]).
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
            lineage: None,
            scope: None,
        }
    }

    /// Create a handler dispatcher retaining ancestry without increasing the
    /// consumer count. Handler dispatchers cannot keep command submission open.
    pub(super) fn parented(
        shared: Arc<Shared>,
        stream_capacity: usize,
        lineage: Arc<Lineage>,
        scope: Option<Arc<str>>,
    ) -> Self {
        Self {
            shared,
            stream_capacity,
            parent: Some(lineage.id),
            lineage: Some(lineage),
            scope,
        }
    }

    /// The dispatch every dispatch made through this value descends from,
    /// if any.
    pub const fn parent(&self) -> Option<EffectId> {
        self.parent
    }

    /// Clone with a caller-supplied recording scope inherited by bound handles
    /// and nested dispatches. Use a stable program identifier, not a runtime handle.
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
        let mut dispatcher = match &self.lineage {
            None => Self::open(Arc::clone(&self.shared), self.stream_capacity),
            Some(lineage) => Self::parented(
                Arc::clone(&self.shared),
                self.stream_capacity,
                lineage.clone(),
                None,
            ),
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

    /// Mint the id a later [`Dispatcher::dispatch_with`] with [`DispatchOptions::with_id`] will carry, so a
    /// hook can see the effect's identity before it is sent.
    pub fn mint_id(&self) -> EffectId {
        self.mint()
    }

    /// Dispatch a unary effect with the default [`DispatchOptions`]: a fresh
    /// id, no tool context, no adapter context. The returned [`Pending`]
    /// resolves to the handler's outcome, or to `BusClosed` /
    /// `HandlerUnavailable`.
    ///
    /// A streaming kind (`Completion { stream: true }`) may be dispatched
    /// unary: the driver folds the handler's events and resolves the
    /// aggregated completion at `Final`.
    pub fn dispatch(&self, key: &HandlerKey, kind: EffectKind) -> Pending {
        self.dispatch_with(key, kind, DispatchOptions::default())
    }

    /// [`dispatch`](Self::dispatch) with explicit [`DispatchOptions`]: an id
    /// minted earlier with [`mint_id`](Self::mint_id), a [`ToolContext`]
    /// that travels beside a tool call to the handler (never on the wire;
    /// what the tool publishes comes back through
    /// [`Pending::published_context`]), and adapter observation context,
    /// which takes precedence over recorder context and is not inherited by
    /// later calls.
    pub fn dispatch_with(
        &self,
        key: &HandlerKey,
        kind: EffectKind,
        options: DispatchOptions,
    ) -> Pending {
        let DispatchOptions {
            id,
            tool_context,
            adapter_context,
        } = options;
        self.dispatch_in(
            id.unwrap_or_else(|| self.mint()),
            key,
            kind,
            tool_context,
            adapter_context,
        )
    }

    fn dispatch_in(
        &self,
        id: EffectId,
        key: &HandlerKey,
        kind: EffectKind,
        context: Option<ToolContext>,
        adapter_context: Option<rig_core::observe::AdapterContext>,
    ) -> Pending {
        let (reply, receiver) = oneshot::channel();
        let (cancel_guard, cancel) = oneshot::channel();
        let published = (context.is_some() || matches!(kind, EffectKind::ToolCall { .. }))
            .then(PublishedContext::new);
        Pending {
            id,
            parent: self.parent,
            state: PendingState::Sending {
                command: Some(Command {
                    lineage: Lineage::new(id, self.lineage.clone()),
                    id,
                    key: key.clone(),
                    kind,
                    parent: self.parent,
                    scope: self.scope.clone(),
                    context,
                    adapter_context,
                    published: published.clone(),
                    reply: Reply::Unary(reply),
                    span: tracing::Span::current(),
                    cancel,
                }),
            },
            receiver,
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            _cancel_guard: cancel_guard,
            published,
        }
    }

    /// Create an identified reply that resolves to `report` on first poll without
    /// reaching the command queue, handler, or recorder.
    pub(crate) fn refused(&self, report: ErrorReport) -> Pending {
        let (_reply, receiver) = oneshot::channel();
        let (cancel_guard, _cancel) = oneshot::channel();
        Pending {
            id: self.mint_id(),
            parent: self.parent,
            state: PendingState::Failed(Some(report)),
            receiver,
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            _cancel_guard: cancel_guard,
            published: None,
        }
    }

    /// Dispatch a streaming effect. A non-streaming kind yields one request
    /// error without reaching a handler.
    pub fn dispatch_stream(&self, key: &HandlerKey, kind: EffectKind) -> EffectStream {
        self.dispatch_stream_with(key, kind, DispatchOptions::default())
    }

    /// [`dispatch_stream`](Self::dispatch_stream) with explicit
    /// [`DispatchOptions`]: an id minted earlier and adapter observation
    /// context, retained through lazy startup and stream consumption
    /// (caller context wins over recording). A stream carries no tool
    /// context; `tool_context` is ignored.
    pub fn dispatch_stream_with(
        &self,
        key: &HandlerKey,
        kind: EffectKind,
        options: DispatchOptions,
    ) -> EffectStream {
        let DispatchOptions {
            id,
            adapter_context,
            ..
        } = options;
        self.dispatch_stream_in(
            id.unwrap_or_else(|| self.mint()),
            key,
            kind,
            adapter_context,
        )
    }

    fn dispatch_stream_in(
        &self,
        id: EffectId,
        key: &HandlerKey,
        kind: EffectKind,
        adapter_context: Option<rig_core::observe::AdapterContext>,
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
                command: Some(Command {
                    lineage: Lineage::new(id, self.lineage.clone()),
                    id,
                    key: key.clone(),
                    kind,
                    parent: self.parent,
                    scope: self.scope.clone(),
                    context: None,
                    adapter_context,
                    published: None,
                    reply: Reply::Stream(events),
                    span: tracing::Span::current(),
                    cancel,
                }),
                receiver: Some(receiver),
            },
            shared: self.shared.clone(),
            parked: Arc::new(AtomicWaker::new()),
            _cancel_guard: Some(cancel_guard),
        }
    }

    /// The descriptor of the handler serving `key`: a snapshot of the
    /// descriptor table, no round trip. `None` when nothing serves the key.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.shared.descriptor(key)
    }

    /// Every registered key, in key order.
    pub fn keys(&self) -> Vec<HandlerKey> {
        self.shared.keys()
    }

    /// Every registered descriptor in key order, captured under one read lock.
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        self.shared.descriptors()
    }

    /// Process-local identity shared by clones and distinct from other live
    /// buses. Do not persist it; allocations may reuse IDs after a bus is dropped.
    /// Pair it with bus-local `EffectId`s for cross-bus bookkeeping.
    pub fn id(&self) -> BusId {
        self.shared.id()
    }

    /// Whether the driver has been dropped. A dispatch on a closed bus
    /// resolves `BusClosed` on first poll.
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }

    /// Number of commands queued for the driver, bounded by the configured
    /// command capacity (clamped to at least one).
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

/// A stream that ended before its `Final`: the returned stream ended
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
        command: Option<Command>,
    },
    Waiting,
    /// Refused before any send: the request had no wire form
    /// ([`rig_core::effect::Family::wrap`] failed). Resolves the report on
    /// the first poll; nothing reaches a handler or a recorder.
    Failed(Option<ErrorReport>),
}

/// Optional dispatch identity, tool inputs, and adapter observation context.
#[derive(Debug, Default)]
pub struct DispatchOptions {
    /// An id minted earlier with [`Dispatcher::mint_id`], so the caller can
    /// name the effect before it is sent; a fresh one otherwise.
    pub id: Option<EffectId>,
    /// The tool context a tool call travels with (unary dispatch only).
    pub tool_context: Option<ToolContext>,
    /// Invocation observation state for the adapter; takes precedence over
    /// recorder context and is not inherited by later calls.
    pub adapter_context: Option<rig_core::observe::AdapterContext>,
}

impl DispatchOptions {
    /// Dispatch under `id`.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_id(mut self, id: EffectId) -> Self {
        self.id = Some(id);
        self
    }

    /// Carry `context` to the tool handler.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_tool_context(mut self, context: ToolContext) -> Self {
        self.tool_context = Some(context);
        self
    }

    /// Observe the dispatch under `context`.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_adapter_context(mut self, context: rig_core::observe::AdapterContext) -> Self {
        self.adapter_context = Some(context);
        self
    }
}

/// Executor-independent unary reply resolving to an outcome or error report.
/// Dropping it signals cancellation; handler cleanup requires polling the driver.
#[must_use = "a dispatch does nothing until polled"]
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
    /// Where a tool call's published context comes back, including raw
    /// tool dispatches without explicit inputs.
    published: Option<Arc<PublishedContext>>,
}

impl Pending {
    /// Where the tool's published context lands once this dispatch resolved
    /// (clone it before awaiting the dispatch): `Some` for tool calls,
    /// including raw dispatches, and explicit context-bearing dispatches.
    pub fn published_context(&self) -> Option<Arc<PublishedContext>> {
        self.published.clone()
    }

    /// The dispatch this one was made from: `Some` when a handler dispatched
    /// it through its dispatch context, `None` for a consumer's own.
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
                        Enqueue::Cancelled(cancelled) => {
                            drop(cancelled);
                            return Poll::Ready(Err(rig_core::serve::cancelled()));
                        }
                        Enqueue::Refused(refused) => {
                            return Poll::Ready(Err(reentrant(&refused.key)));
                        }
                        Enqueue::Closed => return Poll::Ready(Err(bus_closed())),
                    }
                }
                PendingState::Failed(report) => {
                    return Poll::Ready(Err(report.take().unwrap_or_else(|| {
                        ErrorReport::new(ErrorKind::Internal, "a refused dispatch was polled twice")
                    })));
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
        command: Option<Command>,
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

/// Executor-independent stream of events ending with `Final` or an error.
/// Dropping signals cancellation; polling the driver releases handler work.
/// Pausing consumption applies backpressure through the bounded event channel.
#[must_use = "a dispatch does nothing until polled"]
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
                        Enqueue::Cancelled(cancelled) => {
                            drop(cancelled);
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(rig_core::serve::cancelled())));
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
                            // Missing terminal events must surface as errors, not silent EOF.
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
const _: () = {
    const fn assert_dispatcher<T: Clone + Send + Sync + 'static>() {}
    const fn assert_unpin<T: Unpin + 'static>() {}
    // Browser ECS components also require Send; replies must never contain handlers.
    const fn assert_send<T: Send + 'static>() {}
    assert_dispatcher::<Dispatcher>();
    assert_unpin::<Pending>();
    assert_unpin::<EffectStream>();
    assert_send::<Pending>();
    assert_send::<EffectStream>();
};

#[cfg(all(test, not(rig_loom)))]
mod tests;
