//! The serving half of the bus: `BusDriver`.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{
    StreamExt,
    stream::{FusedStream, FuturesUnordered},
};
use tracing::Instrument;

use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    serve::{Observe, Origin, OutcomeSink, Recorder},
    streaming::StreamEvent,
    wasm_compat::WasmBoxedFuture,
};

use rig_core::serve::{ErasedHandler, Serve};

use super::{
    dispatcher::{Command, Dispatcher, Shared, handler_unavailable},
    registrar::{Mailbox, Registrar, Registration},
};

use rig_core::serve::ServingPolicy;

type InFlight = WasmBoxedFuture<'static, (HandlerKey, EffectId)>;
type InFlightServing = Pin<Box<Serving>>;

/// The driver's hold on a recorder: closures, so the driver names no
/// recorder type, and one observer per dispatch — the driver is not on the
/// reply path (the consumer holds the reply channel), so the sink tells it.
struct Recording {
    handlers: Box<dyn Fn(Vec<HandlerDescriptor>) + Send + Sync>,
    begin: Box<dyn Fn(EffectId, HandlerKey, EffectKind, Origin) + Send + Sync>,
    observe: Box<dyn Fn(OutcomeSink, EffectId) -> OutcomeSink + Send + Sync>,
}

/// The record's view of one dispatch: the recorder, told by id.
struct Recorded<R> {
    recorder: R,
    id: EffectId,
}

impl<R: Recorder + Send + Sync> Observe for Recorded<R> {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.recorder.resolve(self.id, outcome.clone());
    }

    fn keep_events(&self) -> bool {
        self.recorder.keep_events()
    }

    fn event(&mut self, event: &StreamEvent) {
        self.recorder.event(self.id, event);
    }

    fn discard(&mut self) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}

impl Recording {
    fn new<R: Recorder + Clone + Send + Sync>(recorder: R) -> Self {
        let for_handlers = recorder.clone();
        let handlers = Box::new(move |described| for_handlers.handlers(described));
        let for_begin = recorder.clone();
        let begin = Box::new(move |id, key, kind, origin| for_begin.begin(id, key, kind, origin));
        let observe = Box::new(move |sink: OutcomeSink, id: EffectId| {
            sink.with_observer(Box::new(Recorded {
                recorder: recorder.clone(),
                id,
            }))
        });
        Self {
            handlers,
            begin,
            observe,
        }
    }
}

/// The serving half of the bus: a plain future that owns the **only**
/// handler table and runs handlers as commands arrive. Handlers reach it
/// by value before it is spawned ([`BusDriver::register`]) or through a
/// [`Registrar`] afterwards; the shared half of the bus carries their
/// descriptors, never the handlers themselves.
///
/// `Send` on native (asserted), so `IoTaskPool::get().spawn(driver)` — or
/// any `Send + 'static` spawn — takes it; `!Send` is allowed on browser
/// wasm, where the pool accepts it too. It completes when every
/// [`Dispatcher`](super::Dispatcher) clone has dropped and no dispatch is in
/// flight. Dropping it before then — a cancelled task, a despawned entity —
/// fails every in-flight and later dispatch with `BusClosed`.
///
/// **Whoever holds the driver drives.** Nothing else advances it; a
/// dispatcher whose driver sits un-polled waits forever, so the owner must
/// spawn it, drive it inline (`select(pending, &mut driver)`), or hand it
/// over together with the dispatcher.
pub struct BusDriver {
    shared: Arc<Shared>,
    mailbox: Arc<Mailbox>,
    handlers: BTreeMap<HandlerKey, ErasedHandler>,
    config: ServingPolicy,
    in_flight: FuturesUnordered<InFlightServing>,
    queued: BTreeMap<HandlerKey, VecDeque<Command>>,
    busy: BTreeSet<HandlerKey>,
    recorder: Option<Recording>,
    commands_closed: bool,
}

impl fmt::Debug for BusDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BusDriver")
            .field("config", &self.config)
            .field("in_flight", &self.in_flight.len())
            .field(
                "queued",
                &self.queued.values().map(VecDeque::len).sum::<usize>(),
            )
            .field("handlers", &self.handlers.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl BusDriver {
    pub(super) fn new(shared: Arc<Shared>, mailbox: Arc<Mailbox>, config: ServingPolicy) -> Self {
        shared.driver_born();
        Self {
            shared,
            mailbox,
            handlers: BTreeMap::new(),
            config,
            in_flight: FuturesUnordered::new(),
            queued: BTreeMap::new(),
            busy: BTreeSet::new(),
            recorder: None,
            commands_closed: false,
        }
    }

    /// Register (or replace) the handler serving `key` while the driver is
    /// in hand — before it is spawned, or between polls when the owner
    /// drives it inline. Installed at once; the same descriptor
    /// [`Registrar::register`] publishes.
    pub fn register(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<(), ErrorReport> {
        self.register_erased(key, ErasedHandler::new(handler))
    }

    /// Register an already-erased handler.
    pub fn register_erased(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: ErasedHandler,
    ) -> Result<(), ErrorReport> {
        let key = key.into();
        self.shared
            .publish_descriptor(key.clone(), handler.descriptor())?;
        self.install(key, handler);
        Ok(())
    }

    /// [`register`](Self::register), returning a [`Key`](rig_core::effect::Key) that carries the
    /// family the handler proved by its descriptor.
    pub fn register_typed<F: rig_core::effect::Family>(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<rig_core::effect::Key<F>, ErrorReport> {
        let key = key.into();
        let handler = ErasedHandler::new(handler);
        let descriptor = handler.descriptor();
        if descriptor.family.family() != F::FAMILY {
            return Err(super::registrar::family_proof_failed(
                &key,
                F::FAMILY,
                &descriptor,
            ));
        }
        self.register_erased(key.clone(), handler)?;
        Ok(rig_core::effect::Key::new_unchecked(key))
    }

    /// Remove the handler serving `key`. Returns whether one was registered.
    pub fn deregister(&mut self, key: &HandlerKey) -> bool {
        let published = self.shared.retract_descriptor(key);
        self.handlers.remove(key).is_some() || published
    }

    /// A handle for registering on this bus once the driver is out of hand
    /// (spawned): see [`Registrar`].
    pub fn registrar(&self) -> Registrar {
        Registrar {
            shared: Arc::clone(&self.shared),
            mailbox: Arc::clone(&self.mailbox),
        }
    }

    /// Put `handler` in the table. The displaced handler, if any, is dropped
    /// here, with no lock held — its `Drop` may touch this bus.
    fn install(&mut self, key: HandlerKey, handler: ErasedHandler) {
        if let Some(recording) = &self.recorder {
            let described = handler.descriptor();
            (recording.handlers)(vec![HandlerDescriptor {
                key: key.clone(),
                family: described.family,
                layers: described.layers,
            }]);
        }
        let displaced = self.handlers.insert(key, handler);
        drop(displaced);
    }

    /// Apply what the registrars posted since the last poll.
    fn apply_registrations(&mut self, cx: &Context<'_>) {
        for registration in self.mailbox.drain(cx) {
            match registration {
                Registration::Install { key, handler } => self.install(key, handler),
                Registration::Remove { key } => {
                    let removed = self.handlers.remove(&key);
                    drop(removed);
                }
            }
        }
    }

    /// Record every served dispatch into `recorder` (a
    /// [`rig_core::serve::Recorder`]); the handlers registered now are
    /// handed to it first ([`Recorder::handlers`]), and each one installed
    /// later as it is installed.
    pub fn record_to<R: Recorder + Clone + Send + Sync>(&mut self, recorder: R) {
        recorder.handlers(
            self.handlers
                .iter()
                .map(|(key, handler)| {
                    let described = handler.descriptor();
                    HandlerDescriptor {
                        key: key.clone(),
                        family: described.family,
                        layers: described.layers,
                    }
                })
                .collect(),
        );
        self.recorder = Some(Recording::new(recorder));
    }

    /// The serving policy.
    pub const fn config(&self) -> &ServingPolicy {
        &self.config
    }

    /// Dispatches currently being served.
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Dispatches accepted but waiting for their key under serial serving.
    pub fn queued(&self) -> usize {
        self.queued.values().map(VecDeque::len).sum()
    }

    /// Start serving `command`. Returns whether it went in flight; a command
    /// with no handler is answered `HandlerUnavailable` on the spot and
    /// never occupies its key, and a command whose consumer is already gone
    /// is dropped unserved — no handler poll, no record.
    fn serve(&mut self, command: Command) -> bool {
        let Command {
            id,
            key,
            kind,
            parent,
            scope,
            reply,
            span,
            mut cancel,
        } = command;
        if cancel.try_recv().is_err() {
            drop(reply);
            self.shared.resolved(id);
            return false;
        }
        let Some(handler) = self.handlers.get(&key).cloned() else {
            reply.fail(handler_unavailable(&key));
            self.shared.resolved(id);
            return false;
        };
        // A dispatch whose ancestor was cancelled while it was queued is
        // dropped unserved: no handler poll, no record.
        let Ok(flag) = self.shared.begin_in_flight(id, key.clone(), parent) else {
            reply.fail(rig_core::serve::cancelled());
            self.shared.resolved(id);
            return false;
        };
        if self.config.serial_per_handler {
            self.busy.insert(key.clone());
        }
        let (done, sink_done) = futures::channel::oneshot::channel();
        // The handler's way back onto this bus: a dispatcher whose dispatches
        // descend from this one.
        let scoped = Dispatcher::parented(
            Arc::clone(&self.shared),
            self.config.stream_capacity,
            id,
            scope.clone(),
        );
        let sink = reply
            .into_sink(id)
            .with_done(done)
            .with_cancel(flag.marker())
            .with_scope(Arc::new(scoped));
        let sink = match &self.recorder {
            Some(recorder) => {
                (recorder.begin)(id, key.clone(), kind.clone(), Origin { parent, scope });
                (recorder.observe)(sink, id)
            }
            None => sink,
        };
        let task_key = key.clone();
        let shared = Arc::clone(&self.shared);
        let task = Box::pin(
            async move {
                use futures::future::{Either, select};
                let serving = handler.handle(kind, sink);
                let ancestor_cancelled = flag.wait();
                // The handler races the consumer's cancel and an ancestor's:
                // either drops the handler future (and the sink, which
                // reports the cancel); the consumer's also reaches every
                // descendant of this dispatch. A handler that returned with
                // its sink detached is still a dispatch in flight, so the
                // cancels keep racing the sink's answer after it.
                // The loser of a race is dropped before the sink is awaited:
                // a handler future dropped is the sink dropped (unless it was
                // detached), which is what resolves `sink_done`.
                match select(serving, select(cancel, ancestor_cancelled)).await {
                    Either::Left((_, cancels)) => match select(sink_done, cancels).await {
                        Either::Left(_) => {}
                        Either::Right((Either::Left(_), sink_done)) => {
                            shared.cancel_descendants(id);
                            let _ = sink_done.await;
                        }
                        Either::Right((Either::Right(_), sink_done)) => {
                            let _ = sink_done.await;
                        }
                    },
                    Either::Right((Either::Left(_), serving)) => {
                        drop(serving);
                        shared.cancel_descendants(id);
                        let _ = sink_done.await;
                    }
                    Either::Right((Either::Right(_), serving)) => {
                        drop(serving);
                        let _ = sink_done.await;
                    }
                }
                (task_key, id)
            }
            .instrument(span),
        );
        self.in_flight.push(Box::pin(Serving { task }));
        true
    }

    fn accept(&mut self, command: Command) {
        if self.config.serial_per_handler && self.busy.contains(&command.key) {
            self.queued
                .entry(command.key.clone())
                .or_default()
                .push_back(command);
        } else {
            self.serve(command);
        }
    }

    /// The in-flight command for `key` finished: serve the next queued one
    /// that can go in flight. A queued command whose handler is gone is
    /// answered on the spot and the loop moves on — a key never strands
    /// its queue behind a command that will not be served.
    fn release(&mut self, key: HandlerKey, id: EffectId) {
        self.shared.resolved(id);
        if self.shared.end_in_flight(id) {
            // Cancelled: the children it still has here are dropped unserved
            // — no handler poll, no record — and answered as cancelled.
            for queue in self.queued.values_mut() {
                let (orphans, kept): (Vec<_>, Vec<_>) = std::mem::take(queue)
                    .into_iter()
                    .partition(|command| command.parent == Some(id));
                *queue = kept.into();
                for orphan in orphans {
                    let id = orphan.id;
                    orphan.reply.fail(rig_core::serve::cancelled());
                    self.shared.resolved(id);
                }
            }
            self.shared.fail_buffered_children(id);
        }
        if !self.config.serial_per_handler {
            return;
        }
        self.busy.remove(&key);
        loop {
            let next = self.queued.get_mut(&key).and_then(VecDeque::pop_front);
            let Some(command) = next else {
                self.queued.remove(&key);
                return;
            };
            if self.serve(command) {
                return;
            }
        }
    }

    /// Answer every command queued for a key that no longer has a handler.
    /// Deregistration wakes the driver so this runs promptly.
    fn drain_orphaned_queues(&mut self) {
        if !self.config.serial_per_handler || self.queued.is_empty() {
            return;
        }
        let orphaned: Vec<HandlerKey> = self
            .queued
            .keys()
            .filter(|key| !self.handlers.contains_key(*key))
            .cloned()
            .collect();
        for key in orphaned {
            if let Some(queue) = self.queued.remove(&key) {
                for command in queue {
                    let id = command.id;
                    command.reply.fail(handler_unavailable(&key));
                    self.shared.resolved(id);
                }
            }
        }
    }
}

/// An in-flight task: the handler's future, the cancels it races and the
/// sink's answer, ending as the key it occupied and the dispatch's id so
/// the driver can release the one and end the other.
struct Serving {
    task: InFlight,
}

impl Future for Serving {
    type Output = (HandlerKey, EffectId);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<(HandlerKey, EffectId)> {
        self.get_mut().task.as_mut().poll(cx)
    }
}

impl Future for BusDriver {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        loop {
            // Take every buffered command; each becomes an in-flight task or
            // a queued one. Draining registers this poll's waker for the
            // next enqueue and releases any dispatch parked on the bound.
            if !this.commands_closed {
                // Take the commands first, then the registrations, then
                // serve: a registration made before a dispatch (program
                // order on the registering thread) is in the mailbox by the
                // time the dispatch is in the queue, so taking the queue
                // first and the mailbox second sees every registration the
                // taken commands rely on. The other order let a
                // registration posted between the two drains be missed for
                // the dispatch that followed it, which was then served as
                // `HandlerUnavailable`.
                let commands = this.shared.drain(cx);
                this.apply_registrations(cx);
                for command in commands {
                    this.accept(*command);
                }
                this.drain_orphaned_queues();
                // The bus is closed for commands once every dispatcher has
                // dropped and nothing it enqueued remains — decided under
                // the queue lock, so no late send lands after the decision.
                // Observable to a `Pending` that outlived its dispatcher: its
                // send answers `BusClosed` now, not after the last in-flight
                // stream ends.
                if this.shared.try_close_commands() {
                    this.commands_closed = true;
                }
            }
            // Drive the tasks. A completion may release a queued command,
            // which is why the outer loop re-enters.
            let mut progressed = false;
            loop {
                if this.in_flight.is_terminated() || this.in_flight.is_empty() {
                    break;
                }
                match this.in_flight.poll_next_unpin(cx) {
                    Poll::Ready(Some((key, id))) => {
                        progressed = true;
                        this.release(key, id);
                    }
                    Poll::Ready(None) | Poll::Pending => break,
                }
            }
            if progressed {
                continue;
            }
            if this.commands_closed && this.in_flight.is_empty() {
                return Poll::Ready(());
            }
            return Poll::Pending;
        }
    }
}

impl Drop for BusDriver {
    fn drop(&mut self) {
        // The guard: after this every reply the channel loses is `BusClosed`,
        // and handlers posted but never installed go with the driver. The
        // descriptor table goes too — it described handlers this driver
        // held — and the bus is free for `Bus::reopen`.
        self.shared.mark_closed();
        self.mailbox.clear();
        self.shared.driver_died();
    }
}

#[cfg(not(target_family = "wasm"))]
const _: () = {
    const fn assert_send<T: Send + 'static>() {}
    assert_send::<BusDriver>();
};
