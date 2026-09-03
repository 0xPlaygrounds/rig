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
    serve::{OnEvent, OnOutcome, OutcomeSink, Recorder},
    streaming::StreamEvent,
    wasm_compat::WasmBoxedFuture,
};

use rig_core::serve::{ErasedHandler, Serve};

use super::{
    dispatcher::{Command, Shared, handler_unavailable},
    registrar::{Mailbox, Registrar, Registration},
};

/// Bus sizing and serving policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BusConfig {
    /// Commands the bus buffers, **bus-wide**, before a `Pending`/
    /// `EffectStream` parks at its send stage (its poll stays `Pending`
    /// until the driver drains). The bound holds across every `Dispatcher`
    /// clone and every dispatch; the caller of `dispatch` is never blocked.
    pub command_capacity: usize,
    /// Stream events buffered per streaming dispatch before the handler
    /// stalls (the client-side pause point).
    pub stream_capacity: usize,
    /// Serve one command at a time per key. `false` serves every command
    /// concurrently; `true` is the cassette-ordered property — a handler
    /// sees its dispatches in the order they arrived.
    ///
    /// Under serial serving a handler must not dispatch to **its own key**
    /// and wait for the answer: that dispatch would queue behind the
    /// command that waits on it. The bus refuses the case it can see — a
    /// dispatch to the key being served, made on the thread the driver is
    /// polling it on — with a `Request` report instead of hanging. It
    /// cannot see a nested dispatch made from another task or thread the
    /// handler spawned (`tokio::spawn`, `IoTaskPool::spawn`): that one
    /// queues behind its parent and waits forever. A handler that needs
    /// its own key serves it from a second key, or the bus runs with
    /// `serial_per_handler: false`.
    pub serial_per_handler: bool,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            command_capacity: 16,
            stream_capacity: 64,
            serial_per_handler: false,
        }
    }
}

type InFlight = WasmBoxedFuture<'static, HandlerKey>;
type InFlightServing = Pin<Box<Serving>>;

/// The driver's hold on a recorder: closures, like the sink's taps, so the
/// driver names no recorder type.
struct Recording {
    handlers: Box<dyn Fn(Vec<HandlerDescriptor>) + Send + Sync>,
    begin: Box<dyn Fn(EffectId, HandlerKey, EffectKind) + Send + Sync>,
    tap: Box<dyn Fn(OutcomeSink, EffectId) -> OutcomeSink + Send + Sync>,
}

impl Recording {
    fn new<R: Recorder + Clone + Send + Sync>(recorder: R) -> Self {
        let for_handlers = recorder.clone();
        let handlers = Box::new(move |described| for_handlers.handlers(described));
        let for_begin = recorder.clone();
        let begin = Box::new(move |id, key, kind| for_begin.begin(id, key, kind));
        let tap = Box::new(move |sink: OutcomeSink, id: EffectId| {
            let on_event: Option<OnEvent> = recorder.keep_events().then(|| {
                let recorder = recorder.clone();
                Box::new(move |event: &StreamEvent| recorder.event(id, event)) as OnEvent
            });
            let recorder = recorder.clone();
            let on_outcome: OnOutcome = Box::new(move |outcome: &Result<Outcome, ErrorReport>| {
                recorder.resolve(id, outcome.clone());
            });
            sink.with_tap(on_outcome, on_event)
        });
        Self {
            handlers,
            begin,
            tap,
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
    config: BusConfig,
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
    pub(super) fn new(shared: Arc<Shared>, mailbox: Arc<Mailbox>, config: BusConfig) -> Self {
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
            (recording.handlers)(vec![HandlerDescriptor {
                key: key.clone(),
                family: handler.descriptor().family,
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
                .map(|(key, handler)| HandlerDescriptor {
                    key: key.clone(),
                    family: handler.descriptor().family,
                })
                .collect(),
        );
        self.recorder = Some(Recording::new(recorder));
    }

    /// The serving policy.
    pub const fn config(&self) -> &BusConfig {
        &self.config
    }

    /// Dispatches currently being served.
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
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
            reply,
            span,
            mut cancel,
        } = command;
        // A `Pending`/`EffectStream` dropped between its send and this poll
        // (a host despawning in the frame it dispatched) has already
        // resolved `cancel`. Serving it would give the handler one poll —
        // enough to start a provider request — and open a record nobody
        // asked for; a dispatch nobody wants is never served.
        if cancel.try_recv().is_err() {
            drop(reply);
            return false;
        }
        let Some(handler) = self.handlers.get(&key).cloned() else {
            reply.fail(handler_unavailable(&key));
            return false;
        };
        if self.config.serial_per_handler {
            self.busy.insert(key.clone());
        }
        // The dispatch is over when the *sink* has answered or been dropped
        // — not when the handler future ends: a handler may detach its
        // sink and hand it to a system that answers later, and until then
        // the key stays busy and the dispatch in flight.
        let (done, sink_done) = futures::channel::oneshot::channel();
        let sink = reply.into_sink(id).with_done(done);
        let sink = match &self.recorder {
            Some(recorder) => {
                // The record's place in the log is its place in the serve
                // order; the outcome fills it in when the dispatch resolves.
                (recorder.begin)(id, key.clone(), kind.clone());
                (recorder.tap)(sink, id)
            }
            None => sink,
        };
        let task_key = key.clone();
        let task = Box::pin(
            async move {
                // Cancellation is drop: the consumer dropping its `Pending` or
                // `EffectStream` resolves `cancel`, which drops the handler
                // future — and with it the provider call or stream inside.
                // The handler is polled first on purpose: a handler in
                // flight observes the cancel on its own next poll (its send
                // answers `SinkClosed`) before the future is dropped, which
                // is how a streaming adapter stops cleanly. A cancel that
                // arrived *before* serving never gets here (`serve` above).
                // The handler future is a `Pin<Box<_>>` and moves into the
                // select by value; the select's result owns whichever side
                // lost and drops it here, with the sink inside — unless the
                // handler detached the sink first.
                let serving = handler.handle(kind, sink);
                drop(futures::future::select(serving, cancel).await);
                // Ended or dropped: if the handler detached its sink, wait
                // for whoever holds it. (Undetached, the sink went with the
                // future and this is already resolved.)
                let _ = sink_done.await;
                task_key
            }
            .instrument(span),
        );
        self.in_flight.push(Box::pin(Serving {
            key,
            shared: Arc::clone(&self.shared),
            task,
        }));
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
    fn release(&mut self, key: HandlerKey) {
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
                    command.reply.fail(handler_unavailable(&key));
                }
            }
        }
    }
}

/// An in-flight task that tells the bus which key is being polled, so a
/// dispatch made from inside the handler can be recognised as re-entrant.
struct Serving {
    key: HandlerKey,
    shared: Arc<Shared>,
    task: InFlight,
}

impl Future for Serving {
    type Output = HandlerKey;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<HandlerKey> {
        let this = self.get_mut();
        this.shared.set_serving(Some(this.key.clone()));
        let polled = this.task.as_mut().poll(cx);
        this.shared.set_serving(None);
        polled
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
                // dropped and nothing it enqueued remains.
                if this.shared.dispatchers() == 0 && this.shared.buffered() == 0 {
                    this.commands_closed = true;
                    // Observable to a `Pending` that outlived its dispatcher:
                    // its send answers `BusClosed` now, not after the last
                    // in-flight stream ends.
                    this.shared.close_commands();
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
                    Poll::Ready(Some(key)) => {
                        progressed = true;
                        this.release(key);
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
