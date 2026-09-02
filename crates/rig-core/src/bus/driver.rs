//! The serving half of the bus: `BusDriver`.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    pin::Pin,
    sync::{Arc, Mutex, PoisonError},
    task::{Context, Poll},
};

use futures::{
    StreamExt,
    stream::{FusedStream, FuturesUnordered},
};
use tracing::Instrument;

use crate::{
    effect::{EffectId, EffectKind, EffectLog, EffectRecord, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::{BlockAccumulator, StreamEvent},
    wasm_compat::WasmBoxedFuture,
};

use super::{
    ErasedHandler, Handler, OutcomeSink,
    dispatcher::{Command, Shared, handler_unavailable},
};

/// Bus sizing and serving policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A bus tap: every dispatch the driver serves is appended, as an
/// [`EffectRecord`], when it resolves. Cloning shares the log; a streaming
/// dispatch is recorded as the aggregated completion its events fold to.
#[derive(Clone, Default)]
pub struct EffectLogRecorder {
    log: Arc<Mutex<EffectLog>>,
}

impl EffectLogRecorder {
    /// An empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of everything recorded so far, in dispatch-resolution order.
    pub fn log(&self) -> EffectLog {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Take the recorded log, leaving the recorder empty.
    pub fn take(&self) -> EffectLog {
        std::mem::take(&mut *self.log.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn push(&self, record: EffectRecord) {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(record);
    }
}

impl fmt::Debug for EffectLogRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectLogRecorder")
            .field("records", &self.log().len())
            .finish()
    }
}

type InFlight = WasmBoxedFuture<'static, HandlerKey>;
type InFlightServing = Pin<Box<Serving>>;

/// The serving half of the bus: a plain future that owns the handler table
/// and runs handlers as commands arrive.
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
    config: BusConfig,
    in_flight: FuturesUnordered<InFlightServing>,
    queued: BTreeMap<HandlerKey, VecDeque<Command>>,
    busy: BTreeSet<HandlerKey>,
    recorder: Option<EffectLogRecorder>,
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
            .field("handlers", &self.shared.keys())
            .finish_non_exhaustive()
    }
}

impl BusDriver {
    pub(super) fn new(shared: Arc<Shared>, config: BusConfig) -> Self {
        Self {
            shared,
            config,
            in_flight: FuturesUnordered::new(),
            queued: BTreeMap::new(),
            busy: BTreeSet::new(),
            recorder: None,
            commands_closed: false,
        }
    }

    /// Register (or replace) the handler serving `key` before the driver is
    /// spawned. The same table [`Dispatcher::register`](super::Dispatcher::register)
    /// writes at runtime.
    pub fn register(&mut self, key: impl Into<HandlerKey>, handler: impl Handler + 'static) {
        self.shared
            .register(key.into(), ErasedHandler::new(handler));
    }

    /// Register an already-erased handler.
    pub fn register_erased(&mut self, key: impl Into<HandlerKey>, handler: ErasedHandler) {
        self.shared.register(key.into(), handler);
    }

    /// Remove the handler serving `key`. Returns whether one was registered.
    pub fn deregister(&mut self, key: &HandlerKey) -> bool {
        self.shared.deregister(key)
    }

    /// Record every served dispatch into `recorder`.
    pub fn record_to(&mut self, recorder: EffectLogRecorder) {
        self.recorder = Some(recorder);
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
    /// never occupies its key.
    fn serve(&mut self, command: Command) -> bool {
        let Command {
            id,
            key,
            kind,
            reply,
            span,
            cancel,
        } = command;
        let Some(handler) = self.shared.handler(&key) else {
            reply.fail(handler_unavailable(&key));
            return false;
        };
        if self.config.serial_per_handler {
            self.busy.insert(key.clone());
        }
        let sink = reply.into_sink(id);
        let sink = match &self.recorder {
            Some(recorder) => tap(sink, recorder.clone(), id, key.clone(), kind.clone()),
            None => sink,
        };
        let task_key = key.clone();
        let task = Box::pin(
            async move {
                // Cancellation is drop: the consumer dropping its `Pending` or
                // `EffectStream` resolves `cancel`, which drops the handler
                // future — and with it the provider call or stream inside.
                let serving = handler.handle(kind, sink);
                futures::pin_mut!(serving);
                let _ = futures::future::select(serving, cancel).await;
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
            .filter(|key| self.shared.handler(key).is_none())
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
                for command in this.shared.drain(cx) {
                    this.accept(*command);
                }
                this.drain_orphaned_queues();
                // The bus is closed for commands once every dispatcher has
                // dropped and nothing it enqueued remains.
                if this.shared.dispatchers() == 0 && this.shared.buffered() == 0 {
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
        // The guard: after this every reply the channel loses is `BusClosed`.
        self.shared.mark_closed();
    }
}

// A handler wrapper that records the dispatch when it resolves.
fn tap(
    sink: OutcomeSink,
    recorder: EffectLogRecorder,
    id: EffectId,
    key: HandlerKey,
    kind: EffectKind,
) -> OutcomeSink {
    sink.with_tap(Box::new(move |outcome: &Result<Outcome, ErrorReport>| {
        recorder.push(EffectRecord {
            id,
            key: key.clone(),
            kind: kind.clone(),
            outcome: outcome.clone(),
        });
    }))
}

/// Folds a tapped stream into the completion the record stores.
pub(super) struct StreamTap {
    accumulator: BlockAccumulator,
    message_id: Option<String>,
}

impl StreamTap {
    pub(super) fn new() -> Self {
        Self {
            accumulator: BlockAccumulator::new(),
            message_id: None,
        }
    }

    /// Fold one event; returns the recorded outcome at the terminal.
    pub(super) fn observe(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
    ) -> Option<Result<Outcome, ErrorReport>> {
        match item {
            Err(report) => Some(Err(report.clone())),
            Ok(StreamEvent::Final(terminal)) => Some(super::handler::finish_unary(
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
                if let Err(error) = self.accumulator.apply(event) {
                    return Some(Err(ErrorReport::from(&error)));
                }
                None
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send<T: Send + 'static>() {}
    assert_send::<BusDriver>();
};
