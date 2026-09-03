//! The agent's bus: the dispatcher every run dispatches through, the driver
//! the agent drives inline while a run is awaited, and the recorder that
//! taps it.
//!
//! **Whoever holds the driver drives — and whoever is awaiting a run holds
//! it.** An agent built with [`AgentBuilder::new`](super::AgentBuilder::new)
//! owns its driver, and every run it produces polls that driver whenever
//! the run is pending ([`AgentBus::drive`], [`Driven`]); no run owns the
//! driver for longer than one poll. The agent never hands its dispatcher
//! out on its own — [`Agent::into_parts`]
//! (super::Agent::into_parts) moves the driver out together with it. An
//! agent built over a host's bus ([`AgentBuilder::over_bus`]
//! (super::AgentBuilder::over_bus)) holds no driver: the host drives.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures::{Stream, lock::Mutex};
use rig_core::{
    bus::{
        BusDriver, Dispatcher, EffectLogRecorder, ErasedHandler, Key, Registrar,
        adapters::CompletionAdapter,
    },
    completion::{CompletionModel, ModelRef},
    effect::{EffectLog, HandlerKey, family},
    error::ErrorReport,
};

/// The per-process counter behind an agent's default owner label.
static NEXT_AGENT: AtomicU64 = AtomicU64::new(0);

/// The owner label an agent gets when its builder names none:
/// `agent#<n>`, distinct per process.
pub(crate) fn default_owner() -> String {
    format!("agent#{}", NEXT_AGENT.fetch_add(1, Ordering::Relaxed))
}

/// The registration of a model under a generated label, scoped to the
/// values that selected it: the last clone dropping deregisters the key.
pub(crate) struct AnonymousModel {
    key: Key<family::Completion>,
    registrar: Registrar,
}

impl AnonymousModel {
    pub(crate) fn key(&self) -> &Key<family::Completion> {
        &self.key
    }
}

impl Drop for AnonymousModel {
    fn drop(&mut self) {
        self.registrar.deregister(self.key.raw());
    }
}

/// The bus an agent dispatches through.
#[derive(Clone)]
pub(crate) struct AgentBus {
    dispatcher: Dispatcher,
    /// The registration handle for the same bus: what the agent's own
    /// registrations (models, memory, tools) go through once the driver is
    /// out of hand.
    registrar: Registrar,
    /// The owner segment of every key this agent mints
    /// (`<owner>/model:<label>`, `<owner>/memory`, ...).
    owner: Arc<str>,
    /// The agent's own driver, when it owns one. Behind an async mutex so
    /// concurrent runs on clones of one agent take turns driving: the run
    /// holding the guard serves every run's dispatches.
    driver: Option<Arc<Mutex<BusDriver>>>,
    /// The wakers of every live run; see [`Driven`].
    wakers: Arc<WakerSet>,
    recorder: Option<EffectLogRecorder>,
    anonymous_models: Arc<AtomicUsize>,
}

impl std::fmt::Debug for AgentBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBus")
            .field("owner", &self.owner)
            .field("owns_driver", &self.driver.is_some())
            .field("recording", &self.recorder.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentBus {
    pub(crate) fn owned(
        dispatcher: Dispatcher,
        registrar: Registrar,
        driver: BusDriver,
        owner: String,
    ) -> Self {
        Self {
            dispatcher,
            registrar,
            owner: Arc::from(owner),
            driver: Some(Arc::new(Mutex::new(driver))),
            wakers: Arc::new(WakerSet::default()),
            recorder: None,
            anonymous_models: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Install a recorder on the owned driver. Called at build, when the
    /// builder is the driver's only holder; a bus this agent does not own
    /// (or one another agent value already shares) cannot record, and says
    /// so.
    pub(crate) fn enable_recording(&mut self) -> Result<(), ErrorReport> {
        let Some(driver) = self.driver.as_mut() else {
            return Err(ErrorReport::new(
                rig_core::error::ErrorKind::Internal,
                "an agent over a host's bus does not record; tap the host's driver",
            ));
        };
        let Some(driver) = Arc::get_mut(driver) else {
            return Err(ErrorReport::new(
                rig_core::error::ErrorKind::Internal,
                "recording is enabled at build, before a clone shares the driver",
            ));
        };
        let recorder = EffectLogRecorder::new();
        driver.get_mut().record_to(recorder.clone());
        self.recorder = Some(recorder);
        Ok(())
    }

    pub(crate) fn over(dispatcher: Dispatcher, registrar: Registrar, owner: String) -> Self {
        Self {
            dispatcher,
            registrar,
            owner: Arc::from(owner),
            driver: None,
            wakers: Arc::new(WakerSet::default()),
            recorder: None,
            anonymous_models: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// This bus without its driver: what an agent keeps when
    /// [`Agent::into_parts`](super::Agent::into_parts) moves the driver
    /// out. The recorder stays, so the moved driver keeps recording into
    /// the agent's log.
    pub(crate) fn detached(&self) -> Self {
        Self {
            dispatcher: self.dispatcher.clone(),
            registrar: self.registrar.clone(),
            owner: self.owner.clone(),
            driver: None,
            wakers: Arc::new(WakerSet::default()),
            recorder: self.recorder.clone(),
            anonymous_models: self.anonymous_models.clone(),
        }
    }

    pub(crate) fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }

    pub(crate) fn registrar(&self) -> &Registrar {
        &self.registrar
    }

    /// The owner segment of the keys this agent mints.
    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    /// The wire key this agent mints for `suffix` (`model:<label>`,
    /// `memory`, `retrieve:context#<n>`).
    pub(crate) fn raw_key(&self, suffix: &str) -> HandlerKey {
        HandlerKey::from(format!("{}/{suffix}", self.owner))
    }

    /// The key this agent mints for `suffix`, typed by the family the
    /// builder registers under it. Minted, so asserted: the builder is the
    /// one that registers the handler and knows its family.
    pub(crate) fn key<F: rig_core::effect::Family>(&self, suffix: &str) -> Key<F> {
        Key::new_unchecked(self.raw_key(suffix))
    }

    /// The key this agent mints for the model labelled `label`.
    pub(crate) fn model_key(&self, label: &str) -> Key<family::Completion> {
        self.key(rig_core::bus::model_key(label).as_str())
    }

    /// The label under `key` when it is a model key this agent minted.
    pub(crate) fn model_label<'k>(&self, key: &'k HandlerKey) -> Option<&'k str> {
        key.as_str()
            .strip_prefix(&*self.owner)
            .and_then(|rest| rest.strip_prefix("/model:"))
    }

    /// Register `handler` under `key`: straight onto the driver while this
    /// value is its only holder (the builder's case), through the registrar
    /// otherwise.
    pub(crate) fn register_erased(
        &mut self,
        key: HandlerKey,
        handler: ErasedHandler,
    ) -> Result<(), ErrorReport> {
        match self.driver.as_mut().and_then(Arc::get_mut) {
            Some(driver) => driver.get_mut().register_erased(key, handler),
            None => self.registrar.register_erased(key, handler),
        }
    }

    pub(crate) fn owns_driver(&self) -> bool {
        self.driver.is_some()
    }

    /// The recorded log so far, when the agent records.
    pub(crate) fn effect_log(&self) -> Option<EffectLog> {
        self.recorder.as_ref().map(EffectLogRecorder::log)
    }

    /// Take the recorded log, when the agent records.
    pub(crate) fn take_effect_log(&self) -> Option<EffectLog> {
        self.recorder.as_ref().map(EffectLogRecorder::take)
    }

    /// Register `model` under `label` (replacing any model under it) and
    /// return the key a run selects it by.
    pub(crate) fn register_model<M>(&self, label: &ModelRef, model: M) -> Key<family::Completion>
    where
        M: CompletionModel + 'static,
    {
        let key = self.model_key(label.as_str());
        register_generated(
            self.registrar
                .register_typed::<family::Completion>(
                    key.raw().clone(),
                    CompletionAdapter::new(label.clone(), model),
                )
                .map(|_| ()),
        );
        key
    }

    /// Register `model` under a fresh generated label, scoped to the
    /// returned guard: the key leaves the bus when the last clone of the
    /// guard drops.
    pub(crate) fn register_anonymous_model<M>(&self, model: M) -> Arc<AnonymousModel>
    where
        M: CompletionModel + 'static,
    {
        let n = self.anonymous_models.fetch_add(1, Ordering::SeqCst);
        let key = self.register_model(&ModelRef::new(format!("anonymous#{n}")), model);
        Arc::new(AnonymousModel {
            key,
            registrar: self.registrar.clone(),
        })
    }

    /// Move the driver out. Fails when another clone of the agent still
    /// shares it — every clone drives, so the driver cannot leave while
    /// one of them may still run.
    pub(crate) fn try_into_parts(self) -> Result<(Dispatcher, BusDriver), Self> {
        let Some(driver) = self.driver else {
            return Err(self);
        };
        match Arc::try_unwrap(driver) {
            Ok(mutex) => Ok((self.dispatcher, mutex.into_inner())),
            Err(driver) => Err(Self {
                driver: Some(driver),
                ..self
            }),
        }
    }

    /// Drive the agent's driver for as long as `inner` runs: every poll of
    /// the returned stream that leaves `inner` pending polls the driver too,
    /// so every dispatch the run makes — from the engine, a hook, a tool —
    /// is served by a run awaiting it. Over a host's bus this is `inner`
    /// unchanged.
    pub(crate) fn drive<S>(&self, inner: S) -> Driven<S> {
        Driven {
            inner: Some(inner),
            driver: self.driver.clone(),
            wakers: Arc::clone(&self.wakers),
            slot: self.wakers.slot(),
        }
    }
}

/// The agent's generated keys are family-prefixed (`model:`, `tool:`,
/// `memory`, `retrieve:`), so a registration under one can never change
/// the key's family — the one thing `Dispatcher::register` refuses. The
/// refusal is therefore unreachable here; it is asserted in debug builds
/// and logged, never swallowed silently, in release.
#[track_caller]
pub(crate) fn register_generated(registered: Result<(), rig_core::error::ErrorReport>) {
    if let Err(report) = registered {
        let caller = std::panic::Location::caller();
        debug_assert!(
            false,
            "a generated key changed family at {caller}: {report}"
        );
        tracing::error!(
            %report,
            %caller,
            "a generated bus key changed family; the registration was refused"
        );
    }
}

/// A stream that drives the agent's bus driver whenever it is pending.
///
/// **Whoever is awaiting drives.** The driver sits behind a mutex that is
/// only ever held for the duration of one synchronous poll: every `Driven`
/// that is polled and finds its run pending takes the lock if it is free,
/// polls the driver once, and releases it. A `Driven` that finds the lock
/// taken is being polled from *inside* another `Driven`'s driver poll (a
/// tool that runs a nested prompt on a clone) and simply yields — the
/// poll in progress serves its dispatches too.
///
/// The driver is polled with a waker that wakes *every* live `Driven` on
/// this bus, so progress inside the driver (a provider reply, a timer, a
/// channel send) reaches whichever run is awaiting it, even when the run
/// that last polled the driver has since finished or been dropped. That is
/// what keeps a finished stream in scope, two streams polled alternately,
/// and a `prompt()` awaited inside a `while let` over a stream from
/// starving each other: none of them owns the driver.
pub(crate) struct Driven<S> {
    /// `None` once the run finished or while dropping: the run is released
    /// before the driver's last poll, so its abandoned dispatches read as
    /// cancelled.
    inner: Option<S>,
    /// The agent's driver, when it owns one. Over a host's bus this is
    /// `None` and the wrapper is `inner` unchanged.
    driver: Option<Arc<Mutex<BusDriver>>>,
    /// The bus-wide set of wakers the driver is polled with.
    wakers: Arc<WakerSet>,
    /// This run's slot in `wakers`.
    slot: u64,
}

impl<S> Driven<S> {
    /// The run is over: it neither drives nor needs waking any more.
    fn finish(&mut self) {
        self.inner = None;
        self.wakers.unregister(self.slot);
    }

    /// Register `cx`'s waker so driver progress under any other run's poll
    /// wakes this one, then poll the driver once if nobody else is polling
    /// it right now.
    fn poll_driver(&mut self, cx: &Context<'_>) {
        let Some(driver) = &self.driver else {
            return;
        };
        self.wakers.register(self.slot, cx.waker());
        if let Some(mut guard) = driver.try_lock() {
            let waker = Waker::from(Arc::clone(&self.wakers));
            let mut driver_cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut *guard).poll(&mut driver_cx);
        }
    }
}

impl<S> Drop for Driven<S> {
    fn drop(&mut self) {
        // A dropped run is a cancelled run: release it first (its pending
        // dispatches drop their cancel guards), then give the driver one last
        // poll so every abandoned dispatch is observed as cancelled now — its
        // handler future (and the provider call or stream inside) drops here
        // rather than when the next run happens to drive. The poll uses the
        // bus-wide waker, never a noop one: the driver's internals must keep
        // waking the runs that are still live. A run that already finished
        // has nothing to cancel.
        let was_live = self.inner.take().is_some();
        self.wakers.unregister(self.slot);
        if !was_live {
            return;
        }
        if let Some(driver) = &self.driver
            && let Some(mut guard) = driver.try_lock()
        {
            let waker = Waker::from(Arc::clone(&self.wakers));
            let mut driver_cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut *guard).poll(&mut driver_cx);
        }
    }
}

impl<S: Stream + Unpin> Stream for Driven<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = &mut this.inner else {
            return Poll::Ready(None);
        };
        match Pin::new(&mut *inner).poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => {
                this.poll_driver(cx);
                // The driver may have served the reply the inner stream
                // waits on; give it one more poll before yielding.
                let Some(inner) = &mut this.inner else {
                    return Poll::Ready(None);
                };
                match Pin::new(&mut *inner).poll_next(cx) {
                    Poll::Ready(None) => {
                        this.finish();
                        Poll::Ready(None)
                    }
                    other => other,
                }
            }
        }
    }
}

impl<S: Unpin> Unpin for Driven<S> {}

/// The wakers of every live run on one agent bus. The driver is polled with
/// a waker built from this set, so its progress wakes every run that may be
/// waiting on it; each run keeps its own slot current on every poll.
#[derive(Default)]
pub(crate) struct WakerSet {
    slots: std::sync::Mutex<Vec<(u64, Waker)>>,
    next_slot: AtomicU64,
}

impl WakerSet {
    fn slot(&self) -> u64 {
        self.next_slot.fetch_add(1, Ordering::Relaxed)
    }

    fn register(&self, slot: u64, waker: &Waker) {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match slots.iter_mut().find(|(id, _)| *id == slot) {
            Some((_, existing)) => {
                if !existing.will_wake(waker) {
                    existing.clone_from(waker);
                }
            }
            None => slots.push((slot, waker.clone())),
        }
    }

    fn unregister(&self, slot: u64) {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slots.retain(|(id, _)| *id != slot);
    }
}

impl Wake for WakerSet {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // Clone out first: a woken task may poll and re-register on this
        // same set from another thread.
        let wakers: Vec<Waker> = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|(_, waker)| waker.clone())
            .collect();
        for waker in wakers {
            waker.wake();
        }
    }
}
