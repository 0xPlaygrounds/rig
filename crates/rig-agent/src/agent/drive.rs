//! Agent-owned bus driving, scoped model registrations, and shared run wakeups.
//! Runs poll owned drivers without retaining a lock between polls; agents over
//! host buses leave driving to the host.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures::Stream;

use crate::sync::Mutex;

#[cfg(all(test, rig_loom))]
mod loom_models;
#[cfg(all(test, not(rig_loom)))]
mod tests;
use crate::bus::{BusDriver, Dispatcher, Recording, Registrar};
use rig_core::serve::ErasedHandler;
use rig_core::serve::ServingPolicy;
use rig_core::serve::adapters::CompletionAdapter;
use rig_core::{
    completion::{CompletionModel, ModelRef},
    effect::{HandlerKey, Key, family},
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
    /// Owned driver protected by a synchronous mutex for one poll at a time.
    /// Each polling run can serve dispatches from every run sharing the bus.
    driver: Option<Arc<Mutex<BusDriver>>>,
    /// The wakers of every live run; see [`Driven`].
    wakers: Arc<WakerSet>,
    anonymous_models: Arc<AtomicUsize>,
    /// The policy the owned bus was created with; `None` over a host's bus.
    config: Option<ServingPolicy>,
}

impl std::fmt::Debug for AgentBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBus")
            .field("owner", &self.owner)
            .field("owns_driver", &self.driver.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentBus {
    pub(crate) fn owned(
        dispatcher: Dispatcher,
        registrar: Registrar,
        driver: BusDriver,
        owner: String,
        config: ServingPolicy,
    ) -> Self {
        Self {
            dispatcher,
            registrar,
            owner: Arc::from(owner),
            driver: Some(Arc::new(Mutex::new(driver))),
            wakers: Arc::new(WakerSet::default()),
            anonymous_models: Arc::new(AtomicUsize::new(0)),
            config: Some(config),
        }
    }

    /// The policy the owned bus runs under; `None` over a host's bus.
    pub(crate) fn config(&self) -> Option<ServingPolicy> {
        self.config
    }

    /// Install a recorder on the owned driver. Called at build, when the
    /// builder is the driver's only holder; a bus this agent does not own
    /// (or one another agent value already shares) cannot record, and says
    /// so.
    pub(crate) fn record_to(&mut self, recorder: Recording) -> Result<(), ErrorReport> {
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
        driver
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record_with(recorder);
        Ok(())
    }

    pub(crate) fn over(dispatcher: Dispatcher, registrar: Registrar, owner: String) -> Self {
        Self {
            dispatcher,
            registrar,
            owner: Arc::from(owner),
            driver: None,
            wakers: Arc::new(WakerSet::default()),
            anonymous_models: Arc::new(AtomicUsize::new(0)),
            config: None,
        }
    }

    /// This bus without its driver: what an agent keeps when
    /// [`Agent::into_parts`](super::Agent::into_parts) moves the driver
    /// out. Recording belongs to that driver and remains active after the move.
    pub(crate) fn detached(&self) -> Self {
        Self {
            dispatcher: self.dispatcher.clone(),
            registrar: self.registrar.clone(),
            owner: self.owner.clone(),
            driver: None,
            wakers: Arc::new(WakerSet::default()),
            anonymous_models: self.anonymous_models.clone(),
            config: self.config,
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
        self.key(rig_core::effect::model_key(label).as_str())
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
            Some(driver) => driver
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .register_erased(key, handler),
            None => self.registrar.register_erased(key, handler),
        }
    }

    pub(crate) fn owns_driver(&self) -> bool {
        self.driver.is_some()
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

    /// Move out an exclusively owned driver and dispatcher. Returns this bus
    /// unchanged if it has no driver or another value still shares the driver.
    pub(crate) fn try_into_parts(self) -> Result<(Dispatcher, BusDriver), Self> {
        let Some(driver) = self.driver else {
            return Err(self);
        };
        match Arc::try_unwrap(driver) {
            Ok(mutex) => Ok((
                self.dispatcher,
                mutex
                    .into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )),
            Err(driver) => Err(Self {
                driver: Some(driver),
                ..self
            }),
        }
    }

    /// Wrap a stream to poll the owned driver when the stream is pending.
    /// Over a host bus, delegates to the inner stream without driving.
    pub(crate) fn drive<S>(&self, inner: S) -> Driven<S> {
        Driven {
            inner: Some(inner),
            driver: self.driver.clone(),
            wakers: Arc::clone(&self.wakers),
            slot: self.wakers.slot(),
        }
    }
}

/// Check a generated-key registration expected to preserve its family.
/// Panics on refusal in debug builds; logs the error in release builds.
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

/// Stream wrapper polling the shared driver while pending. Holds the driver lock
/// only during a synchronous poll and yields on contention, including reentrant
/// polls. A bus-wide waker notifies all live runs so progress does not depend on
/// the last polling run remaining alive.
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
    /// The run is over: it neither drives nor needs waking any more. The
    /// runs still registered are woken: one of them may have found the
    /// driver lock taken under this run's last poll and buffered a command
    /// since that poll's drain, and nothing polls the driver between runs.
    fn finish(&mut self) {
        self.inner = None;
        self.wakers.unregister(self.slot);
        self.wakers.wake_by_ref();
    }

    /// Poll cancellations until the in-flight count stops decreasing, without
    /// waiting for live dispatches. Owned drivers are not polled between runs,
    /// so abandoned work must be settled before releasing the final run.
    fn settle_in_flight(&mut self, cx: &Context<'_>) {
        let Some(driver) = self.driver.clone() else {
            return;
        };
        let mut before = usize::MAX;
        loop {
            let in_flight = match try_lock(&driver) {
                Some(guard) => guard.in_flight(),
                None => return,
            };
            if in_flight == 0 || in_flight >= before {
                return;
            }
            before = in_flight;
            self.poll_driver(cx);
        }
    }

    /// Register `cx`'s waker so driver progress under any other run's poll
    /// wakes this one, then poll the driver once if nobody else is polling
    /// it right now.
    fn poll_driver(&mut self, cx: &Context<'_>) {
        let Some(driver) = &self.driver else {
            return;
        };
        self.wakers.register(self.slot, cx.waker());
        if let Some(mut guard) = try_lock(driver) {
            let waker = Waker::from(Arc::clone(&self.wakers));
            let mut driver_cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut *guard).poll(&mut driver_cx);
        }
    }
}

/// Acquire without blocking, recovering poisoned locks. Returns `None` on
/// contention so concurrent or nested driver polls can yield.
fn try_lock<T>(driver: &Mutex<T>) -> Option<crate::sync::MutexGuard<'_, T>> {
    match driver.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

impl<S> Drop for Driven<S> {
    fn drop(&mut self) {
        // Drop the run before polling so abandoned dispatches expose cancellation.
        // Keep the bus-wide waker so surviving runs still receive driver progress.
        let was_live = self.inner.take().is_some();
        self.wakers.unregister(self.slot);
        if !was_live {
            return;
        }
        if let Some(driver) = &self.driver
            && let Some(mut guard) = try_lock(driver)
        {
            let waker = Waker::from(Arc::clone(&self.wakers));
            let mut driver_cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut *guard).poll(&mut driver_cx);
        }
        // As at `finish`: a run that found the lock taken under this poll
        // takes over once it is released.
        self.wakers.wake_by_ref();
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
                this.settle_in_flight(cx);
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
                        this.settle_in_flight(cx);
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
    slots: Mutex<Vec<(u64, Waker)>>,
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
