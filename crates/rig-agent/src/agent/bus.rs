//! The agent's bus: the dispatcher every run dispatches through, the driver
//! the agent drives inline while a run is awaited, and the recorder that
//! taps it.
//!
//! **Whoever holds the driver drives.** An agent built with
//! [`AgentBuilder::new`](super::AgentBuilder::new) holds its own driver and
//! drives it for the life of every run it produces ([`AgentBus::drive`]);
//! it never hands its dispatcher out on its own — [`Agent::into_parts`]
//! (super::Agent::into_parts) moves the driver out together with it. An
//! agent built over a host's bus ([`AgentBuilder::over_bus`]
//! (super::AgentBuilder::over_bus)) holds no driver: the host drives.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures::{
    Stream,
    lock::{Mutex, OwnedMutexGuard, OwnedMutexLockFuture},
};
use rig_core::{
    bus::{BusDriver, Dispatcher, EffectLogRecorder, adapters::CompletionAdapter, model_key},
    completion::{CompletionModel, ModelRef},
    effect::{EffectLog, HandlerKey},
};

/// The bus an agent dispatches through.
#[derive(Clone)]
pub(crate) struct AgentBus {
    dispatcher: Dispatcher,
    /// The agent's own driver, when it owns one. Behind an async mutex so
    /// concurrent runs on clones of one agent take turns driving: the run
    /// holding the guard serves every run's dispatches.
    driver: Option<Arc<Mutex<BusDriver>>>,
    recorder: Option<EffectLogRecorder>,
    anonymous_models: Arc<AtomicUsize>,
}

impl std::fmt::Debug for AgentBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBus")
            .field("owns_driver", &self.driver.is_some())
            .field("recording", &self.recorder.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentBus {
    pub(crate) fn owned(dispatcher: Dispatcher, driver: BusDriver) -> Self {
        Self {
            dispatcher,
            driver: Some(Arc::new(Mutex::new(driver))),
            recorder: None,
            anonymous_models: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Install a recorder on the owned driver (before any run drives it).
    pub(crate) fn enable_recording(&mut self) {
        let recorder = EffectLogRecorder::new();
        if let Some(driver) = &self.driver
            && let Some(mut guard) = driver.try_lock()
        {
            guard.record_to(recorder.clone());
            self.recorder = Some(recorder);
        }
    }

    pub(crate) fn over(dispatcher: Dispatcher) -> Self {
        Self {
            dispatcher,
            driver: None,
            recorder: None,
            anonymous_models: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
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
    pub(crate) fn register_model<M>(&self, label: &ModelRef, model: M) -> HandlerKey
    where
        M: CompletionModel + 'static,
    {
        let key = model_key(label.as_str());
        self.dispatcher
            .register(key.clone(), CompletionAdapter::new(label.clone(), model));
        key
    }

    /// Register `model` under a fresh generated label.
    pub(crate) fn register_anonymous_model<M>(&self, model: M) -> HandlerKey
    where
        M: CompletionModel + 'static,
    {
        let n = self.anonymous_models.fetch_add(1, Ordering::SeqCst);
        self.register_model(&ModelRef::new(format!("anonymous-{n}")), model)
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
    /// is served by the run awaiting it. Over a host's bus this is `inner`
    /// unchanged.
    pub(crate) fn drive<S>(&self, inner: S) -> Driven<S> {
        Driven {
            inner: Some(inner),
            driver: self.driver.clone(),
            lock: None,
            guard: None,
        }
    }
}

/// A stream that drives the agent's bus driver whenever it is pending.
pub(crate) struct Driven<S> {
    /// `None` only while dropping: the run is released before the driver's
    /// last poll, so its abandoned dispatches read as cancelled.
    inner: Option<S>,
    driver: Option<Arc<Mutex<BusDriver>>>,
    lock: Option<OwnedMutexLockFuture<BusDriver>>,
    guard: Option<OwnedMutexGuard<BusDriver>>,
}

impl<S> Driven<S> {
    fn poll_driver(&mut self, cx: &mut Context<'_>) {
        let Some(driver) = &self.driver else {
            return;
        };
        if self.guard.is_none() {
            if self.lock.is_none() {
                self.lock = Some(Arc::clone(driver).lock_owned());
            }
            let Some(lock) = &mut self.lock else {
                return;
            };
            match Pin::new(lock).poll(cx) {
                Poll::Ready(guard) => {
                    self.guard = Some(guard);
                    self.lock = None;
                }
                Poll::Pending => return,
            }
        }
        if let Some(guard) = &mut self.guard {
            let _ = Pin::new(&mut **guard).poll(cx);
        }
    }
}

impl<S> Drop for Driven<S> {
    fn drop(&mut self) {
        // A dropped run is a cancelled run: release it first (its pending
        // dispatches drop their cancel guards), then give the driver one last
        // poll so every abandoned dispatch is observed as cancelled now — its
        // handler future (and the provider call or stream inside) drops here
        // rather than when the next run happens to drive.
        self.inner = None;
        if let Some(guard) = &mut self.guard {
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut **guard).poll(&mut cx);
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
            Poll::Ready(item) => Poll::Ready(item),
            Poll::Pending => {
                this.poll_driver(cx);
                // The driver may have served the reply the inner stream
                // waits on; give it one more poll before yielding.
                let Some(inner) = &mut this.inner else {
                    return Poll::Ready(None);
                };
                Pin::new(&mut *inner).poll_next(cx)
            }
        }
    }
}

impl<S: Unpin> Unpin for Driven<S> {}
