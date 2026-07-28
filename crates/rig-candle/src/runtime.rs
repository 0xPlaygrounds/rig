//! Validated execution-device state shared by loading and generation.

use candle_core::{DType, Device};

#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;
#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::CandleError;

/// CPU execution state owned by a loaded model.
///
/// Keeping the tensor device and KV-cache dtype together prevents loading and
/// generation paths from silently selecting incompatible defaults.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeDevice {
    device: Device,
    cache_dtype: DType,
}

#[cfg(all(test, not(target_family = "wasm")))]
pub(crate) struct TestControl {
    gate: std::sync::Mutex<bool>,
    gate_changed: std::sync::Condvar,
    entered: AtomicBool,
    entered_notification: tokio::sync::Notify,
    panic_after_gate: AtomicBool,
    delivery_attempts: std::sync::atomic::AtomicUsize,
    delivery_notification: tokio::sync::Notify,
}

#[cfg(all(test, not(target_family = "wasm")))]
impl TestControl {
    pub(crate) fn new(blocked: bool, panic_after_gate: bool) -> Self {
        Self {
            gate: std::sync::Mutex::new(blocked),
            gate_changed: std::sync::Condvar::new(),
            entered: AtomicBool::new(false),
            entered_notification: tokio::sync::Notify::new(),
            panic_after_gate: AtomicBool::new(panic_after_gate),
            delivery_attempts: std::sync::atomic::AtomicUsize::new(0),
            delivery_notification: tokio::sync::Notify::new(),
        }
    }

    pub(crate) fn enter_generation(&self) -> Result<(), CandleError> {
        self.entered.store(true, Ordering::Release);
        self.entered_notification.notify_waiters();
        let mut blocked = self
            .gate
            .lock()
            .map_err(|_| CandleError::Inference("test generation gate was poisoned".to_string()))?;
        while *blocked {
            blocked = self.gate_changed.wait(blocked).map_err(|_| {
                CandleError::Inference("test generation gate was poisoned".to_string())
            })?;
        }
        if self.panic_after_gate.load(Ordering::Acquire) {
            std::panic::resume_unwind(Box::new("intentional blocking-task test panic"));
        }
        Ok(())
    }

    pub(crate) async fn wait_until_entered(&self) {
        loop {
            let notified = self.entered_notification.notified();
            if self.entered.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    pub(crate) fn release(&self) -> Result<(), CandleError> {
        let mut blocked = self
            .gate
            .lock()
            .map_err(|_| CandleError::Inference("test generation gate was poisoned".to_string()))?;
        *blocked = false;
        self.gate_changed.notify_all();
        Ok(())
    }

    pub(crate) fn record_delivery_attempt(&self) {
        self.delivery_attempts.fetch_add(1, Ordering::AcqRel);
        self.delivery_notification.notify_waiters();
    }

    pub(crate) fn delivery_attempt_count(&self) -> usize {
        self.delivery_attempts.load(Ordering::Acquire)
    }

    pub(crate) async fn wait_for_delivery_attempts(&self, expected: usize) {
        loop {
            let notified = self.delivery_notification.notified();
            if self.delivery_attempts.load(Ordering::Acquire) >= expected {
                return;
            }
            notified.await;
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Default)]
pub(crate) struct CancellationSignal(Arc<AtomicBool>);

#[cfg(not(target_family = "wasm"))]
impl CancellationSignal {
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Default)]
pub(crate) struct CancellationSignal;

#[cfg(target_family = "wasm")]
impl CancellationSignal {
    pub(crate) fn is_cancelled(&self) -> bool {
        false
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) struct CancelOnDrop {
    signal: CancellationSignal,
    armed: bool,
}

#[cfg(not(target_family = "wasm"))]
impl CancelOnDrop {
    pub(crate) fn new(signal: CancellationSignal) -> Self {
        Self {
            signal,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(not(target_family = "wasm"))]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.signal.cancel();
        }
    }
}

pub(crate) fn check_cancellation(signal: &CancellationSignal) -> Result<(), CandleError> {
    if signal.is_cancelled() {
        Err(CandleError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) async fn acquire_concurrency(
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<tokio::sync::OwnedSemaphorePermit, CandleError> {
    semaphore
        .acquire_owned()
        .await
        .map_err(|_| CandleError::ConcurrencyControllerClosed)
}

impl RuntimeDevice {
    pub(crate) fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            cache_dtype: DType::F32,
        }
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn cache_dtype(&self) -> DType {
        self.cache_dtype
    }

    #[cfg(test)]
    pub(crate) fn is_consistent_cpu(&self) -> bool {
        self.device.is_cpu() && self.cache_dtype == DType::F32
    }
}
