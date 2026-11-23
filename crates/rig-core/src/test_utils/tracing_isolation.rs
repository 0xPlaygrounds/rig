//! Serialization for tests that install scoped tracing subscribers.

use futures::lock::{Mutex, MutexGuard};
use std::sync::LazyLock;

// An async mutex: the guard is deliberately held across awaits for a test's
// whole lifetime, which a std mutex could not do soundly.
static GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Serializes tests that install scoped tracing subscribers
/// (`tracing::subscriber::set_default` / `with_default`).
///
/// `tracing` caches per-callsite interest globally, from the dispatcher of
/// the first thread to hit a callsite, so an unguarded test on another thread
/// can cache `Interest::never` for a callsite a capturing test relies on.
/// Holding this guard for the subscriber's whole lifetime is not enough on
/// its own. Under the guard, run the path once, call
/// `tracing::callsite::rebuild_interest_cache()`, clear the capture, then run
/// the path being asserted on:
///
/// ```ignore
/// let _isolation = scoped_tracing_subscriber_guard().await;
/// let _default = tracing::subscriber::set_default(subscriber);
/// run_the_path().await;
/// tracing::callsite::rebuild_interest_cache();
/// captured.clear();
/// run_the_path().await;
/// ```
///
/// A poisoned callsite captures nothing, so an assertion that something is
/// absent passes vacuously. Assert absence only beside a positive anchor that
/// proves the capture was live, or isolate the test in its own binary.
pub async fn scoped_tracing_subscriber_guard() -> MutexGuard<'static, ()> {
    GUARD.lock().await
}

/// Blocking variant of [`scoped_tracing_subscriber_guard`] for synchronous
/// tests, under the same rules.
pub fn scoped_tracing_subscriber_guard_blocking() -> MutexGuard<'static, ()> {
    futures::executor::block_on(GUARD.lock())
}
