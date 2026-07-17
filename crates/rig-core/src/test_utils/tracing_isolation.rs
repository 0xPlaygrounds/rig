//! Serialization for tests that install scoped tracing subscribers.

use tokio::sync::{Mutex, MutexGuard};

static GUARD: Mutex<()> = Mutex::const_new(());

/// Serializes tests that install scoped tracing subscribers
/// (`tracing::subscriber::set_default` / `with_default`).
///
/// `tracing` caches per-callsite interest globally, and the first thread to
/// hit a callsite computes that interest from its own thread's dispatcher. A
/// test running in parallel without a subscriber can therefore cache
/// `Interest::never` for callsites a span-asserting test relies on, and
/// dispatcher registration churn rebuilds the cache at arbitrary times. Any
/// test that installs a scoped subscriber and asserts on captured spans must
/// hold this guard for the subscriber's whole lifetime (and warm/rebuild the
/// callsites it asserts on; see `assert_stream_usage_recorded_on_chat_spans`).
pub async fn scoped_tracing_subscriber_guard() -> MutexGuard<'static, ()> {
    GUARD.lock().await
}

/// Blocking variant of [`scoped_tracing_subscriber_guard`] for synchronous
/// tests.
pub fn scoped_tracing_subscriber_guard_blocking() -> MutexGuard<'static, ()> {
    GUARD.blocking_lock()
}
