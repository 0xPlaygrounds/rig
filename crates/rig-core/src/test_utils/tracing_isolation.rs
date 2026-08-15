//! Serialization for tests that install scoped tracing subscribers.

use tokio::sync::{Mutex, MutexGuard};

static GUARD: Mutex<()> = Mutex::const_new(());

/// Serializes tests that install scoped tracing subscribers
/// (`tracing::subscriber::set_default` / `with_default`).
///
/// `tracing` caches per-callsite interest **globally**, and the first thread to
/// hit a callsite computes that interest from its own thread's dispatcher. A
/// test running in parallel without a subscriber can therefore cache
/// `Interest::never` for callsites a capturing test relies on, and dispatcher
/// registration churn rebuilds the cache at arbitrary times.
///
/// Holding this guard for the subscriber's whole lifetime is necessary but
/// **not sufficient**: it serializes the tests that *take* it, and cannot stop
/// an unguarded sibling on another thread from re-caching `Interest::never`
/// mid-test. Two further rules apply.
///
/// # 1. Warm, then rebuild — in that order
///
/// Under the guard, run the code path once before asserting on it, then
/// rebuild the interest cache, then run it again for real:
///
/// ```ignore
/// let _isolation = scoped_tracing_subscriber_guard().await;
/// let _default = tracing::subscriber::set_default(subscriber);
///
/// // (1) Warm: first-register this path's callsites against THIS subscriber.
/// run_the_path().await;
/// // (2) Rebuild: heal callsites a foreign thread already poisoned.
/// tracing::callsite::rebuild_interest_cache();
/// captured.clear();
///
/// // (3) The run being asserted on.
/// run_the_path().await;
/// ```
///
/// The rebuild alone is not enough, and the difference is measurable rather
/// than theoretical: with sibling tests hammering the same callsite, a
/// rebuild-only capture came back empty in 3 of 52 runs, while warm-then-rebuild
/// passed 52 of 52 (rig#2346). Both steps must sit inside the guard.
///
/// # 2. An absence assertion needs a positive anchor
///
/// The two failure directions are not equally visible. A poisoned callsite
/// captures *nothing*, so:
///
/// | assertion shape | under a poisoned callsite |
/// | --- | --- |
/// | `assert!(logs.contains(X))` | fails loudly — flaky red |
/// | `assert!(!logs.contains(X))` | **passes vacuously** — the test is dead |
///
/// So an assertion that something is **absent** from captured output is only
/// sound where something else proves the capture was live. Assert absence
/// beside a positive anchor, or isolate the test in its own binary.
///
/// # Worked examples
///
/// - Log capture: `crates/rig-core/tests/common/tracing_capture.rs` packages
///   all of the above, and `crates/rig-core/tests/model_listing_warnings.rs`
///   uses it.
/// - Span capture: `assert_stream_usage_recorded_on_chat_spans` in
///   `crates/rig-agent/src/agent/prompt_request/streaming.rs`.
pub async fn scoped_tracing_subscriber_guard() -> MutexGuard<'static, ()> {
    GUARD.lock().await
}

/// Blocking variant of [`scoped_tracing_subscriber_guard`] for synchronous
/// tests.
///
/// The same two rules apply — see [`scoped_tracing_subscriber_guard`].
pub fn scoped_tracing_subscriber_guard_blocking() -> MutexGuard<'static, ()> {
    GUARD.blocking_lock()
}
