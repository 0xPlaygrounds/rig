//! Target-conditional portability bounds for Rig's async public APIs.
//!
//! "Maybe" means target-conditional, not runtime-optional. On browser WASM
//! (`all(target_arch = "wasm32", target_os = "unknown")`) the marker traits do
//! not require thread-safety and [`BoxFuture`] is local. Every other target,
//! including WASI, retains ordinary `Send`/`Sync` guarantees and sendable boxed
//! futures.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Send` except on browser WASM, where this is a no-op marker.
///
/// The relaxed target is exactly `wasm32-unknown-unknown`; this trait still
/// requires `Send` on WASI and all other targets.
pub trait MaybeSend: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Send` except on browser WASM, where this is a no-op marker.
///
/// The relaxed target is exactly `wasm32-unknown-unknown`; this trait still
/// requires `Send` on WASI and all other targets.
pub trait MaybeSend {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> MaybeSend for T where T: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> MaybeSend for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Sync` except on browser WASM, where this is a no-op marker.
///
/// The relaxed target is exactly `wasm32-unknown-unknown`; this trait still
/// requires `Sync` on WASI and all other targets.
pub trait MaybeSync: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Sync` except on browser WASM, where this is a no-op marker.
///
/// The relaxed target is exactly `wasm32-unknown-unknown`; this trait still
/// requires `Sync` on WASI and all other targets.
pub trait MaybeSync {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> MaybeSync for T where T: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> MaybeSync for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// A sendable boxed future on every target except browser WASM.
///
/// This is [`futures::future::BoxFuture`] on native, WASI, and other
/// non-browser targets.
pub use futures::future::BoxFuture;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// A boxed future that may be non-`Send` on browser WASM.
///
/// This re-exports [`futures::future::LocalBoxFuture`] as `BoxFuture` exactly
/// on `wasm32-unknown-unknown`.
pub use futures::future::LocalBoxFuture as BoxFuture;

// This module is compiled by an ordinary browser-WASM `cargo check`, so it
// guards the relaxed marker and local-future contract without requiring a WASM
// test runner.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod browser_wasm_contract {
    use std::rc::Rc;

    use super::{BoxFuture, MaybeSend, MaybeSync};

    fn accepts_maybe_thread_safe<T: MaybeSend + MaybeSync>(_: &T) {}

    #[allow(dead_code)]
    fn accepts_rc_and_local_future() {
        let state = Rc::new(());
        accepts_maybe_thread_safe(&state);

        let future: BoxFuture<'static, Rc<()>> = Box::pin(async move { state });
        drop(future);
    }
}

/// Error returned by [`timeout`] when the future does not complete in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("future timed out")
    }
}

impl std::error::Error for Elapsed {}

/// Await `future`, returning `Err(`[`Elapsed`]`)` if it does not complete within
/// `duration`.
///
/// A cross-platform (native + wasm) replacement for `tokio::time::timeout`: rig's
/// `tokio` dependency is built without the `time` feature, and `tokio::time` does
/// not function on wasm. This is built on [`futures_timer::Delay`], which rig
/// already uses for SSE retry backoff.
///
/// On elapse the pending `future` is **dropped** (cancelled by drop); it gets no
/// chance to run cleanup beyond its own `Drop`. A zero or already-elapsed
/// `duration` still polls `future` once before electing `Elapsed`, and an absurdly
/// large `duration` may panic when added to `Instant::now()` inside the timer.
///
/// # Wasm
/// On browser wasm (`wasm32-unknown-unknown`) the `futures-timer` `wasm-bindgen`
/// (`setTimeout`) backend is selected automatically via a target-scoped
/// dependency, so the timer fires without depending on any cargo feature. (The
/// `futures_timer::Delay` SSE retry backoff relies on the same backend.)
pub async fn timeout<F>(duration: std::time::Duration, future: F) -> Result<F::Output, Elapsed>
where
    F: Future,
{
    use futures::future::{Either, select};

    let delay = futures_timer::Delay::new(duration);
    futures::pin_mut!(future);
    futures::pin_mut!(delay);
    match select(future, delay).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(((), _)) => Err(Elapsed),
    }
}

#[macro_export]
macro_rules! if_wasm {
    ($($tokens:tt)*) => {
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        $($tokens)*

    };
}

#[macro_export]
macro_rules! if_not_wasm {
    ($($tokens:tt)*) => {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        $($tokens)*

    };
}

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod tests {
    use super::{BoxFuture, Elapsed, MaybeSend, MaybeSync, timeout};
    use std::time::Duration;

    fn maybe_send_implies_send<T: MaybeSend>() {
        fn assert_send<T: Send>() {}
        assert_send::<T>();
    }

    fn maybe_sync_implies_sync<T: MaybeSync>() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<T>();
    }

    #[test]
    fn native_compatibility_types_retain_thread_safety() {
        fn assert_send<T: Send>() {}

        maybe_send_implies_send::<String>();
        maybe_sync_implies_sync::<String>();
        assert_send::<BoxFuture<'static, ()>>();
    }

    #[tokio::test]
    async fn timeout_returns_ok_for_a_future_that_completes_in_time() {
        let result = timeout(Duration::from_secs(5), async { 42 }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn timeout_returns_elapsed_for_a_future_that_never_completes() {
        let result = timeout(Duration::from_millis(20), std::future::pending::<()>()).await;
        assert_eq!(result, Err(Elapsed));
    }

    #[tokio::test]
    async fn timeout_zero_duration_still_polls_a_ready_future_once() {
        // Documented contract: a zero/already-elapsed duration still polls the
        // future once before electing `Elapsed`, so a ready future wins.
        let result = timeout(Duration::ZERO, async { 7 }).await;
        assert_eq!(result, Ok(7));
    }
}
