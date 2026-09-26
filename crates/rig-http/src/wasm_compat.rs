//! Target-dependent thread bounds, boxed futures, and executor-independent timers.
//!
//! ```
//! use rig_http::wasm_compat::BoxFuture;
//! let future: BoxFuture<'_, u32> = Box::pin(async { 42 });
//! ```

use std::pin::Pin;

use futures::Stream;

// Browser markers assume single-threaded execution; atomics would invalidate
// that assumption. Relaxed bounds do not make non-Send handlers thread-safe.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    target_feature = "atomics"
))]
compile_error!(
    "rig-http does not support threaded wasm (`+atomics`): its wasm-compat markers assume a \
     single-threaded target"
);

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Send` on native targets, a no-op marker on browser wasm.
///
/// ```compile_fail
/// use std::rc::Rc;
/// use rig_http::wasm_compat::MaybeSend;
///
/// fn require_send<T: MaybeSend>(_: T) {}
/// require_send(Rc::new(1));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Send`, as required on native targets",
    label = "not `Send`",
    note = "use thread-safe ownership such as `Arc`, not `Rc`; browser wasm permits thread-local values"
)]
pub trait MaybeSend: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Send` on native targets, a no-op marker on browser wasm.
pub trait MaybeSend {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T: ?Sized> MaybeSend for T where T: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T: ?Sized> MaybeSend for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// A boxed stream that can move between native executor threads.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// A boxed browser stream, permitting JavaScript's thread-local handles.
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Sync` on native targets, a no-op marker on browser wasm.
///
/// ```compile_fail
/// use std::cell::Cell;
/// use rig_http::wasm_compat::MaybeSync;
///
/// fn require_sync<T: MaybeSync>(_: T) {}
/// require_sync(Cell::new(1));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Sync`, as required on native targets",
    label = "not `Sync`",
    note = "use `Mutex` or atomics instead of `Cell` or `RefCell`; browser wasm permits thread-local values"
)]
pub trait MaybeSync: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Sync` on native targets, a no-op marker on browser wasm.
pub trait MaybeSync {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T: ?Sized> MaybeSync for T where T: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T: ?Sized> MaybeSync for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// Boxed future with `Send` on the same targets as [`MaybeSend`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Boxed future type without `Send`, on browser wasm.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Error returned by [`timeout`] when the future does not complete in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("future timed out")
    }
}

impl std::error::Error for Elapsed {}

/// Await `future` until `duration` elapses, then drop it and return [`Elapsed`].
/// The future is polled first, even for zero duration; cancellation runs only
/// its drop cleanup. Uses a native timer thread or browser `setTimeout`, without
/// requiring an executor timer or a feature flag.
///
/// # Panics
/// May panic if the timer cannot represent the deadline for `duration`.
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

/// Sleep for `duration` using the native or browser timer backend of [`timeout`].
///
/// # Panics
/// May panic if the timer cannot represent the deadline for `duration`.
pub async fn sleep(duration: std::time::Duration) {
    futures_timer::Delay::new(duration).await;
}

#[cfg(test)]
mod tests;
