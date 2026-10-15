//! Target-dependent thread bounds, boxed futures, and executor-independent timers.
//!
//! ```
//! use rig_http::wasm_compat::WasmBoxedFuture;
//! let future: WasmBoxedFuture<'_, u32> = Box::pin(async { 42 });
//! ```

use bytes::Bytes;
use std::pin::Pin;

use futures::Stream;

// Wasm markers assume single-threaded execution; atomics would invalidate
// that assumption. Relaxed bounds do not make non-Send handlers thread-safe.
#[cfg(all(target_family = "wasm", target_feature = "atomics"))]
compile_error!(
    "rig-http does not support threaded wasm (`+atomics`): its wasm-compat markers assume a \
     single-threaded target"
);

#[cfg(not(target_family = "wasm"))]
/// `Send` on native targets, a no-op marker on wasm.
///
/// ```compile_fail
/// use std::rc::Rc;
/// use rig_http::wasm_compat::WasmCompatSend;
///
/// fn shared<T: WasmCompatSend>(_: T) {}
/// shared(Rc::new(0u8));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Send`, and Rig needs `Send` natively",
    label = "not `Send`",
    note = "handlers and transports run inside a driver's task: hold shared state behind an `Arc` (never an `Rc`), or use a `!Send` value only on wasm, where this marker is a no-op"
)]
pub trait WasmCompatSend: Send {}
#[cfg(target_family = "wasm")]
/// `Send` on native targets, a no-op marker on wasm.
pub trait WasmCompatSend {}

#[cfg(not(target_family = "wasm"))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(target_family = "wasm")]
impl<T> WasmCompatSend for T {}

#[cfg(not(target_family = "wasm"))]
/// Streaming response bound that includes `Send` on native targets.
pub trait WasmCompatSendStream:
    Stream<Item = Result<Bytes, crate::http_client::Error>> + Send
{
    type InnerItem: Send;
}

#[cfg(target_family = "wasm")]
/// Streaming response bound without `Send` on wasm.
pub trait WasmCompatSendStream: Stream<Item = Result<Bytes, crate::http_client::Error>> {
    type InnerItem;
}

#[cfg(not(target_family = "wasm"))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>> + Send,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(target_family = "wasm")]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>>,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(not(target_family = "wasm"))]
/// `Sync` on native targets, a no-op marker on wasm.
///
/// ```compile_fail
/// use std::cell::Cell;
/// use rig_http::wasm_compat::WasmCompatSync;
///
/// fn shared<T: WasmCompatSync>(_: T) {}
/// shared(Cell::new(0u8));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Sync`, and Rig needs `Sync` natively",
    label = "not `Sync`",
    note = "handlers and transports are shared between a driver and its in-flight tasks: use `Mutex`/atomics instead of `Cell`/`RefCell`, or use a `!Sync` value only on wasm, where this marker is a no-op"
)]
pub trait WasmCompatSync: Sync {}
#[cfg(target_family = "wasm")]
/// `Sync` on native targets, a no-op marker on wasm.
pub trait WasmCompatSync {}

#[cfg(not(target_family = "wasm"))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(target_family = "wasm")]
impl<T> WasmCompatSync for T {}

#[cfg(not(target_family = "wasm"))]
/// Boxed future with `Send` on the same targets as [`WasmCompatSend`].
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
/// Boxed future type without `Send`, on wasm.
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(not(target_family = "wasm"))]
/// Boxed stream with `Send` on the same targets as [`WasmCompatSend`].
pub type WasmBoxedStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
/// Boxed stream type without `Send`, on wasm.
pub type WasmBoxedStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;

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
/// its drop cleanup. Uses a native timer thread, browser `setTimeout`, or on
/// WASI a `wasi:clocks` timer, without a feature flag. On WASI the timer needs
/// the `wstd` runtime (`wstd::main` or `wstd::runtime::block_on`).
///
/// # Panics
/// May panic if the timer cannot represent the deadline for `duration`, or on
/// WASI when polled outside the `wstd` runtime.
pub async fn timeout<F>(duration: std::time::Duration, future: F) -> Result<F::Output, Elapsed>
where
    F: Future,
{
    use futures::future::{Either, select};

    let delay = sleep(duration);
    futures::pin_mut!(future);
    futures::pin_mut!(delay);
    match select(future, delay).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(((), _)) => Err(Elapsed),
    }
}

/// Sleep for `duration` using the timer backend of [`timeout`].
///
/// # Panics
/// As [`timeout`].
pub async fn sleep(duration: std::time::Duration) {
    #[cfg(not(target_os = "wasi"))]
    futures_timer::Delay::new(duration).await;
    #[cfg(target_os = "wasi")]
    wstd::task::sleep(duration.into()).await;
}

#[cfg(test)]
mod tests;
