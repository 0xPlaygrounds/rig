//! Target-dependent thread bounds, boxed futures, and executor-independent timers.
//!
//! ```
//! use rig_core::wasm_compat::WasmBoxedFuture;
//! let future: WasmBoxedFuture<'_, u32> = Box::pin(async { 42 });
//! ```

use bytes::Bytes;
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
    "rig-core does not support threaded wasm (`+atomics`): its wasm-compat markers assume a \
     single-threaded target"
);

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Send` on native targets, a no-op marker on browser wasm.
///
/// ```compile_fail
/// use std::rc::Rc;
/// use rig_core::{serve::{Dispatch, Reply, Serve}, effect::{EffectKind, HandlerDescriptor, family}};
///
/// struct Local(Rc<u8>);
/// impl Serve for Local {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { unimplemented!() }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply { unimplemented!() }
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Send`, and every bus handler must be `Send + Sync` natively",
    label = "not `Send`",
    note = "a handler runs inside the driver's task: hold the model, tool or memory behind an `Arc` (never an `Rc`), or register a `!Send` value only on browser wasm, where this marker is a no-op"
)]
pub trait WasmCompatSend: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Send` on native targets, a no-op marker on browser wasm.
pub trait WasmCompatSend {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> WasmCompatSend for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// Streaming response bound that includes `Send` on native targets.
pub trait WasmCompatSendStream:
    Stream<Item = Result<Bytes, crate::http_client::Error>> + Send
{
    type InnerItem: Send;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Streaming response bound without `Send` on browser wasm.
pub trait WasmCompatSendStream: Stream<Item = Result<Bytes, crate::http_client::Error>> {
    type InnerItem;
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>> + Send,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>>,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// `Sync` on native targets, a no-op marker on browser wasm.
///
/// ```compile_fail
/// use std::cell::Cell;
/// use rig_core::{serve::{Dispatch, Reply, Serve}, effect::{EffectKind, HandlerDescriptor, family}};
///
/// struct Local(Cell<u8>);
/// impl Serve for Local {
///     type Family = family::Dynamic;
///     fn descriptor(&self) -> HandlerDescriptor { unimplemented!() }
///     async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply { unimplemented!() }
/// }
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not `Sync`, and every bus handler must be `Send + Sync` natively",
    label = "not `Sync`",
    note = "a handler is shared between the driver and its in-flight tasks: use `Mutex`/atomics instead of `Cell`/`RefCell`, or register a `!Sync` value only on browser wasm, where this marker is a no-op"
)]
pub trait WasmCompatSync: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// `Sync` on native targets, a no-op marker on browser wasm.
pub trait WasmCompatSync {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> WasmCompatSync for T {}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
/// Boxed future with `Send` on the same targets as [`WasmCompatSend`].
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
/// Boxed future type without `Send`, on browser wasm.
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

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
