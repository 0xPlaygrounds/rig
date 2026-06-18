use bytes::Bytes;
use std::pin::Pin;

use futures::Stream;

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
/// `Send` on native targets and a no-op marker on wasm32 with the `wasm` feature.
pub trait WasmCompatSend: Send {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
/// `Send` on native targets and a no-op marker on wasm32 with the `wasm` feature.
pub trait WasmCompatSend {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSend for T {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
/// Streaming response bound that includes `Send` on native targets.
pub trait WasmCompatSendStream:
    Stream<Item = Result<Bytes, crate::http_client::Error>> + Send
{
    type InnerItem: Send;
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
/// Streaming response bound without `Send` on wasm32 with the `wasm` feature.
pub trait WasmCompatSendStream: Stream<Item = Result<Bytes, crate::http_client::Error>> {
    type InnerItem;
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>> + Send,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>>,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
/// `Sync` on native targets and a no-op marker on wasm32 with the `wasm` feature.
pub trait WasmCompatSync: Sync {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
/// `Sync` on native targets and a no-op marker on wasm32 with the `wasm` feature.
pub trait WasmCompatSync {}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
impl<T> WasmCompatSync for T {}

#[cfg(not(target_family = "wasm"))]
/// Boxed future type that includes `Send` on non-wasm targets.
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
/// Boxed future type without `Send` on wasm targets.
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

/// Await `future`, returning `Err(`[`Elapsed`]`)` if it does not complete within
/// `duration`.
///
/// A cross-platform (native + wasm) replacement for `tokio::time::timeout`: rig's
/// `tokio` dependency is built without the `time` feature, and `tokio::time` does
/// not function on wasm. This is built on [`futures_timer::Delay`], which rig
/// already uses for SSE retry backoff and which supports wasm via its
/// `wasm-bindgen` feature (enabled by rig's `wasm` feature).
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
        #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
        $($tokens)*

    };
}

#[macro_export]
macro_rules! if_not_wasm {
    ($($tokens:tt)*) => {
        #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
        $($tokens)*

    };
}
