use bytes::Bytes;
use std::pin::Pin;

use futures::Stream;

#[cfg(not(feature = "wasm"))]
pub trait WasmCompatSend: Send {}
#[cfg(feature = "wasm")]
pub trait WasmCompatSend {}

#[cfg(not(feature = "wasm"))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(feature = "wasm")]
impl<T> WasmCompatSend for T {}

#[cfg(not(feature = "wasm"))]
pub trait WasmCompatSendStream:
    Stream<Item = Result<Bytes, crate::http_client::Error>> + Send
{
    type InnerItem: Send;
}

#[cfg(feature = "wasm")]
pub trait WasmCompatSendStream: Stream<Item = Result<Bytes, crate::http_client::Error>> {
    type InnerItem;
}

#[cfg(not(feature = "wasm"))]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>> + Send,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(feature = "wasm")]
impl<T> WasmCompatSendStream for T
where
    T: Stream<Item = Result<Bytes, crate::http_client::Error>>,
{
    type InnerItem = Result<Bytes, crate::http_client::Error>;
}

#[cfg(not(feature = "wasm"))]
pub trait WasmCompatSync: Sync {}
#[cfg(feature = "wasm")]
pub trait WasmCompatSync {}

#[cfg(not(feature = "wasm"))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(feature = "wasm")]
impl<T> WasmCompatSync for T {}

#[cfg(not(feature = "wasm"))]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(feature = "wasm")]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[macro_export]
macro_rules! if_wasm {
    ($($tokens:tt)*) => {
        #[cfg(feature = "wasm")]
        $($tokens)*

    };
}

#[macro_export]
macro_rules! if_not_wasm {
    ($($tokens:tt)*) => {
        #[cfg(not(feature = "wasm"))]
        $($tokens)*

    };
}
