use std::pin::Pin;

#[cfg(not(target_arch = "wasm32"))]
pub trait WasmCompatSend: Send {}
#[cfg(target_arch = "wasm32")]
pub trait WasmCompatSend {}

#[cfg(not(target_arch = "wasm32"))]
impl<T> WasmCompatSend for T where T: Send {}
#[cfg(target_arch = "wasm32")]
impl<T> WasmCompatSend for T {}

#[cfg(not(target_arch = "wasm32"))]
pub trait WasmCompatSync: Sync {}
#[cfg(target_arch = "wasm32")]
pub trait WasmCompatSync {}

#[cfg(not(target_arch = "wasm32"))]
impl<T> WasmCompatSync for T where T: Sync {}
#[cfg(target_arch = "wasm32")]
impl<T> WasmCompatSync for T {}

#[cfg(not(target_family = "wasm"))]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
pub type WasmBoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[macro_export]
macro_rules! if_wasm {
    ($($tokens:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        $($tokens)*

    };
}

#[macro_export]
macro_rules! if_not_wasm {
    ($($tokens:tt)*) => {
        #[cfg(not(target_arch = "wasm32"))]
        $($tokens)*

    };
}
