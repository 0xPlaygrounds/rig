#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! The WASI HTTP transport for Rig. `WstdClient` implements Rig's
//! `HttpClientExt` over the `wasi:http` client of
//! [`wstd`](https://docs.rs/wstd), so provider clients run inside a WASI 0.2
//! component, for example under `wasmtime run -S http`.
//!
//! wstd's reactor drives every request, response body and stream, so await
//! them inside the wstd runtime (`#[wstd::main]` or `wstd::runtime::block_on`).
//! On targets other than WASI the crate is empty.
//!
//! ```no_run
//! # #[cfg(target_os = "wasi")]
//! # {
//! use rig_http::http_client::DynHttpClient;
//!
//! // `OpenAIConfig::from_env()?.connect(http)` sends through it.
//! let http = DynHttpClient::new(rig_wstd::WstdClient::default());
//! # let _ = http;
//! # }
//! ```

#[cfg(target_os = "wasi")]
mod client;
#[cfg(target_os = "wasi")]
pub use client::WstdClient;
