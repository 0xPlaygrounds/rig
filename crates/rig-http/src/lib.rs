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
//! Rig's transport contracts. A transport implements
//! [`HttpClientExt`](http_client::HttpClientExt) (and, for websocket sessions,
//! [`WebSocketClientExt`](ws_client::WebSocketClientExt)), and every Rig
//! provider sends through it. This crate holds no transport of its own:
//! `rig-reqwest` and `rig-tungstenite` are the bundled ones. `rig-core`
//! re-exports these modules at `rig_core::http_client`, `rig_core::ws_client`
//! and `rig_core::wasm_compat`.
//!
//! ```
//! use rig_http::http_client::{DynHttpClient, HttpClientExt};
//!
//! fn erase(client: impl HttpClientExt + Clone + 'static) -> DynHttpClient {
//!     DynHttpClient::new(client)
//! }
//! ```

pub mod http_client;
#[cfg(all(feature = "tokio", not(target_family = "wasm")))]
#[cfg_attr(docsrs, doc(cfg(feature = "tokio")))]
pub mod runtime;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod wasm_compat;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod ws_client;
