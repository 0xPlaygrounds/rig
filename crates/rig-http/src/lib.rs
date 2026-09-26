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
//! use rig_http::http_client::{HeaderMap, bearer_auth_header};
//!
//! let mut headers = HeaderMap::new();
//! bearer_auth_header(&mut headers, "example-token")?;
//! # Ok::<(), rig_http::http_client::Error>(())
//! ```

pub mod http_client;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod wasm_compat;
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod ws_client;
