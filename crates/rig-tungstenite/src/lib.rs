#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! The bundled native websocket backend for Rig: a
//! [`rig_http::ws_client::WebSocketClientExt`] implementation over tokio-tungstenite. rig-core's `tungstenite` feature
//! opens Responses WebSocket sessions over it with no backend named.
//!
//! The caller owns and polls every socket, in the current Tokio runtime's
//! context or the lazy fallback's ([`rig_http::runtime`]), from any executor.
//!
//! ```
//! let backend = rig_tungstenite::TungsteniteClient::new();
//! ```

// The native dependency is target-gated; diagnose unsupported WASM use before
// unresolved imports obscure the required browser backend.
#[cfg(target_family = "wasm")]
compile_error!(
    "rig-tungstenite is a native websocket backend (tokio-tungstenite). On wasm, implement \
     `rig_http::ws_client::WebSocketClientExt` over `web_sys::WebSocket` and open sessions with \
     `connect_with(..)`."
);

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::*;
