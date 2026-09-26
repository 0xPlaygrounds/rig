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
//! HTTP and websocket contracts without a transport implementation.
//!
//! Use `rig-reqwest` for HTTP and `rig-tungstenite` for native websockets,
//! or implement the contracts with a host's own networking stack.
//!
//! ```
//! use rig_http::http_client::framing::SseFramer;
//!
//! let mut frames = SseFramer::new();
//! assert_eq!(frames.push(b"data: hello\n\n").next().map(|event| event.data), Some("hello".into()));
//! ```

pub mod http_client;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod wasm_compat;
pub use wasm_compat::{BoxFuture, BoxStream, MaybeSend, MaybeSync};

#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod ws_client;
