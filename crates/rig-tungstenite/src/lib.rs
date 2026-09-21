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
//! The bundled native websocket backend and session constructors for Rig.
//!
//! Sockets use the current Tokio runtime or a lazy fallback runtime. Off-runtime
//! callers communicate through channels without polling socket I/O themselves.
//!
//! ```
//! let backend = rig_tungstenite::TungsteniteClient::new();
//! ```

// The native dependency is target-gated; diagnose unsupported WASM use before
// unresolved imports obscure the required browser backend.
#[cfg(target_family = "wasm")]
compile_error!(
    "rig-tungstenite is a native websocket backend (tokio-tungstenite). On wasm, implement \
     `rig_core::ws_client::WebSocketClientExt` over `web_sys::WebSocket` and open sessions with \
     `connect_with(..)`."
);

#[cfg(not(target_family = "wasm"))]
pub use tokio_tungstenite;

#[cfg(not(target_family = "wasm"))]
mod connection;
#[cfg(not(target_family = "wasm"))]
mod runtime;
#[cfg(not(target_family = "wasm"))]
mod session;

#[cfg(not(target_family = "wasm"))]
pub use session::{DefaultWebSocketBuilder, DefaultWebSocketClient};

#[cfg(not(target_family = "wasm"))]
use connection::{DirectConnection, ForwardedConnection};
#[cfg(not(target_family = "wasm"))]
use rig_core::http_client::{Error, NoBody, Request, Result};
#[cfg(not(target_family = "wasm"))]
use rig_core::ws_client::{BoxedWebSocketConnection, ConnectOptions, WebSocketClientExt};
#[cfg(not(target_family = "wasm"))]
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

/// Bring the default-backend traits into scope.
#[cfg(not(target_family = "wasm"))]
pub mod prelude {
    pub use crate::session::{DefaultWebSocketBuilder, DefaultWebSocketClient};
    pub use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketExt;
}

#[cfg(not(target_family = "wasm"))]
/// A stateless websocket backend that takes handshake configuration from each
/// request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TungsteniteClient;

#[cfg(not(target_family = "wasm"))]
impl TungsteniteClient {
    /// The bundled backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(target_family = "wasm"))]
impl WebSocketClientExt for TungsteniteClient {
    async fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> Result<BoxedWebSocketConnection> {
        let request = client_request(request)?;

        #[cfg(not(target_family = "wasm"))]
        if !runtime::in_tokio() {
            let timeout = options.timeout;
            return runtime::run_off_runtime(async move {
                let socket = handshake(request, timeout).await?;
                ForwardedConnection::spawn(socket)
            })
            .await?;
        }

        let socket = handshake(request, options.timeout).await?;
        Ok(Box::new(DirectConnection::new(socket)))
    }
}

#[cfg(not(target_family = "wasm"))]
/// Build a tungstenite handshake request, returning an error for an invalid URI.
/// Caller headers, including authentication, override generated handshake headers.
fn client_request(request: Request<NoBody>) -> Result<tungstenite::handshake::client::Request> {
    let (parts, _) = request.into_parts();
    let mut request = parts
        .uri
        .to_string()
        .into_client_request()
        .map_err(from_tungstenite)?;
    for (name, value) in &parts.headers {
        request.headers_mut().insert(name, value.clone());
    }
    Ok(request)
}

#[cfg(not(target_family = "wasm"))]
/// A handshake that did not complete in time.
#[derive(Debug, thiserror::Error)]
#[error("timed out connecting the websocket after {0:?}")]
struct ConnectTimeout(Duration);

#[cfg(not(target_family = "wasm"))]
async fn handshake(
    request: tungstenite::handshake::client::Request,
    timeout: Option<Duration>,
) -> Result<connection::Socket> {
    let connect = async {
        tokio_tungstenite::connect_async(request)
            .await
            .map(|(socket, _)| socket)
            .map_err(from_tungstenite)
    };

    let Some(timeout) = timeout else {
        return connect.await;
    };

    match rig_core::wasm_compat::timeout(timeout, connect).await {
        Ok(result) => result,
        Err(_) => Err(Error::instance(ConnectTimeout(timeout))),
    }
}

#[cfg(not(target_family = "wasm"))]
/// Convert a tungstenite failure to a transport error, preserving the status,
/// headers, and body of a rejected upgrade.
/// Other failures become [`Error::Instance`].
fn from_tungstenite(error: tungstenite::Error) -> Error {
    let tungstenite::Error::Http(response) = error else {
        return Error::instance(error);
    };

    let (parts, body) = (*response).into_parts();
    let body = body
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .unwrap_or_default();

    Error::non_success_with_details(parts.status, parts.headers, body)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
