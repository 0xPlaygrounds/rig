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
pub use tokio_tungstenite;

#[cfg(not(target_family = "wasm"))]
mod connection;

#[cfg(not(target_family = "wasm"))]
use connection::Connection;
#[cfg(not(target_family = "wasm"))]
use rig_http::http_client::{Error, NoBody, Request, Result};
#[cfg(not(target_family = "wasm"))]
use rig_http::ws_client::{BoxedWebSocketConnection, ConnectOptions, WebSocketClientExt};
#[cfg(not(target_family = "wasm"))]
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

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
        let socket =
            rig_http::runtime::bind(Box::pin(handshake(request, options.timeout)))?.await?;
        Ok(Box::new(Connection(socket)))
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

    match rig_http::wasm_compat::timeout(timeout, connect).await {
        Ok(result) => result,
        Err(_) => Err(Error::transient(ConnectTimeout(timeout))),
    }
}

#[cfg(not(target_family = "wasm"))]
/// Convert a tungstenite failure to a transport error, preserving the status,
/// headers, and body of a rejected upgrade. A connection that dropped or
/// could not be made is [`Error::Transient`]; a URL, request, message or
/// protocol error that the same send meets again is [`Error::Permanent`].
fn from_tungstenite(error: tungstenite::Error) -> Error {
    use tungstenite::error::{ProtocolError, UrlError};
    let transient = match &error {
        tungstenite::Error::Http(_) => false,
        tungstenite::Error::ConnectionClosed
        | tungstenite::Error::WriteBufferFull(_)
        | tungstenite::Error::Url(UrlError::UnableToConnect(_))
        | tungstenite::Error::Protocol(
            ProtocolError::HandshakeIncomplete | ProtocolError::ResetWithoutClosingHandshake,
        ) => true,
        // rustls reports a certificate it rejects as invalid data.
        tungstenite::Error::Io(error) => error.kind() != std::io::ErrorKind::InvalidData,
        _ => false,
    };
    let tungstenite::Error::Http(response) = error else {
        return if transient {
            Error::transient(error)
        } else {
            Error::permanent(error)
        };
    };

    let (parts, body) = (*response).into_parts();
    let body = body
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .unwrap_or_default();

    Error::non_success_with_details(parts.status, parts.headers, body)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
