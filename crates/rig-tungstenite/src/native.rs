//! The backend itself, compiled on native targets only.

pub use tokio_tungstenite;

mod connection;

use connection::Connection;
use rig_http::http_client::{Error, NoBody, Request, Result};
use rig_http::ws_client::{BoxedWebSocketConnection, ConnectOptions, WebSocketClientExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

/// A stateless websocket backend that takes handshake configuration from each
/// request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TungsteniteClient;

impl TungsteniteClient {
    /// The bundled backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

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

/// A handshake that did not complete in time.
#[derive(Debug, thiserror::Error)]
#[error("timed out connecting the websocket after {0:?}")]
struct ConnectTimeout(Duration);

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

/// Convert a tungstenite failure to a transport error, preserving the status,
/// headers, and body of a rejected upgrade. A connection that dropped or
/// could not be made, and any TLS failure, is [`Error::Transient`]; a URL,
/// request, message or protocol error that the same send meets again is
/// [`Error::Permanent`].
fn from_tungstenite(error: tungstenite::Error) -> Error {
    use tungstenite::error::{ProtocolError, UrlError};
    match error {
        tungstenite::Error::Http(response) => {
            let (parts, body) = (*response).into_parts();
            let body = body
                .map(|body| String::from_utf8_lossy(&body).into_owned())
                .unwrap_or_default();
            Error::non_success_with_details(parts.status, parts.headers, body)
        }
        tungstenite::Error::ConnectionClosed
        | tungstenite::Error::WriteBufferFull(_)
        | tungstenite::Error::Url(UrlError::UnableToConnect(_))
        | tungstenite::Error::Protocol(
            ProtocolError::HandshakeIncomplete | ProtocolError::ResetWithoutClosingHandshake,
        ) => Error::transient(error),
        // rustls reports a certificate it rejects as invalid data.
        tungstenite::Error::Io(ref io) if io.kind() != std::io::ErrorKind::InvalidData => {
            Error::transient(error)
        }
        // native-tls reports a handshake the peer reset as it reports a
        // certificate it refused.
        tungstenite::Error::Tls(_) => Error::transient(error),
        error => Error::permanent(error),
    }
}

#[cfg(test)]
mod tests;
