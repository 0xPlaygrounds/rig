//! Transport-independent websocket handshakes, connections, and frames.
//! Backends preserve rejected upgrades as HTTP errors with status, headers, and body.
//!
//! ```
//! use rig_core::ws_client::websocket_url;
//!
//! assert_eq!(websocket_url("https://example.com/v1", "responses")?,
//!            "wss://example.com/v1/responses");
//! # Ok::<(), rig_core::http_client::Error>(())
//! ```

use crate::http_client::{Error, NoBody, Request, Result};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use bytes::Bytes;
use std::time::Duration;

/// One websocket data or control frame, in either direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    /// A UTF-8 text frame.
    Text(String),
    /// A binary frame.
    Binary(Bytes),
    /// A ping, with its application payload.
    Ping(Bytes),
    /// A pong, with its application payload.
    Pong(Bytes),
    /// A close frame, with the peer's status and reason when it sent one.
    Close(Option<CloseFrame>),
}

/// The status code and reason carried by a websocket close frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CloseFrame {
    /// The RFC 6455 close code.
    pub code: u16,
    /// The peer's reason, empty when it sent none.
    pub reason: String,
}

/// Options enforced by the backend during the handshake.
#[derive(Clone, Debug, Default)]
pub struct ConnectOptions {
    /// Backend-enforced handshake timeout, separate from session event timeouts.
    /// `None` imposes no handshake deadline.
    pub timeout: Option<Duration>,
}

impl ConnectOptions {
    /// Options with no connect timeout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the handshake timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Opens websocket connections from requests with WS(S) URIs and authentication
/// headers. Backends supply websocket handshake headers such as
/// `Sec-WebSocket-Key`; callers must not supply those headers.
pub trait WebSocketClientExt: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Opens a connection, preserving rejected upgrades with their HTTP status,
    /// headers, and response body.
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = Result<BoxedWebSocketConnection>> + WasmCompatSend;
}

/// One open websocket connection, usable as a trait object.
/// Calls are sequential: sessions must not poll send and receive concurrently.
/// WASM-compatible bounds preserve the containing session's thread-safety contract.
pub trait WebSocketConnection: WasmCompatSend + WasmCompatSync {
    /// Write one frame.
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>>;

    /// Read the next frame; `Ok(None)` means the peer ended the stream.
    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>>;

    /// Completes a close handshake. Callers must avoid repeated closes;
    /// backends may return an error for an already closed socket.
    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>>;
}

/// A type-erased [`WebSocketConnection`].
pub type BoxedWebSocketConnection = Box<dyn WebSocketConnection>;

impl WebSocketConnection for BoxedWebSocketConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).send(frame)
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        (**self).recv()
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        (**self).close(frame)
    }
}

/// A base URL that cannot be turned into a websocket URL.
#[derive(Debug, thiserror::Error)]
#[error("invalid websocket base URL: {0}")]
pub struct InvalidWebSocketUrl(String);

/// Appends `path` to a base URL, converting HTTP(S) to WS(S) and retaining
/// existing WS(S) schemes, query, and fragment. Trims boundary slashes from the
/// appended path. Returns an error for invalid URLs or unsupported schemes.
pub fn websocket_url(base_url: &str, path: &str) -> Result<String> {
    fn invalid(message: impl Into<String>) -> Error {
        Error::instance(InvalidWebSocketUrl(message.into()))
    }

    let mut url =
        url::Url::parse(base_url).map_err(|error| invalid(format!("{base_url}: {error}")))?;

    let scheme = match url.scheme() {
        "https" | "wss" => "wss",
        "http" | "ws" => "ws",
        other => {
            return Err(invalid(format!(
                "unsupported base URL scheme for websocket mode: {other}"
            )));
        }
    };
    url.set_scheme(scheme)
        .map_err(|()| invalid(format!("failed to convert {base_url} to a websocket URL")))?;

    let path = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        path.trim_matches('/')
    );
    url.set_path(&path);
    Ok(url.to_string())
}

#[cfg(test)]
mod tests;
