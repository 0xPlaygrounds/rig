//! A transport-agnostic websocket contract.
//!
//! rig-core names no transport. [`http_client::HttpClientExt`](crate::http_client::HttpClientExt)
//! states what a request/response transport must do and `rig-reqwest` supplies
//! one; this module is the same arrangement for websockets, so a provider's
//! websocket protocol — its event envelopes, its session state machine — can
//! live beside the provider's other code instead of inside whichever crate
//! happens to own the socket library.
//!
//! Two traits, split by lifetime:
//!
//! - [`WebSocketClientExt`] is the *backend*: it performs one handshake and
//!   hands back a live connection. Hosts hold it the way they hold an
//!   `HttpClientExt`.
//! - [`WebSocketConnection`] is one open socket. It is **object-safe**, and
//!   sessions hold it erased as a [`BoxedWebSocketConnection`], so a provider
//!   session type keeps a single generic parameter instead of gaining one per
//!   transport. A session exchanges a handful of frames per turn, so the
//!   indirection is not on any hot path.
//!
//! Object safety is why [`WebSocketConnection`]'s methods return boxed futures
//! while [`WebSocketClientExt::connect`] returns an `impl Future`: only the
//! connection is erased.
//!
//! # Errors
//!
//! Both traits report [`http_client::Error`](crate::http_client::Error). That
//! is deliberate: a *rejected* websocket upgrade is not a websocket at all, it
//! is an ordinary HTTP response with a status, headers and a provider error
//! body, and [`Error::non_success_with_details`](crate::http_client::Error::non_success_with_details)
//! is already the shape that keeps all three inspectable (rig#2314). A backend
//! that flattens a rejection to a display string throws away the provider's
//! own diagnosis; do not.

use crate::http_client::{Error, NoBody, Request, Result};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use bytes::Bytes;
use std::time::Duration;

/// One websocket frame, in either direction.
///
/// The control frames are modeled rather than hidden because a provider
/// session needs them: a `Close` mid-turn is a protocol event with a reason
/// worth reporting, and `Ping`/`Pong` are frames a session skips explicitly
/// rather than by accident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    /// A UTF-8 text frame — how every JSON websocket protocol talks.
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

/// How a backend should perform the handshake.
///
/// A struct rather than a bare argument so a backend can gain knobs (proxy,
/// subprotocols, frame-size caps) without breaking every call site.
#[derive(Clone, Debug, Default)]
pub struct ConnectOptions {
    /// Abandon the handshake if it has not completed within this duration.
    /// `None` waits indefinitely.
    ///
    /// The backend enforces this and reports the elapse, since it owns the
    /// handshake; a session's own event timeout is separate and stays with the
    /// session, which is the only side that knows where a turn ends.
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

/// A websocket backend: opens connections, then gets out of the way.
///
/// The handshake is an ordinary HTTP request — the URI carries the `ws`/`wss`
/// scheme and the headers carry the provider's authentication — so a backend
/// takes the same [`Request`] type the HTTP transport does. Implementations
/// supply the websocket-specific handshake headers (`Sec-WebSocket-Key` and
/// friends) themselves; callers must not.
pub trait WebSocketClientExt: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Open a connection, or fail with the provider's own rejection preserved
    /// (see the module's error note).
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = Result<BoxedWebSocketConnection>> + WasmCompatSend;
}

/// One open websocket connection.
///
/// Object-safe by construction: sessions hold this erased (see
/// [`BoxedWebSocketConnection`]). The contract is sequential — a session drives
/// one turn at a time and never polls `send` and `recv` concurrently — so
/// `&mut self` is enough and no backend needs an internal split.
pub trait WebSocketConnection: WasmCompatSend {
    /// Write one frame.
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>>;

    /// Read the next frame; `Ok(None)` means the peer ended the stream.
    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>>;

    /// Complete a close handshake. Calling this more than once is the caller's
    /// business to avoid; backends may report an error on a closed socket.
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

/// Derive the websocket URL for `path` from an HTTP(S) base URL, upgrading the
/// scheme (`https` -> `wss`, `http` -> `ws`).
///
/// Providers expose one base URL for both transports, so this is shared rather
/// than reimplemented per provider. An unsupported scheme is an error, not a
/// silent passthrough: a URL built from an unexpected base is a failure the
/// caller should see at connect time rather than as a confusing handshake
/// rejection.
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
mod tests {
    use super::*;

    #[test]
    fn websocket_url_upgrades_the_scheme_and_appends_the_path() {
        assert_eq!(
            websocket_url("https://api.openai.com/v1", "responses").expect("https upgrades"),
            "wss://api.openai.com/v1/responses"
        );
        assert_eq!(
            websocket_url("http://127.0.0.1:8080/v1", "responses").expect("http upgrades"),
            "ws://127.0.0.1:8080/v1/responses"
        );
    }

    /// A base URL that already names the websocket scheme is left on it rather
    /// than rejected: hosts that configure `wss://` directly are not wrong.
    #[test]
    fn websocket_url_accepts_an_already_websocket_scheme() {
        assert_eq!(
            websocket_url("wss://api.openai.com/v1", "responses").expect("wss stays wss"),
            "wss://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn websocket_url_trims_a_trailing_slash() {
        assert_eq!(
            websocket_url("https://api.openai.com/v1/", "responses").expect("trailing slash"),
            "wss://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn websocket_url_rejects_an_unsupported_scheme() {
        let error = websocket_url("ftp://api.openai.com/v1", "responses")
            .expect_err("ftp is not a websocket base");
        assert!(
            error.to_string().contains("ftp"),
            "the error should name the scheme, got {error}"
        );
    }
}
