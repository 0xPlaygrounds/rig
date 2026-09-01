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
//! The bundled `tokio-tungstenite` websocket backend for Rig.
//!
//! rig-core is transport-agnostic: a provider's websocket session drives a
//! [`WebSocketConnection`](rig_core::ws_client::WebSocketConnection) and rig-core
//! names no backend, exactly as it names no HTTP transport. This crate supplies
//! one, and the conveniences that let a caller who is happy with it never say so:
//!
//! - [`TungsteniteClient`], a [`WebSocketClientExt`] over `tokio-tungstenite`.
//! - [`DefaultWebSocketClient`] / [`DefaultWebSocketBuilder`], the
//!   default-backend traits — `client.responses_websocket("gpt-5.4")` with no
//!   backend named — mirroring `rig-reqwest`'s `DefaultTransportClient`.
//!
//! # Running without a tokio runtime
//!
//! `tokio-tungstenite` needs a tokio reactor. Inside a tokio runtime this
//! backend drives the socket directly; outside one it moves the socket onto a
//! lazily started fallback runtime and talks to it over `futures` channels, so
//! the caller polls only runtime-agnostic futures.

// A caller who reaches for this crate on wasm gets one sentence instead of a
// page of unresolved imports: the dependency is target-gated, so nothing below
// would resolve. `rig_core::ws_client` is the seam a browser backend plugs
// into, and rig-core's own websocket support builds for wasm today.
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
/// The bundled websocket backend.
///
/// Cheap to clone and to construct: a handshake takes its configuration from
/// the request, so the client itself holds nothing yet. It is a unit struct
/// rather than a wrapper because `tokio-tungstenite` has no client object to
/// wrap — `connect_async` is a free function.
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
/// Lower rig's transport-agnostic handshake request onto tungstenite's.
///
/// `into_client_request` supplies the websocket handshake headers
/// (`Sec-WebSocket-Key` and friends) from the URI; the caller's headers — the
/// provider's authentication — are copied on top.
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
/// Map a tungstenite failure onto rig's transport error, **preserving a
/// rejected upgrade's response**.
///
/// This is the one piece of the websocket path that must not lose information.
/// A provider that refuses the upgrade never opens a websocket at all: it
/// answers with an ordinary HTTP response carrying a status, its own error
/// envelope, and the headers a caller needs to back off (`Retry-After`, and the
/// provider's request id). `tungstenite` hands all of it back on
/// [`tungstenite::Error::Http`], with the body filled in from the read tail, so
/// flattening the error to its display string — `"HTTP error: 401
/// Unauthorized"` — is the difference between a caller that can diagnose a bad
/// key and one that cannot (rig#2314, rig#2315, rig#2210).
///
/// [`Error::non_success_with_details`] is the shape that keeps all three, and
/// the provider layer reads its own request-id header back off it. Failures
/// that never reached the provider — TLS, DNS, a protocol violation — have no
/// response to preserve and become [`Error::Instance`].
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
