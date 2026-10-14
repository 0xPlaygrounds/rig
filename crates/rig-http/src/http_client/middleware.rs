//! Transport-boundary middleware for [`DynHttpClient`](super::DynHttpClient).
//! Attach hooks with [`with_middleware`](super::DynHttpClient::with_middleware).
//!
//! ```
//! use rig_http::http_client::middleware::HttpMiddleware;
//!
//! struct PassThrough;
//! impl HttpMiddleware for PassThrough {}
//! ```

use bytes::Bytes;
use http::{Extensions, HeaderMap, Method, StatusCode, Uri};

use super::Result;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// The request a hook runs for: its method, URI and extensions, which carry
/// what the caller attached to it, such as the model call a request serves
/// (rig-core's `serve::Serving`). Built by
/// [`DynHttpClient`](super::DynHttpClient); a field added later breaks no
/// hook.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Exchange<'a> {
    /// The request's method.
    pub method: &'a Method,
    /// The request's URI.
    pub uri: &'a Uri,
    /// The request's extensions, as the transport receives them.
    pub extensions: &'a Extensions,
}

/// Transport-boundary hooks applied by
/// [`DynHttpClient`](super::DynHttpClient) around every request.
///
/// Methods default to no-ops. All header hooks run in attachment order, then
/// all body hooks run in that order with the final headers and preceding body
/// replacements. Response hooks run in attachment order on every reply,
/// before streaming-body consumption: a response, or a non-success reply a
/// transport turned into an error that keeps its status and headers, as
/// `rig-reqwest` does. Multipart requests skip body hooks.
///
/// Any hook error fails the request; request-side errors abort before sending.
/// An error reply fails with its own error whatever its hooks return.
/// Response hooks cannot modify responses. Hooks run on the request future and
/// must not block.
pub trait HttpMiddleware: WasmCompatSend + WasmCompatSync {
    /// Mutate the outgoing request headers in place.
    ///
    /// Runs before [`before_request_body`](Self::before_request_body). An
    /// error aborts the request before it is sent.
    fn before_request_headers<'a>(
        &'a self,
        _exchange: Exchange<'a>,
        _headers: &'a mut HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Observe the serialized request body, returning the body to send.
    ///
    /// Return `body` unchanged to pass through, or a replacement to rewrite
    /// the payload. `headers` reflects every middleware's header mutations.
    /// Not invoked for multipart requests. An error aborts the request before
    /// it is sent.
    fn before_request_body<'a>(
        &'a self,
        _exchange: Exchange<'a>,
        _headers: &'a HeaderMap,
        body: Bytes,
    ) -> WasmBoxedFuture<'a, Result<Bytes>> {
        Box::pin(async move { Ok(body) })
    }

    /// Observe the response status and headers as soon as they arrive,
    /// including a non-success reply's (such as a 429's rate-limit headers)
    /// on a transport that reports it as an error. Not invoked when the
    /// request failed with no reply.
    ///
    /// For streaming responses this runs before any of the body stream is
    /// consumed. Observe-only: the response cannot be modified, but returning
    /// an error fails the request with that error.
    fn after_response<'a>(
        &'a self,
        _exchange: Exchange<'a>,
        _status: StatusCode,
        _headers: &'a HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}
