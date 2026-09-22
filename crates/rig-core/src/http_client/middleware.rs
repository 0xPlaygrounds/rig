//! Transport-boundary middleware for [`BoxedHttpClient`](super::BoxedHttpClient).
//! Attach hooks with [`with_middleware`](super::BoxedHttpClient::with_middleware).
//!
//! ```
//! use rig_core::http_client::middleware::HttpMiddleware;
//!
//! struct PassThrough;
//! impl HttpMiddleware for PassThrough {}
//! ```

use bytes::Bytes;
use http::{HeaderMap, Method, StatusCode, Uri};

use super::Result;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Transport-boundary hooks applied by
/// [`BoxedHttpClient`](super::BoxedHttpClient) around every request.
///
/// Methods default to no-ops. All header hooks run in attachment order, then
/// all body hooks run in that order with the final headers and preceding body
/// replacements. Response hooks run in attachment order after transport returns,
/// before streaming-body consumption. Multipart requests skip body hooks.
///
/// Any hook error fails the request; request-side errors abort before sending.
/// Response hooks cannot modify responses. Hooks run on the request future and
/// must not block.
pub trait HttpMiddleware: WasmCompatSend + WasmCompatSync {
    /// Mutate the outgoing request headers in place.
    ///
    /// Runs before [`before_request_body`](Self::before_request_body). An
    /// error aborts the request before it is sent.
    fn before_request_headers<'a>(
        &'a self,
        _method: &'a Method,
        _uri: &'a Uri,
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
        _method: &'a Method,
        _uri: &'a Uri,
        _headers: &'a HeaderMap,
        body: Bytes,
    ) -> WasmBoxedFuture<'a, Result<Bytes>> {
        Box::pin(async move { Ok(body) })
    }

    /// Observe the response status and headers as soon as they arrive.
    ///
    /// For streaming responses this runs before any of the body stream is
    /// consumed. Observe-only: the response cannot be modified, but returning
    /// an error fails the request with that error.
    fn after_response<'a>(
        &'a self,
        _method: &'a Method,
        _uri: &'a Uri,
        _status: StatusCode,
        _headers: &'a HeaderMap,
    ) -> WasmBoxedFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}
