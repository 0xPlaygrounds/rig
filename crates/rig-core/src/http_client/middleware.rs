//! Transport-boundary middleware for [`BoxedHttpClient`](super::BoxedHttpClient).
//!
//! Agent-level hooks (`AgentHook` in rig-agent) stop at the semantic layer:
//! they patch a request's *meaning* (preamble, temperature, history) but never
//! see the serialized provider payload, the HTTP headers, or the raw HTTP
//! response. [`HttpMiddleware`] is the seam below that: it runs inside the
//! erased transport, so one implementation observes and shapes every provider's
//! wire traffic — unary, streaming, and multipart — without touching provider
//! code.
//!
//! The three methods mirror the three moments a transport-level extension
//! cares about:
//!
//! - [`before_request_headers`](HttpMiddleware::before_request_headers)
//!   mutates the outgoing [`HeaderMap`] in place (per-request beta or feature
//!   headers, auth decoration).
//! - [`before_request_body`](HttpMiddleware::before_request_body) sees the
//!   serialized body and may replace it (logging/replay capture, payload
//!   patching that no semantic knob covers).
//! - [`after_response`](HttpMiddleware::after_response) sees the response
//!   status and headers as soon as they arrive — for a streaming call this is
//!   **before stream consumption**, so rate-limit headers and request ids are
//!   readable without buffering the body.
//!
//! Middlewares attach with
//! [`BoxedHttpClient::with_middleware`](super::BoxedHttpClient::with_middleware)
//! and compose in attachment order:
//!
//! - all `before_request_headers` run first, in order, each seeing the
//!   previous one's mutations;
//! - then all `before_request_body` run in order, each seeing the previous
//!   one's replacement body (and the final headers);
//! - after the transport returns, all `after_response` run in order.
//!
//! `after_response` is observe-only for the response itself, but any method
//! may *fail* the request by returning an error, which surfaces through the
//! normal [`Error`](super::Error) path — a middleware can reject a response it
//! refuses to accept, but cannot silently swallow or alter one.
//!
//! Multipart requests run the header and response methods; the body method is
//! **not** invoked for them (the form is encoded transport-side and carries no
//! single serialized payload to replace).
//!
//! Middleware failures on the request side abort before any bytes are sent.
//! A middleware must not block; it runs on the request's own future.

use bytes::Bytes;
use http::{HeaderMap, Method, StatusCode, Uri};

use super::Result;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Transport-boundary hooks applied by
/// [`BoxedHttpClient`](super::BoxedHttpClient) around every request.
///
/// All methods are default no-ops; implement only the moments you need. See
/// the [module docs](self) for ordering, composition, and error semantics.
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
