//! A type-erased HTTP transport.
//!
//! [`HttpClientExt`] is generic at the method level (`send<T, U>`), so it is
//! not object-safe and every client that holds a transport is generic over
//! `H`. That is the right default for a library — monomorphized, zero-cost —
//! but a *host* that owns one transport for many providers (a worker pool, an
//! ECS resource, a plugin) does not want `H` leaking into every type it holds
//! and every signature that touches it. [`BoxedHttpClient`] is that host's
//! transport: one `Arc<dyn …>` that implements [`HttpClientExt`] by collapsing
//! the method generics to [`Bytes`] at the boundary and re-expanding them on
//! the way out. It is the rig-core counterpart of the erased model and tool
//! handles in rig-agent.
//!
//! The adapter is byte-transparent: method, URI, headers and body reach the
//! inner transport exactly as given, and the response body is the inner
//! transport's bytes converted with `U::from`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::{Request, Response};

use super::middleware::HttpMiddleware;
use super::{HttpClientExt, LazyBody, MultipartForm, Result, StreamingResponse};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Object-safe mirror of [`HttpClientExt`] with the generics fixed to
/// [`Bytes`]. Private: the only way to reach it is through
/// [`BoxedHttpClient`], which re-exposes the generic surface.
pub(crate) trait ErasedHttpClient: WasmCompatSend + WasmCompatSync {
    fn send_bytes(
        &self,
        req: Request<Bytes>,
    ) -> WasmBoxedFuture<'static, Result<Response<LazyBody<Bytes>>>>;

    fn send_multipart_bytes(
        &self,
        req: Request<MultipartForm>,
    ) -> WasmBoxedFuture<'static, Result<Response<LazyBody<Bytes>>>>;

    /// Borrows `self`: [`HttpClientExt::send_streaming`] does not promise a
    /// `'static` future, so neither can its erasure.
    fn send_streaming_bytes(
        &self,
        req: Request<Bytes>,
    ) -> WasmBoxedFuture<'_, Result<StreamingResponse>>;
}

impl<H> ErasedHttpClient for H
where
    H: HttpClientExt + 'static,
{
    fn send_bytes(
        &self,
        req: Request<Bytes>,
    ) -> WasmBoxedFuture<'static, Result<Response<LazyBody<Bytes>>>> {
        Box::pin(self.send::<Bytes, Bytes>(req))
    }

    fn send_multipart_bytes(
        &self,
        req: Request<MultipartForm>,
    ) -> WasmBoxedFuture<'static, Result<Response<LazyBody<Bytes>>>> {
        Box::pin(self.send_multipart::<Bytes>(req))
    }

    fn send_streaming_bytes(
        &self,
        req: Request<Bytes>,
    ) -> WasmBoxedFuture<'_, Result<StreamingResponse>> {
        Box::pin(self.send_streaming::<Bytes>(req))
    }
}

/// A type-erased, cheaply cloneable HTTP transport.
///
/// Wraps any `H: HttpClientExt + 'static` behind one `Arc<dyn …>` and
/// implements [`HttpClientExt`] itself, so a `Client<Ext, BoxedHttpClient>`
/// names no concrete transport. `Clone` is a reference-count bump: one
/// `BoxedHttpClient` can be handed to every provider client a host builds.
///
/// Use it when *holding* a transport (a host runtime, an ECS resource, a
/// registry built at startup); keep the generic `H` when *writing* a provider
/// or when a monomorphized transport is what you want. Boxing an already
/// boxed transport returns a clone of it, never a second layer.
///
/// `Debug` prints only the type name — the inner transport may carry
/// credentials in its configuration.
///
/// Not serializable: a transport is a live connection pool, not data.
///
/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<rig_core::http_client::BoxedHttpClient>();
/// ```
#[derive(Clone)]
pub struct BoxedHttpClient {
    inner: Arc<dyn ErasedHttpClient>,
    /// Transport-boundary middleware, applied in attachment order around
    /// every request this handle sends. Cloned handles share the same stack.
    middleware: Vec<Arc<dyn HttpMiddleware>>,
}

impl BoxedHttpClient {
    /// Erase `http`. If `http` is already a `BoxedHttpClient`, this is a clone
    /// (its attached middleware included).
    pub fn new<H>(http: H) -> Self
    where
        H: HttpClientExt + 'static,
    {
        if let Some(already) = (&http as &dyn Any).downcast_ref::<BoxedHttpClient>() {
            return already.clone();
        }
        Self {
            inner: Arc::new(http),
            middleware: Vec::new(),
        }
    }

    /// Attach a transport-boundary [`HttpMiddleware`] to this handle.
    ///
    /// Middlewares run in attachment order — see the
    /// [`middleware`](super::middleware) module docs for the exact per-phase
    /// ordering and error semantics. Attaching returns a new handle; existing
    /// clones keep their previous stack. The underlying transport is shared,
    /// so [`ptr_eq`](Self::ptr_eq) still reports the two as the same
    /// transport.
    pub fn with_middleware<M>(mut self, middleware: M) -> Self
    where
        M: HttpMiddleware + 'static,
    {
        self.middleware.push(Arc::new(middleware));
        self
    }

    /// Whether two handles share the same underlying transport.
    ///
    /// Compares the transport only: handles that differ in attached
    /// middleware but wrap the same transport still compare equal.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Run the request-side middleware phases: all header hooks in order,
    /// then all body hooks in order (each seeing the final headers).
    async fn apply_request_middleware(
        &self,
        parts: &mut http::request::Parts,
        body: Bytes,
    ) -> Result<Bytes> {
        for mw in &self.middleware {
            mw.before_request_headers(&parts.method, &parts.uri, &mut parts.headers)
                .await?;
        }
        let mut body = body;
        for mw in &self.middleware {
            body = mw
                .before_request_body(&parts.method, &parts.uri, &parts.headers, body)
                .await?;
        }
        Ok(body)
    }

    /// Run only the header hooks (the multipart path, which has no single
    /// serialized body to hand to the body hooks).
    async fn apply_header_middleware(&self, parts: &mut http::request::Parts) -> Result<()> {
        for mw in &self.middleware {
            mw.before_request_headers(&parts.method, &parts.uri, &mut parts.headers)
                .await?;
        }
        Ok(())
    }

    /// Run the response hooks in attachment order.
    async fn apply_response_middleware(
        &self,
        method: &http::Method,
        uri: &http::Uri,
        status: http::StatusCode,
        headers: &http::HeaderMap,
    ) -> Result<()> {
        for mw in &self.middleware {
            mw.after_response(method, uri, status, headers).await?;
        }
        Ok(())
    }
}

impl fmt::Debug for BoxedHttpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BoxedHttpClient")
    }
}

impl HttpClientExt for BoxedHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let this = self.clone();
        let (mut parts, body) = req.map(Into::into).into_parts();
        async move {
            let body = this.apply_request_middleware(&mut parts, body).await?;
            let (method, uri) = (parts.method.clone(), parts.uri.clone());
            let response = this
                .inner
                .send_bytes(Request::from_parts(parts, body))
                .await?;
            this.apply_response_middleware(&method, &uri, response.status(), response.headers())
                .await?;
            Ok(convert_body::<U>(response))
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let this = self.clone();
        let (mut parts, body) = req.into_parts();
        async move {
            this.apply_header_middleware(&mut parts).await?;
            let (method, uri) = (parts.method.clone(), parts.uri.clone());
            let response = this
                .inner
                .send_multipart_bytes(Request::from_parts(parts, body))
                .await?;
            this.apply_response_middleware(&method, &uri, response.status(), response.headers())
                .await?;
            Ok(convert_body::<U>(response))
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let (mut parts, body) = req.map(Into::into).into_parts();
        async move {
            let body = self.apply_request_middleware(&mut parts, body).await?;
            let (method, uri) = (parts.method.clone(), parts.uri.clone());
            let response = self
                .inner
                .send_streaming_bytes(Request::from_parts(parts, body))
                .await?;
            // The response hooks run before any of the body stream is
            // consumed — the point of `after_response` for streaming calls.
            self.apply_response_middleware(&method, &uri, response.status(), response.headers())
                .await?;
            Ok(response)
        }
    }
}

fn convert_body<U>(response: Response<LazyBody<Bytes>>) -> Response<LazyBody<U>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    response.map(|body| -> LazyBody<U> { Box::pin(async move { body.await.map(U::from) }) })
}

#[cfg(test)]
mod tests;
