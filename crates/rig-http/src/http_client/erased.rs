//! Type-erased HTTP transport with optional middleware.
//! Without middleware, requests pass through unchanged after body conversion to
//! bytes; lazy response bodies convert through `U::from`.
//!
//! ```
//! use rig_http::http_client::{DynHttpClient, HttpClientExt};
//!
//! fn erase(client: impl HttpClientExt + 'static) -> DynHttpClient {
//!     DynHttpClient::new(client)
//! }
//! ```

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use http::{Request, Response};

use super::middleware::{Exchange, HttpMiddleware};
use super::{HttpClientExt, LazyBody, MultipartForm, Result, StreamingResponse};
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// Object-safe mirror of [`HttpClientExt`] with the generics fixed to
/// [`Bytes`]. Private: the only way to reach it is through
/// [`DynHttpClient`], which re-exposes the generic surface.
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
/// Clones share the transport and middleware instances but copy the middleware
/// list. Erasing an already erased transport clones it without another layer.
/// `Debug` prints only the type name to avoid exposing transport credentials.
/// The transport is not serializable.
///
/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<rig_http::http_client::DynHttpClient>();
/// ```
#[derive(Clone)]
pub struct DynHttpClient {
    inner: Arc<dyn ErasedHttpClient>,
    /// Transport-boundary middleware, applied in attachment order around
    /// every request this handle sends. Cloned handles share the same stack.
    middleware: Vec<Arc<dyn HttpMiddleware>>,
}

impl DynHttpClient {
    /// Erase `http`. If `http` is already a `DynHttpClient`, this is a clone
    /// (its attached middleware included).
    pub fn new<H>(http: H) -> Self
    where
        H: HttpClientExt + 'static,
    {
        if let Some(already) = (&http as &dyn Any).downcast_ref::<DynHttpClient>() {
            return already.clone();
        }
        Self {
            inner: Arc::new(http),
            middleware: Vec::new(),
        }
    }

    /// Attach a transport-boundary [`HttpMiddleware`] to this handle.
    ///
    /// Hooks run in attachment order with the phases defined by [`HttpMiddleware`].
    /// Existing clones keep their previous stack. The underlying transport stays
    /// shared, so [`Self::ptr_eq`] is unaffected.
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
        self.apply_header_middleware(parts).await?;
        let exchange = Exchange {
            method: &parts.method,
            uri: &parts.uri,
            extensions: &parts.extensions,
        };
        let mut body = body;
        for mw in &self.middleware {
            body = mw
                .before_request_body(exchange, &parts.headers, body)
                .await?;
        }
        Ok(body)
    }

    /// Run only the header hooks (the multipart path, which has no single
    /// serialized body to hand to the body hooks).
    async fn apply_header_middleware(&self, parts: &mut http::request::Parts) -> Result<()> {
        for mw in &self.middleware {
            let exchange = Exchange {
                method: &parts.method,
                uri: &parts.uri,
                extensions: &parts.extensions,
            };
            mw.before_request_headers(exchange, &mut parts.headers)
                .await?;
        }
        Ok(())
    }

    /// Send `parts` and `body` through `send`, then run the response hooks
    /// in attachment order on its reply: a response, or an error that kept
    /// a non-success reply's status and headers. A hook's error fails a
    /// response with that error; an error reply stops at a failing hook and
    /// still fails with its own error.
    async fn observe<T, B, F>(
        &self,
        parts: http::request::Parts,
        body: T,
        send: impl FnOnce(Request<T>) -> F,
    ) -> Result<Response<B>>
    where
        F: Future<Output = Result<Response<B>>>,
    {
        let (method, uri, extensions) = (
            parts.method.clone(),
            parts.uri.clone(),
            parts.extensions.clone(),
        );
        let exchange = Exchange {
            method: &method,
            uri: &uri,
            extensions: &extensions,
        };
        let error = match send(Request::from_parts(parts, body)).await {
            Ok(response) => {
                for mw in &self.middleware {
                    mw.after_response(exchange, response.status(), response.headers())
                        .await?;
                }
                return Ok(response);
            }
            Err(error) => error,
        };
        if let (Some(status), Some(headers)) =
            (error.non_success_status(), error.non_success_headers())
        {
            for mw in &self.middleware {
                if mw.after_response(exchange, status, headers).await.is_err() {
                    break;
                }
            }
        }
        Err(error)
    }
}

impl fmt::Debug for DynHttpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DynHttpClient")
    }
}

impl HttpClientExt for DynHttpClient {
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
            let response = this
                .observe(parts, body, |request| this.inner.send_bytes(request))
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
            let response = this
                .observe(parts, body, |request| {
                    this.inner.send_multipart_bytes(request)
                })
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
            self.observe(parts, body, |request| {
                self.inner.send_streaming_bytes(request)
            })
            .await
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
