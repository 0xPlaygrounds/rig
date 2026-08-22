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
pub struct BoxedHttpClient(Arc<dyn ErasedHttpClient>);

impl BoxedHttpClient {
    /// Erase `http`. If `http` is already a `BoxedHttpClient`, this is a clone.
    pub fn new<H>(http: H) -> Self
    where
        H: HttpClientExt + 'static,
    {
        if let Some(already) = (&http as &dyn Any).downcast_ref::<BoxedHttpClient>() {
            return already.clone();
        }
        Self(Arc::new(http))
    }

    /// Whether two handles share the same underlying transport.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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
        let fut = self.0.send_bytes(req.map(Into::into));
        async move { fut.await.map(convert_body::<U>) }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let fut = self.0.send_multipart_bytes(req);
        async move { fut.await.map(convert_body::<U>) }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.0.send_streaming_bytes(req.map(Into::into))
    }
}

fn convert_body<U>(response: Response<LazyBody<Bytes>>) -> Response<LazyBody<U>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    response.map(|body| -> LazyBody<U> { Box::pin(async move { body.await.map(U::from) }) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::RecordingHttpClient;
    use futures::executor::block_on;

    fn request(body: &'static str) -> Request<&'static str> {
        Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/echo?x=1")
            .header("x-probe", "yes")
            .body(body)
            .expect("valid request")
    }

    #[test]
    fn unary_request_passes_through_unchanged_and_body_converts() {
        let inner = RecordingHttpClient::new(r#"{"ok":true}"#);
        let boxed = BoxedHttpClient::new(inner.clone());

        let response = block_on(boxed.send::<_, Vec<u8>>(request("hello"))).expect("send");
        let body = block_on(response.into_body()).expect("body");
        assert_eq!(body, br#"{"ok":true}"#);

        let captured = inner.requests();
        assert_eq!(captured.len(), 1);
        let req = &captured[0];
        assert_eq!(req.uri, "https://example.test/v1/echo?x=1");
        assert_eq!(req.body, Bytes::from_static(b"hello"));
        assert_eq!(
            req.headers.get("x-probe").and_then(|v| v.to_str().ok()),
            Some("yes")
        );
    }

    #[test]
    fn boxing_a_boxed_client_is_a_clone_not_a_second_layer() {
        let boxed = BoxedHttpClient::new(RecordingHttpClient::new(""));
        let again = BoxedHttpClient::new(boxed.clone());
        assert!(boxed.ptr_eq(&again));
        assert!(boxed.ptr_eq(&boxed.clone()));
    }

    #[test]
    fn debug_never_prints_the_inner_transport() {
        let boxed = BoxedHttpClient::new(RecordingHttpClient::new("secret-bearing config"));
        assert_eq!(format!("{boxed:?}"), "BoxedHttpClient");
    }

    #[test]
    fn transport_errors_surface_unchanged() {
        let inner = RecordingHttpClient::with_error(http::StatusCode::TOO_MANY_REQUESTS, "slow");
        let boxed = BoxedHttpClient::new(inner);
        let err = match block_on(boxed.send::<_, Bytes>(request(""))) {
            Ok(_) => panic!("expected a transport error"),
            Err(err) => err,
        };
        assert_eq!(
            err.non_success_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
    }
}
