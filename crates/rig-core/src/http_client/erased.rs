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

    mod middleware {
        use super::*;
        use crate::http_client::middleware::HttpMiddleware;
        use crate::test_utils::MockStreamingClient;
        use crate::wasm_compat::WasmBoxedFuture;
        use http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
        use std::sync::{Arc, Mutex};

        /// Appends its tag to a header, to the body, and to a shared log —
        /// making phase ordering across a stack observable.
        struct Tagger {
            tag: &'static str,
            log: Arc<Mutex<Vec<String>>>,
        }

        impl Tagger {
            fn log(&self, phase: &str) {
                self.log
                    .lock()
                    .expect("log")
                    .push(format!("{}:{}", self.tag, phase));
            }
        }

        impl HttpMiddleware for Tagger {
            fn before_request_headers<'a>(
                &'a self,
                _method: &'a Method,
                _uri: &'a Uri,
                headers: &'a mut HeaderMap,
            ) -> WasmBoxedFuture<'a, Result<()>> {
                Box::pin(async move {
                    self.log("headers");
                    headers.append("x-tag", HeaderValue::from_static(self.tag));
                    Ok(())
                })
            }

            fn before_request_body<'a>(
                &'a self,
                _method: &'a Method,
                _uri: &'a Uri,
                headers: &'a HeaderMap,
                body: Bytes,
            ) -> WasmBoxedFuture<'a, Result<Bytes>> {
                Box::pin(async move {
                    self.log("body");
                    // Every body hook sees the fully mutated headers: its own
                    // tag was already appended by the headers phase.
                    assert!(
                        headers
                            .get_all("x-tag")
                            .iter()
                            .any(|v| v.as_bytes() == self.tag.as_bytes()),
                        "body hooks run after all header hooks"
                    );
                    let mut bytes = body.to_vec();
                    bytes.extend_from_slice(self.tag.as_bytes());
                    Ok(Bytes::from(bytes))
                })
            }

            fn after_response<'a>(
                &'a self,
                _method: &'a Method,
                _uri: &'a Uri,
                status: StatusCode,
                _headers: &'a HeaderMap,
            ) -> WasmBoxedFuture<'a, Result<()>> {
                Box::pin(async move {
                    self.log(&format!("response:{}", status.as_u16()));
                    Ok(())
                })
            }
        }

        #[test]
        fn phases_run_in_attachment_order_and_mutations_chain() {
            let inner = RecordingHttpClient::new("ok");
            let log = Arc::new(Mutex::new(Vec::new()));
            let boxed = BoxedHttpClient::new(inner.clone())
                .with_middleware(Tagger {
                    tag: "a",
                    log: log.clone(),
                })
                .with_middleware(Tagger {
                    tag: "b",
                    log: log.clone(),
                });

            block_on(boxed.send::<_, Bytes>(request("body-"))).expect("send");

            let captured = inner.requests();
            assert_eq!(captured.len(), 1);
            // Header mutations chain in attachment order.
            let tags: Vec<_> = captured[0]
                .headers
                .get_all("x-tag")
                .iter()
                .map(|v| v.to_str().expect("ascii"))
                .collect();
            assert_eq!(tags, ["a", "b"]);
            // Body replacements chain in attachment order.
            assert_eq!(captured[0].body, Bytes::from_static(b"body-ab"));
            // All header hooks run before any body hook; response hooks last.
            assert_eq!(
                log.lock().expect("log").as_slice(),
                [
                    "a:headers",
                    "b:headers",
                    "a:body",
                    "b:body",
                    "a:response:200",
                    "b:response:200"
                ]
            );
        }

        /// A middleware that fails the given phase.
        struct FailAt(&'static str);

        impl HttpMiddleware for FailAt {
            fn before_request_headers<'a>(
                &'a self,
                _method: &'a Method,
                _uri: &'a Uri,
                _headers: &'a mut HeaderMap,
            ) -> WasmBoxedFuture<'a, Result<()>> {
                let fail = self.0 == "headers";
                Box::pin(async move {
                    if fail {
                        return Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                            StatusCode::BAD_REQUEST,
                            "rejected headers".into(),
                        ));
                    }
                    Ok(())
                })
            }

            fn after_response<'a>(
                &'a self,
                _method: &'a Method,
                _uri: &'a Uri,
                _status: StatusCode,
                _headers: &'a HeaderMap,
            ) -> WasmBoxedFuture<'a, Result<()>> {
                let fail = self.0 == "response";
                Box::pin(async move {
                    if fail {
                        return Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                            StatusCode::TOO_MANY_REQUESTS,
                            "rejected response".into(),
                        ));
                    }
                    Ok(())
                })
            }
        }

        #[test]
        fn request_side_failure_aborts_before_sending() {
            let inner = RecordingHttpClient::new("ok");
            let boxed = BoxedHttpClient::new(inner.clone()).with_middleware(FailAt("headers"));
            let err = match block_on(boxed.send::<_, Bytes>(request("hello"))) {
                Ok(_) => panic!("must abort"),
                Err(err) => err,
            };
            assert_eq!(err.non_success_status(), Some(StatusCode::BAD_REQUEST));
            assert!(inner.requests().is_empty(), "nothing reached the transport");
        }

        #[test]
        fn response_hook_failure_surfaces_as_the_request_error() {
            let inner = RecordingHttpClient::new("ok");
            let boxed = BoxedHttpClient::new(inner.clone()).with_middleware(FailAt("response"));
            let err = match block_on(boxed.send::<_, Bytes>(request("hello"))) {
                Ok(_) => panic!("must fail"),
                Err(err) => err,
            };
            assert_eq!(
                err.non_success_status(),
                Some(StatusCode::TOO_MANY_REQUESTS)
            );
            // The request itself was sent; only acceptance was refused.
            assert_eq!(inner.requests().len(), 1);
        }

        #[test]
        fn streaming_response_hook_sees_status_and_headers_before_consumption() {
            let seen = Arc::new(Mutex::new(None));

            struct CaptureContentType(Arc<Mutex<Option<String>>>);
            impl HttpMiddleware for CaptureContentType {
                fn after_response<'a>(
                    &'a self,
                    _method: &'a Method,
                    _uri: &'a Uri,
                    status: StatusCode,
                    headers: &'a HeaderMap,
                ) -> WasmBoxedFuture<'a, Result<()>> {
                    Box::pin(async move {
                        let content_type = headers
                            .get(http::header::CONTENT_TYPE)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_owned);
                        *self.0.lock().expect("seen") = Some(format!(
                            "{}:{}",
                            status.as_u16(),
                            content_type.unwrap_or_default()
                        ));
                        Ok(())
                    })
                }
            }

            let inner = MockStreamingClient {
                sse_bytes: Bytes::from_static(b"data: hi\n\n"),
            };
            let boxed =
                BoxedHttpClient::new(inner).with_middleware(CaptureContentType(seen.clone()));
            // The hook has run by the time `send_streaming` resolves — before
            // any of the body stream is polled.
            let response = block_on(boxed.send_streaming(request(""))).expect("stream");
            assert_eq!(
                seen.lock().expect("seen").as_deref(),
                Some("200:text/event-stream")
            );
            drop(response);
        }

        #[test]
        fn boxing_preserves_middleware_and_ptr_eq_ignores_it() {
            let log = Arc::new(Mutex::new(Vec::new()));
            let plain = BoxedHttpClient::new(RecordingHttpClient::new("ok"));
            let layered = plain.clone().with_middleware(Tagger {
                tag: "a",
                log: log.clone(),
            });
            // Same transport, so ptr_eq holds despite differing middleware.
            assert!(plain.ptr_eq(&layered));
            // Re-boxing clones the handle, middleware included.
            let reboxed = BoxedHttpClient::new(layered);
            block_on(reboxed.send::<_, Bytes>(request("x"))).expect("send");
            assert!(!log.lock().expect("log").is_empty());
        }
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
