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
        let Err(err) = block_on(boxed.send::<_, Bytes>(request("hello"))) else {
            panic!("must abort")
        };
        assert_eq!(err.non_success_status(), Some(StatusCode::BAD_REQUEST));
        assert!(inner.requests().is_empty(), "nothing reached the transport");
    }

    #[test]
    fn response_hook_failure_surfaces_as_the_request_error() {
        let inner = RecordingHttpClient::new("ok");
        let boxed = BoxedHttpClient::new(inner.clone()).with_middleware(FailAt("response"));
        let Err(err) = block_on(boxed.send::<_, Bytes>(request("hello"))) else {
            panic!("must fail")
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
        let boxed = BoxedHttpClient::new(inner).with_middleware(CaptureContentType(seen.clone()));
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
    let Err(err) = block_on(boxed.send::<_, Bytes>(request(""))) else {
        panic!("expected a transport error")
    };
    assert_eq!(
        err.non_success_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
}
