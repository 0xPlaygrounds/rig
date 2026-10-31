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
    let erased = DynHttpClient::new(inner.clone());

    let response = block_on(erased.send::<_, Vec<u8>>(request("hello"))).expect("send");
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
fn erasing_an_erased_client_is_a_clone_not_a_second_layer() {
    let erased = DynHttpClient::new(RecordingHttpClient::new(""));
    let again = DynHttpClient::new(erased.clone());
    assert!(erased.ptr_eq(&again));
    assert!(erased.ptr_eq(&erased.clone()));
}

#[test]
fn debug_never_prints_the_inner_transport() {
    let erased = DynHttpClient::new(RecordingHttpClient::new("secret-bearing config"));
    assert_eq!(format!("{erased:?}"), "DynHttpClient");
}

mod middleware {
    use super::*;
    use crate::http_client::middleware::{Exchange, HttpMiddleware};
    use crate::test_utils::MockStreamingClient;
    use crate::wasm_compat::WasmBoxedFuture;
    use http::{HeaderMap, HeaderValue, StatusCode};
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
            _exchange: Exchange<'a>,
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
            _exchange: Exchange<'a>,
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
            _exchange: Exchange<'a>,
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
        let erased = DynHttpClient::new(inner.clone())
            .with_middleware(Tagger {
                tag: "a",
                log: log.clone(),
            })
            .with_middleware(Tagger {
                tag: "b",
                log: log.clone(),
            });

        block_on(erased.send::<_, Bytes>(request("body-"))).expect("send");

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
            _exchange: Exchange<'a>,
            _headers: &'a mut HeaderMap,
        ) -> WasmBoxedFuture<'a, Result<()>> {
            let fail = self.0 == "headers";
            Box::pin(async move {
                if fail {
                    return Err(crate::http_client::Error::non_success_with_details(
                        StatusCode::BAD_REQUEST,
                        http::HeaderMap::new(),
                        "rejected headers".into(),
                    ));
                }
                Ok(())
            })
        }

        fn after_response<'a>(
            &'a self,
            _exchange: Exchange<'a>,
            _status: StatusCode,
            _headers: &'a HeaderMap,
        ) -> WasmBoxedFuture<'a, Result<()>> {
            let fail = self.0 == "response";
            Box::pin(async move {
                if fail {
                    return Err(crate::http_client::Error::non_success_with_details(
                        StatusCode::TOO_MANY_REQUESTS,
                        http::HeaderMap::new(),
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
        let erased = DynHttpClient::new(inner.clone()).with_middleware(FailAt("headers"));
        let Err(err) = block_on(erased.send::<_, Bytes>(request("hello"))) else {
            panic!("must abort")
        };
        assert_eq!(err.non_success_status(), Some(StatusCode::BAD_REQUEST));
        assert!(inner.requests().is_empty(), "nothing reached the transport");
    }

    #[test]
    fn response_hook_failure_surfaces_as_the_request_error() {
        let inner = RecordingHttpClient::new("ok");
        let erased = DynHttpClient::new(inner.clone()).with_middleware(FailAt("response"));
        let Err(err) = block_on(erased.send::<_, Bytes>(request("hello"))) else {
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
                _exchange: Exchange<'a>,
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
        let erased = DynHttpClient::new(inner).with_middleware(CaptureContentType(seen.clone()));
        // The hook has run by the time `send_streaming` resolves — before
        // any of the body stream is polled.
        let response = block_on(erased.send_streaming(request(""))).expect("stream");
        assert_eq!(
            seen.lock().expect("seen").as_deref(),
            Some("200:text/event-stream")
        );
        drop(response);
    }

    /// Records each reply's status and `retry-after` header.
    struct SeeReplies(Arc<Mutex<Vec<String>>>);

    impl HttpMiddleware for SeeReplies {
        fn after_response<'a>(
            &'a self,
            _exchange: Exchange<'a>,
            status: StatusCode,
            headers: &'a HeaderMap,
        ) -> WasmBoxedFuture<'a, Result<()>> {
            Box::pin(async move {
                let wait = headers
                    .get(http::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("-");
                self.0
                    .lock()
                    .expect("seen")
                    .push(format!("{}:{wait}", status.as_u16()));
                Ok(())
            })
        }
    }

    #[test]
    fn an_error_reply_reaches_the_response_hooks_and_still_fails_the_request() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, HeaderValue::from_static("7"));
        let inner =
            RecordingHttpClient::with_error_headers(StatusCode::TOO_MANY_REQUESTS, "slow", headers);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let erased = DynHttpClient::new(inner)
            .with_middleware(SeeReplies(seen.clone()))
            .with_middleware(FailAt("response"));
        let Err(err) = block_on(erased.send::<_, Bytes>(request("hello"))) else {
            panic!("an error reply fails the request")
        };
        assert_eq!(seen.lock().expect("seen").as_slice(), ["429:7"]);
        assert_eq!(
            err.non_success_body(),
            Some("slow"),
            "the reply's own error"
        );
    }

    /// A transport error that kept no reply, such as a reset connection, has
    /// nothing to show a response hook, so none runs and the request fails
    /// with that error.
    #[test]
    fn a_transport_error_without_a_reply_runs_no_response_hook() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let erased = DynHttpClient::new(RecordingHttpClient::new(""))
            .with_middleware(SeeReplies(seen.clone()));
        let (parts, body) = request("hello").into_parts();
        let failed = block_on(erased.observe(parts, body, |_| async {
            Err::<Response<Bytes>, _>(crate::http_client::Error::transient(std::io::Error::other(
                "connection reset",
            )))
        }));
        let err = failed.err().map(|err| err.to_string());
        assert!(
            err.is_some_and(|err| err.contains("connection reset")),
            "the transport's error"
        );
        assert!(seen.lock().expect("seen").is_empty());
    }

    /// What a caller attached to a request, such as the call it serves.
    #[derive(Clone, Debug, PartialEq)]
    struct Serves(u64);

    /// Records the [`Serves`] each phase's exchange carries.
    struct SeeServes(Arc<Mutex<Vec<String>>>);

    impl SeeServes {
        fn note(&self, phase: &str, exchange: Exchange<'_>) {
            let serves = exchange.extensions.get::<Serves>().map(|serves| serves.0);
            self.0
                .lock()
                .expect("seen")
                .push(format!("{phase}:{} {serves:?}", exchange.method));
        }
    }

    impl HttpMiddleware for SeeServes {
        fn before_request_headers<'a>(
            &'a self,
            exchange: Exchange<'a>,
            _headers: &'a mut HeaderMap,
        ) -> WasmBoxedFuture<'a, Result<()>> {
            self.note("headers", exchange);
            Box::pin(async { Ok(()) })
        }

        fn before_request_body<'a>(
            &'a self,
            exchange: Exchange<'a>,
            _headers: &'a HeaderMap,
            body: Bytes,
        ) -> WasmBoxedFuture<'a, Result<Bytes>> {
            self.note("body", exchange);
            Box::pin(async move { Ok(body) })
        }

        fn after_response<'a>(
            &'a self,
            exchange: Exchange<'a>,
            _status: StatusCode,
            _headers: &'a HeaderMap,
        ) -> WasmBoxedFuture<'a, Result<()>> {
            self.note("response", exchange);
            Box::pin(async { Ok(()) })
        }
    }

    #[test]
    fn every_hook_reads_what_the_request_carries() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let inner = RecordingHttpClient::with_error(StatusCode::TOO_MANY_REQUESTS, "slow");
        let erased = DynHttpClient::new(inner).with_middleware(SeeServes(seen.clone()));
        let mut marked = request("hello");
        marked.extensions_mut().insert(Serves(7));
        let _ = block_on(erased.send::<_, Bytes>(marked));
        assert_eq!(
            seen.lock().expect("seen").as_slice(),
            [
                "headers:POST Some(7)",
                "body:POST Some(7)",
                "response:POST Some(7)"
            ]
        );
    }

    #[test]
    fn erasing_preserves_middleware_and_ptr_eq_ignores_it() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plain = DynHttpClient::new(RecordingHttpClient::new("ok"));
        let layered = plain.clone().with_middleware(Tagger {
            tag: "a",
            log: log.clone(),
        });
        // Same transport, so ptr_eq holds despite differing middleware.
        assert!(plain.ptr_eq(&layered));
        // Re-erasing clones the handle, middleware included.
        let reerased = DynHttpClient::new(layered);
        block_on(reerased.send::<_, Bytes>(request("x"))).expect("send");
        assert!(!log.lock().expect("log").is_empty());
    }
}

#[test]
fn transport_errors_surface_unchanged() {
    let inner = RecordingHttpClient::with_error(http::StatusCode::TOO_MANY_REQUESTS, "slow");
    let erased = DynHttpClient::new(inner);
    let Err(err) = block_on(erased.send::<_, Bytes>(request(""))) else {
        panic!("expected a transport error")
    };
    assert_eq!(
        err.non_success_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
}
