use super::*;
use futures::StreamExt;
use http::StatusCode;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Default)]
struct ScriptClock(AtomicU64);

impl rig_core::observe::Clock for ScriptClock {
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.0.load(Ordering::SeqCst))
    }
}

#[derive(Clone)]
struct ChunkedUnary {
    clock: Arc<ScriptClock>,
    status: StatusCode,
    timing: BodyTiming,
    fail_body: bool,
}

impl HttpClientExt for ChunkedUnary {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let observer = req
            .extensions()
            .get::<rig_core::observe::ResponseBodyObserver>()
            .cloned();
        let clock = self.clock.clone();
        let status = self.status;
        let timing = self.timing;
        let fail_body = self.fail_body;
        async move {
            let payload = if status.is_success() {
                r#"{"candidates":[{"content":{"parts":[{"text":"pong"}],"role":"model"},"finishReason":"STOP"}]}"#
            } else {
                r#"{"error":{"code":429,"message":"rate limited"}}"#
            };
            let chunks = futures::stream::iter([
                (12, Bytes::new()),
                (23, Bytes::copy_from_slice(&payload.as_bytes()[..1])),
                (60, Bytes::copy_from_slice(&payload.as_bytes()[1..])),
            ])
            .map(move |(at, bytes)| {
                clock.0.store(at, Ordering::SeqCst);
                if fail_body && at == 60 {
                    Err(std::io::Error::other("scripted body failure"))
                } else {
                    Ok(bytes)
                }
            });
            let response = http::Response::builder()
                .status(status)
                .header("retry-after", "20")
                .body(reqwest::Body::wrap_stream(chunks))
                .unwrap();
            into_response::<U>(reqwest::Response::from(response), timing, observer).await
        }
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(Error::NoHeaders))
    }

    fn send_streaming<T>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        std::future::ready(Err(Error::NoHeaders))
    }
}

#[tokio::test]
async fn unary_first_byte_precedes_buffering_for_success_and_failure() {
    use rig_core::{
        client::CompletionClient,
        completion::CompletionModel,
        observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject},
    };
    for status in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
        for timing in [BodyTiming::Lazy, BodyTiming::Eager] {
            for fail_body in [false, true] {
                let mut baseline = None;
                for observed in [false, true] {
                    let clock = Arc::new(ScriptClock(AtomicU64::new(10)));
                    let client = rig_core::providers::gemini::Client::builder()
                        .api_key("test-key")
                        .http_client(ChunkedUnary {
                            clock: clock.clone(),
                            status,
                            timing,
                            fail_body,
                        })
                        .build()
                        .unwrap();
                    let model = client.completion_model("gemini-test");
                    let log = Arc::new(ObservationLog::default().with_clock(clock));
                    let mut request = model.completion_request("hello").build();
                    if observed {
                        request.observation =
                            Some(AdapterContext::new(log.clone(), Subject::default(), "call"));
                    }
                    let result = model.completion(request).await;
                    assert_eq!(result.is_ok(), status.is_success() && !fail_body);
                    let result = result
                        .map(|r| serde_json::to_value(r.choice).unwrap())
                        .map_err(|e| rig_core::error::ErrorReport::from(&e));
                    if observed {
                        assert_eq!(baseline.as_ref(), Some(&result));
                        let trace = log.trace();
                        let timings: Vec<_> = trace
                            .observations
                            .iter()
                            .filter_map(|o| match &o.action {
                                Action::Adapter { observation }
                                    if matches!(
                                        observation.event,
                                        AdapterEvent::Finished { .. }
                                    ) =>
                                {
                                    observation.analysis.as_ref()?.timing.as_ref()
                                }
                                _ => None,
                            })
                            .collect();
                        assert_eq!(timings.len(), 1);
                        assert_eq!(
                            timings[0].request_duration,
                            Some(std::time::Duration::from_millis(50))
                        );
                        assert_eq!(
                            timings[0].time_to_first_byte,
                            Some(std::time::Duration::from_millis(13))
                        );
                    } else {
                        assert!(log.trace().observations.is_empty());
                        baseline = Some(result);
                    }
                }
            }
        }
    }
}

#[test]
fn streamed_http_errors_keep_first_byte_on_native_and_fallback_runtimes() {
    use rig_core::{
        client::CompletionClient,
        completion::CompletionModel,
        observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject},
    };
    use std::io::{Read, Write};
    use std::time::Duration;
    for native in [false, true] {
        for status in [429, 503] {
            let mut baseline = None;
            for observed in [false, true] {
                let clock = Arc::new(ScriptClock(AtomicU64::new(10)));
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                let address = listener.local_addr().unwrap();
                let server_clock = clock.clone();
                let server = std::thread::spawn(move || {
                    let (mut socket, _) = listener.accept().unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut bytes = Vec::new();
                    let mut byte = [0];
                    while !bytes.ends_with(b"\r\n\r\n") {
                        socket.read_exact(&mut byte).unwrap();
                        bytes.push(byte[0]);
                    }
                    let headers = String::from_utf8(bytes).unwrap();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|value| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    let mut body = vec![0; length];
                    socket.read_exact(&mut body).unwrap();
                    let response = format!(
                        "{{\"error\":{{\"code\":{status},\"message\":\"provider unavailable\"}}}}"
                    );
                    write!(socket, "HTTP/1.1 {status} Error\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n", response.len()).unwrap();
                    socket.flush().unwrap();
                    server_clock.0.store(23, Ordering::SeqCst);
                    socket.write_all(response.as_bytes()).unwrap();
                    socket.flush().unwrap();
                });
                let log = Arc::new(ObservationLog::default().with_clock(clock));
                let execute = async {
                    let client = rig_core::providers::gemini::Client::builder()
                        .api_key("test-key")
                        .base_url(format!("http://{address}"))
                        .http_client(ReqwestClient::new(
                            reqwest::Client::builder()
                                .no_proxy()
                                .retry(reqwest::retry::never())
                                .build()
                                .unwrap(),
                        ))
                        .build()
                        .unwrap();
                    let model = client.completion_model("gemini-test");
                    let mut request = model.completion_request("hello").build();
                    if observed {
                        request.observation =
                            Some(AdapterContext::new(log.clone(), Subject::default(), "call"));
                    }
                    match model.stream(request).await {
                        Ok(mut stream) => match stream.next().await {
                            Some(Err(error)) => error,
                            _ => panic!("HTTP rejection must yield an error"),
                        },
                        Err(error) => rig_core::error::ErrorReport::from(&error),
                    }
                };
                let error = if native {
                    tokio::runtime::Runtime::new().unwrap().block_on(execute)
                } else {
                    futures::executor::block_on(execute)
                };
                server.join().unwrap();
                if observed {
                    assert_eq!(baseline.as_ref(), Some(&error));
                    let trace = log.trace();
                    assert_eq!(trace.observations.iter().filter(|fact| matches!(&fact.action, Action::Adapter { observation } if matches!(observation.event, AdapterEvent::Started { .. }))).count(), 1);
                    let closures: Vec<_> = trace
                        .observations
                        .iter()
                        .filter_map(|fact| match &fact.action {
                            Action::Adapter { observation }
                                if matches!(observation.event, AdapterEvent::Finished { .. }) =>
                            {
                                Some(observation)
                            }
                            _ => None,
                        })
                        .collect();
                    assert_eq!(closures.len(), 1);
                    assert_eq!(
                        closures[0]
                            .analysis
                            .as_ref()
                            .unwrap()
                            .timing
                            .as_ref()
                            .unwrap()
                            .time_to_first_byte,
                        Some(Duration::from_millis(13))
                    );
                } else {
                    assert!(log.trace().observations.is_empty());
                    baseline = Some(error);
                }
            }
        }
    }
}

/// rig#2210: the bundled transport's own error constructor is where the
/// headers are captured, so drive it with a real `reqwest::Response`.
#[tokio::test]
async fn non_success_status_error_preserves_response_headers() {
    let response = http::Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("retry-after", "20")
        .header("x-ratelimit-remaining", "0")
        .body(r#"{"error":{"message":"rate limited"}}"#)
        .expect("valid response");

    let error = non_success_status_error(reqwest::Response::from(response)).await;

    assert!(matches!(
        &error,
        Error::InvalidStatusCodeWithDetails { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS
    ));
    let headers = error
        .non_success_headers()
        .expect("headers captured at error construction");
    assert_eq!(
        headers.get("retry-after").and_then(|v| v.to_str().ok()),
        Some("20")
    );
    assert_eq!(
        headers
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0")
    );
}
