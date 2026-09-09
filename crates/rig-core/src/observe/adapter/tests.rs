use super::*;
use crate::observe::ObservationLog;

#[derive(Default)]
struct ManualClock(std::sync::atomic::AtomicU64);

impl crate::observe::Clock for ManualClock {
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl ManualClock {
    fn set(&self, millis: u64) {
        self.0.store(millis, std::sync::atomic::Ordering::SeqCst);
    }
}

#[test]
fn timing_uses_body_bytes_and_closure_once_and_is_not_semantic() {
    let clock = Arc::new(ManualClock::default());
    let mut baseline = None;
    for enabled in [false, true] {
        let log = Arc::new(if enabled {
            ObservationLog::default().with_clock(clock.clone())
        } else {
            ObservationLog::default()
        });
        let context = AdapterContext::new(log.clone(), Subject::default(), "call");
        let mut request = http::Request::new(());
        context.attach(&mut request, "/completion");
        let slot = AdapterSlot::default();
        clock.set(10);
        slot.start(&request);
        clock.set(12);
        slot.bytes(b"");
        clock.set(15);
        slot.response(http::StatusCode::OK);
        clock.set(23);
        slot.bytes(b"d"); // A body byte before even a complete SSE field exists.
        clock.set(40);
        slot.bytes(b"ata: {}\n\n");
        clock.set(60);
        slot.finish(AdapterEnding::Terminal);
        clock.set(100);
        slot.finish(AdapterEnding::Dropped);
        drop(slot);
        let trace = log.trace();
        let closures: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| match &o.action {
                Action::Adapter { observation }
                    if matches!(observation.event, AdapterEvent::Finished { .. }) =>
                {
                    Some(observation)
                }
                _ => None,
            })
            .collect();
        assert_eq!(closures.len(), 1);
        if enabled {
            let timing = closures[0]
                .analysis
                .as_ref()
                .unwrap()
                .timing
                .as_ref()
                .unwrap();
            assert_eq!(
                timing.request_duration,
                Some(std::time::Duration::from_millis(50))
            );
            assert_eq!(
                timing.time_to_first_byte,
                Some(std::time::Duration::from_millis(13))
            );
            assert_eq!(
                crate::observe::compare(baseline.as_ref().unwrap(), &trace),
                crate::observe::Comparison::Equal
            );
        } else {
            assert!(closures[0].analysis.is_none());
            baseline = Some(trace);
        }
    }
}

#[test]
fn dropped_attempts_are_timed_without_inventing_unseen_bytes() {
    let clock = Arc::new(ManualClock::default());
    let log = Arc::new(ObservationLog::default().with_clock(clock.clone()));
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    clock.set(10);
    let attempt = context.begin(&http::Method::POST, "/completion").unwrap();
    clock.set(30);
    drop(attempt);
    let trace = log.trace();
    let Action::Adapter { observation } = &trace.observations[1].action else {
        panic!()
    };
    assert_eq!(
        observation.event,
        AdapterEvent::Finished {
            ending: AdapterEnding::Dropped
        }
    );
    let timing = observation
        .analysis
        .as_ref()
        .unwrap()
        .timing
        .as_ref()
        .unwrap();
    assert_eq!(
        timing.request_duration,
        Some(std::time::Duration::from_millis(20))
    );
    assert_eq!(timing.time_to_first_byte, None);
}

#[test]
fn transport_observer_cannot_sample_after_drop_or_repair_an_invalid_first_sample() {
    let clock = Arc::new(ManualClock::default());
    let log = Arc::new(ObservationLog::default().with_clock(clock.clone()));
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    for first_at in [None, Some(5)] {
        clock.set(10);
        let attempt = context.begin(&http::Method::POST, "/completion").unwrap();
        let observer = attempt.body_observer().unwrap();
        if let Some(at) = first_at {
            clock.set(at);
            observer.observe(b"first");
        }
        clock.set(30);
        drop(attempt);
        clock.set(40);
        observer.observe(b"late");
        assert_eq!(observer.close(), None);
    }
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 4);
    for fact in trace.observations.iter().skip(1).step_by(2) {
        let Action::Adapter { observation } = &fact.action else {
            panic!()
        };
        let timing = observation
            .analysis
            .as_ref()
            .unwrap()
            .timing
            .as_ref()
            .unwrap();
        assert_eq!(
            timing.request_duration,
            Some(std::time::Duration::from_millis(20))
        );
        assert_eq!(timing.time_to_first_byte, None);
    }
    let clockless = AdapterContext::new(
        Arc::new(ObservationLog::default()),
        Subject::default(),
        "clockless",
    );
    assert!(
        clockless
            .begin(&http::Method::POST, "/completion")
            .unwrap()
            .body_observer()
            .is_none()
    );
}

#[test]
fn native_http_errors_preserve_distinct_boundaries_before_report_erasure() {
    use crate::{completion::CompletionError, http_client::Error};
    for (native, boundary) in [
        (Error::NoHeaders, AdapterErrorBoundary::Request),
        (
            Error::InvalidContentType(http::HeaderValue::from_static("text/plain")),
            AdapterErrorBoundary::Decode,
        ),
        (Error::StreamEnded, AdapterErrorBoundary::Transport),
        (
            Error::instance(std::io::Error::other("opaque client failure")),
            AdapterErrorBoundary::Unknown,
        ),
    ] {
        let error = CompletionError::HttpError(native);
        let report = crate::error::ErrorReport::from(&error);
        assert_eq!(report.kind, crate::error::ErrorKind::Http { status: None });
        let log = Arc::new(ObservationLog::default());
        let context = AdapterContext::new(log.clone(), Subject::default(), "call");
        let mut request = http::Request::new(());
        request.extensions_mut().insert((context, "/completion"));
        let slot = AdapterSlot::default();
        slot.start(&request);
        slot.fail(&error);
        drop(slot);
        let trace = log.trace();
        assert_eq!(trace.observations.len(), 2);
        assert!(
            matches!(&trace.observations[1].action, Action::Adapter { observation }
            if observation.event == AdapterEvent::Finished { ending: AdapterEnding::Error {
                boundary, kind: "http".into(), status: None, retryable: report.is_retryable(),
            }})
        );
        assert_eq!(crate::error::ErrorReport::from(&error), report);
    }
}

#[tokio::test]
async fn streamed_body_failure_preserves_boundary_through_provider_error_conversion() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _,
        test_utils::SequencedStreamingHttpClient,
    };
    use futures::StreamExt;
    let mut baseline = None;
    for enabled in [false, true] {
        let http =
            SequencedStreamingHttpClient::new(vec![Err(crate::http_client::Error::instance(
                std::io::Error::other("scripted body transfer failure"),
            ))]);
        let client = crate::providers::gemini::Client::builder()
            .api_key("test-key")
            .http_client(http)
            .build()
            .unwrap();
        let model = client.completion_model("gemini-test");
        let log = Arc::new(ObservationLog::default());
        let request = model.completion_request("hello").build();
        let context = enabled.then(|| AdapterContext::new(log.clone(), Subject::default(), "call"));
        let mut stream = model.stream_with_context(request, context).await.unwrap();
        let error = loop {
            match stream.next().await {
                Some(Err(error)) => break error,
                Some(Ok(_)) => {}
                None => panic!("body error must reach the caller"),
            }
        };
        drop(stream);
        let report = error;
        if enabled {
            assert_eq!(baseline.as_ref(), Some(&report));
            let trace = log.trace();
            let endings: Vec<_> = trace
                .observations
                .iter()
                .filter_map(|o| match &o.action {
                    Action::Adapter { observation } => match &observation.event {
                        AdapterEvent::Finished { ending } => Some(ending),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            assert_eq!(
                endings,
                [&AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::Transport,
                    kind: report.kind.code().into(),
                    status: report.http_status,
                    retryable: report.is_retryable(),
                }]
            );
        } else {
            baseline = Some(report);
            assert!(log.trace().observations.is_empty());
        }
    }
}

#[test]
fn host_attempts_share_send_ordinals_without_relabeling_in_flight_facts() {
    let log = Arc::new(ObservationLog::default());
    let operation = AdapterContext::new(log.clone(), Subject::default(), "logical-call");
    let first_subject = Subject::scoped("run/1");
    let retry_subject = Subject::scoped("run/2");
    let first = operation.for_host_attempt(first_subject.clone(), 1.try_into().unwrap());
    let retry = operation.for_host_attempt(retry_subject.clone(), 2.try_into().unwrap());
    let mut send_one = first.begin(&http::Method::POST, "/completion").unwrap();
    let mut send_two = retry.begin(&http::Method::POST, "/completion").unwrap();
    // An adapter may send again within the same host dispatch.
    let send_three = first
        .clone()
        .begin(&http::Method::POST, "/completion")
        .unwrap();
    send_two.finish(AdapterEnding::Decoded);
    send_one.finish(AdapterEnding::Error {
        boundary: AdapterErrorBoundary::ProviderResponse,
        kind: "http".into(),
        status: Some(503),
        retryable: true,
    });
    drop(send_three);
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 6);
    let expected = [(1, 1), (2, 2), (3, 1), (2, 2), (1, 1), (3, 1)];
    for (o, (send, host)) in trace.observations.iter().zip(expected) {
        let Action::Adapter { observation } = &o.action else {
            panic!("adapter fact");
        };
        assert_eq!(observation.operation, "logical-call");
        assert_eq!(observation.attempt, Some(send));
        assert_eq!(
            observation.host_attempt.map(std::num::NonZeroU64::get),
            Some(host)
        );
        assert_eq!(
            o.subject,
            if host == 1 {
                first_subject.clone()
            } else {
                retry_subject.clone()
            }
        );
    }
    let decoded: crate::observe::ObservationTrace =
        serde_json::from_str(&serde_json::to_string(&trace).unwrap()).unwrap();
    assert_eq!(decoded, trace);
    let separate = AdapterContext::new(log.clone(), Subject::default(), "another-call");
    drop(separate.begin(&http::Method::POST, "/completion"));
    let last = log.trace().observations.pop().unwrap();
    assert!(matches!(last.action, Action::Adapter { observation }
        if observation.operation == "another-call" && observation.attempt == Some(1) && observation.host_attempt.is_none()));
}

#[test]
fn cloned_context_numbers_attempts_and_closes_once_without_payloads() {
    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "operation-1");
    let mut first = context
        .begin(&http::Method::POST, "/models/{model}")
        .unwrap();
    first.response(http::StatusCode::OK);
    first.response(http::StatusCode::OK);
    first.finish(AdapterEnding::Decoded);
    drop(first);
    drop(
        context
            .clone()
            .begin(&http::Method::POST, "/models/{model}"),
    );
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 5);
    let facts: Vec<_> = trace
        .observations
        .iter()
        .map(|o| {
            let Action::Adapter { observation } = &o.action else {
                panic!("adapter fact")
            };
            observation
        })
        .collect();
    assert_eq!(
        facts.iter().map(|f| f.attempt).collect::<Vec<_>>(),
        [Some(1), Some(1), Some(1), Some(2), Some(2)]
    );
    assert_eq!(
        facts[4].event,
        AdapterEvent::Finished {
            ending: AdapterEnding::Dropped
        }
    );
    let encoded = serde_json::to_string(&trace).unwrap();
    assert_eq!(
        serde_json::from_str::<crate::observe::ObservationTrace>(&encoded).unwrap(),
        trace
    );
}

#[test]
fn context_is_not_part_of_serialized_completion_requests() {
    let request = crate::completion::CompletionRequestBuilder::unbound("hello").build();
    let encoded = serde_json::to_string(&request).unwrap();
    assert!(!encoded.contains("local-only-secret"));
    assert!(!encoded.contains("observation"));
    let decoded: crate::completion::CompletionRequest = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(request).unwrap()
    );
}

#[tokio::test]
async fn gemini_unary_emits_the_actual_http_boundary_without_changing_the_request() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _, test_utils::RecordingHttpClient,
    };
    let body = r#"{"candidates":[{"content":{"parts":[{"text":"pong"}],"role":"model"},"finishReason":"STOP"}]}"#;
    let http = RecordingHttpClient::new(body);
    let client = crate::providers::gemini::Client::builder()
        .api_key("synthetic-secret-key")
        .http_client(http.clone())
        .build()
        .unwrap();
    let model = client.completion_model("gemini-test");
    let plain = model.completion_request("hello").build();
    model.completion(plain.clone()).await.unwrap();
    let log = Arc::new(ObservationLog::default().with_clock(Arc::new(ManualClock::default())));
    let observed = plain;
    let context = Some(AdapterContext::new(
        log.clone(),
        Subject::default(),
        "call-1",
    ));
    model
        .completion_with_context(observed, context)
        .await
        .unwrap();
    let requests = http.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0], requests[1]);
    let trace = log.trace();
    let events: Vec<_> = trace
        .observations
        .iter()
        .map(|o| {
            let Action::Adapter { observation } = &o.action else {
                panic!("adapter fact")
            };
            observation.event.clone()
        })
        .collect();
    assert_eq!(
        events,
        [
            AdapterEvent::Started {
                method: "POST".into(),
                route: "/models/{model}:generateContent".into()
            },
            AdapterEvent::Response { status: 200 },
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("STOP".into()),
                    ..AdapterVerdict::default()
                }
            },
            AdapterEvent::Finished {
                ending: AdapterEnding::Decoded
            },
        ]
    );
    let Action::Adapter { observation } = &trace.observations.last().unwrap().action else {
        panic!()
    };
    let timing = observation
        .analysis
        .as_ref()
        .unwrap()
        .timing
        .as_ref()
        .unwrap();
    assert_eq!(timing.request_duration, Some(std::time::Duration::ZERO));
    assert_eq!(
        timing.time_to_first_byte, None,
        "buffered unary transport has no first-byte boundary"
    );
    assert!(
        !serde_json::to_string(&trace)
            .unwrap()
            .contains("synthetic-secret-key")
    );
}

#[test]
fn exhausted_identity_never_wraps_or_reuses_an_attempt() {
    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "last-call");
    *context.inner.next.lock().unwrap() = Some(u64::MAX);
    drop(context.begin(&http::Method::POST, "/completion"));
    assert!(context.begin(&http::Method::POST, "/completion").is_none());
    let trace = log.trace();
    assert!(trace.observations.iter().any(|o| matches!(
        &o.action, Action::Adapter { observation } if observation.event == AdapterEvent::IdentityExhausted
    )));
}

#[tokio::test]
async fn unary_failure_facts_preserve_retryability_without_copying_error_bodies() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _, test_utils::RecordingHttpClient,
    };
    let http = RecordingHttpClient::with_error(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{"message":"synthetic-sensitive-body"},"usageMetadata":{"promptTokenCount":3}}"#,
    );
    let client = crate::providers::gemini::Client::builder()
        .api_key("synthetic-sensitive-body")
        .http_client(http.clone())
        .build()
        .unwrap();
    let model = client.completion_model("gemini-test");
    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "retry-operation");
    for _ in 0..2 {
        let error = model
            .completion_with_context(
                model.completion_request("hello").build(),
                Some(context.clone()),
            )
            .await
            .unwrap_err();
        assert!(error.is_retryable());
    }
    assert_eq!(http.requests().len(), 2);
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 10);
    for (index, observation) in trace.observations.iter().enumerate() {
        let Action::Adapter { observation } = &observation.action else {
            panic!("adapter fact")
        };
        assert_eq!(observation.attempt, Some((index / 5 + 1) as u64));
        if index % 5 == 2 {
            assert_eq!(
                observation.event,
                AdapterEvent::Usage {
                    usage: AdapterUsage {
                        input_tokens: Some(3),
                        ..AdapterUsage::default()
                    }
                }
            );
        }
        if index % 5 == 3 {
            assert_eq!(
                observation.event,
                AdapterEvent::ErrorEnvelope {
                    error: AdapterErrorEnvelope {
                        message: Some("[redacted]".into()),
                        ..AdapterErrorEnvelope::default()
                    }
                }
            );
        }
        if index % 5 == 4 {
            assert_eq!(
                observation.event,
                AdapterEvent::Finished {
                    ending: AdapterEnding::Error {
                        boundary: AdapterErrorBoundary::ProviderResponse,
                        kind: "http".into(),
                        status: Some(429),
                        retryable: true
                    }
                }
            );
        }
    }
    assert!(
        !serde_json::to_string(&trace)
            .unwrap()
            .contains("synthetic-sensitive-body")
    );
}

#[derive(Clone)]
struct PendingHttp {
    body_pending: bool,
}

impl crate::http_client::HttpClientExt for PendingHttp {
    fn send<T, U>(
        &self,
        _request: http::Request<T>,
    ) -> impl Future<
        Output = crate::http_client::Result<http::Response<crate::http_client::LazyBody<U>>>,
    > + crate::wasm_compat::WasmCompatSend
    + 'static
    where
        T: Into<bytes::Bytes> + crate::wasm_compat::WasmCompatSend,
        U: From<bytes::Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
    {
        let body_pending = self.body_pending;
        async move {
            if !body_pending {
                return std::future::pending().await;
            }
            let body: crate::http_client::LazyBody<U> = Box::pin(std::future::pending());
            Ok(http::Response::new(body))
        }
    }

    fn send_multipart<U>(
        &self,
        _request: http::Request<crate::http_client::MultipartForm>,
    ) -> impl Future<
        Output = crate::http_client::Result<http::Response<crate::http_client::LazyBody<U>>>,
    > + crate::wasm_compat::WasmCompatSend
    + 'static
    where
        U: From<bytes::Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
    {
        std::future::pending()
    }

    fn send_streaming<T>(
        &self,
        _request: http::Request<T>,
    ) -> impl Future<Output = crate::http_client::Result<crate::http_client::StreamingResponse>>
    + crate::wasm_compat::WasmCompatSend
    where
        T: Into<bytes::Bytes> + crate::wasm_compat::WasmCompatSend,
    {
        let body_pending = self.body_pending;
        async move {
            if !body_pending {
                return std::future::pending().await;
            }
            let body: crate::http_client::sse::BoxedStream = Box::pin(futures::stream::pending());
            http::Response::builder()
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(body)
                .map_err(crate::http_client::Error::Protocol)
        }
    }
}

#[tokio::test]
async fn dropping_pending_transport_or_body_closes_the_attempt_once() {
    use crate::{client::CompletionClient, completion::CompletionModel as _};
    for body_pending in [false, true] {
        let client = crate::providers::gemini::Client::builder()
            .api_key("test-key")
            .http_client(PendingHttp { body_pending })
            .build()
            .unwrap();
        let model = client.completion_model("gemini-test");
        let log = Arc::new(ObservationLog::default());
        let request = model.completion_request("hello").build();
        let context = Some(AdapterContext::new(
            log.clone(),
            Subject::default(),
            "cancelled-call",
        ));
        let mut future = Box::pin(model.completion_with_context(request, context));
        assert!(futures::poll!(future.as_mut()).is_pending());
        drop(future);
        let trace = log.trace();
        assert_eq!(trace.observations.len(), if body_pending { 3 } else { 2 });
        assert!(matches!(&trace.observations.last().unwrap().action,
            Action::Adapter { observation } if observation.event == AdapterEvent::Finished { ending: AdapterEnding::Dropped }
        ));
    }
}

#[tokio::test]
async fn shared_arc_model_keeps_mixed_invocations_distinct_after_context_scope_ends() {
    use crate::{client::CompletionClient, completion::CompletionModel as _};
    use futures::StreamExt;

    let client = crate::providers::gemini::Client::builder()
        .api_key("test-key")
        .http_client(PendingHttp { body_pending: true })
        .build()
        .unwrap();
    let model = Arc::new(client.completion_model("gemini-test"));
    let sink = Arc::new(ObservationLog::default());
    let request = model.completion_request("same request").build();
    let mut stream = {
        let context = AdapterContext::new(sink.clone(), Subject::default(), "stream");
        model
            .stream_with_context(request.clone(), Some(context))
            .await
            .unwrap()
    };
    assert!(sink.is_empty(), "stream creation remains lazy");
    let mut unary = Box::pin(model.completion_with_context(
        request,
        Some(AdapterContext::new(
            sink.clone(),
            Subject::default(),
            "unary",
        )),
    ));
    assert!(futures::poll!(unary.as_mut()).is_pending());
    // Move the lazy stream to another task before starting its HTTP attempt.
    tokio::spawn(async move {
        assert!(futures::poll!(stream.next()).is_pending());
        drop(stream);
    })
    .await
    .unwrap();
    drop(unary);
    let trace = sink.trace();
    for operation in ["unary", "stream"] {
        let facts: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|fact| {
                let Action::Adapter { observation } = &fact.action else {
                    return None;
                };
                (observation.operation == operation).then_some(observation)
            })
            .collect();
        assert_eq!(facts.len(), 3);
        assert!(facts.iter().all(|fact| fact.attempt == Some(1)));
        assert_eq!(
            facts.last().unwrap().event,
            AdapterEvent::Finished {
                ending: AdapterEnding::Dropped
            }
        );
    }
    assert_eq!(trace.observations.len(), 6);
}

#[tokio::test]
async fn dropping_stream_pending_on_connection_or_body_closes_once() {
    use crate::{client::CompletionClient, completion::CompletionModel as _};
    use futures::StreamExt;
    for body_pending in [false, true] {
        let client = crate::providers::gemini::Client::builder()
            .api_key("test-key")
            .http_client(PendingHttp { body_pending })
            .build()
            .unwrap();
        let model = client.completion_model("gemini-test");
        let log = Arc::new(ObservationLog::default());
        let request = model.completion_request("hello").build();
        let context = Some(AdapterContext::new(
            log.clone(),
            Subject::default(),
            "pending-stream",
        ));
        let mut stream = model.stream_with_context(request, context).await.unwrap();
        assert!(log.is_empty());
        assert!(futures::poll!(stream.next()).is_pending());
        drop(stream);
        let trace = log.trace();
        assert_eq!(trace.observations.len(), if body_pending { 3 } else { 2 });
        assert!(matches!(&trace.observations.last().unwrap().action,
            Action::Adapter { observation } if observation.event == AdapterEvent::Finished { ending: AdapterEnding::Dropped }
        ));
    }
}

async fn observed_stream(bytes: &str, stop_after_first: bool) -> crate::observe::ObservationTrace {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _, test_utils::MockStreamingClient,
    };
    use futures::StreamExt;
    let client = crate::providers::gemini::Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::copy_from_slice(bytes.as_bytes()),
        })
        .build()
        .unwrap();
    let model = client.completion_model("gemini-test");
    let log = Arc::new(ObservationLog::default());
    let request = model.completion_request("hello").build();
    let mut plain_stream = model.stream(request.clone()).await.unwrap();
    let mut plain_items = Vec::new();
    while let Some(item) = plain_stream.next().await {
        let terminal = matches!(&item, Ok(crate::streaming::StreamEvent::Final(_)));
        plain_items.push(
            item.map(|event| serde_json::to_value(event).unwrap())
                .map_err(|e| e.to_string()),
        );
        if stop_after_first || terminal {
            break;
        }
    }
    drop(plain_stream);
    let context = Some(AdapterContext::new(
        log.clone(),
        Subject::default(),
        "stream-call",
    ));
    let mut stream = model.stream_with_context(request, context).await.unwrap();
    assert!(
        log.is_empty(),
        "an unpolled lazy stream has not sent a request"
    );
    let mut observed_items = Vec::new();
    while let Some(item) = stream.next().await {
        let terminal = matches!(&item, Ok(crate::streaming::StreamEvent::Final(_)));
        observed_items.push(
            item.map(|event| serde_json::to_value(event).unwrap())
                .map_err(|e| e.to_string()),
        );
        // Consumers may drop immediately on Final instead of polling None.
        // EOF already seen by the driver must survive that drop.
        if stop_after_first || terminal {
            break;
        }
    }
    drop(stream);
    assert_eq!(
        observed_items, plain_items,
        "observation preserves stream semantics"
    );
    log.trace()
}

#[tokio::test]
async fn stream_terminal_eof_error_and_drop_have_distinct_closures() {
    let content = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\"role\":\"model\"},\"index\":0}]}\n\n";
    let terminal = "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"index\":0}]}\n\n";
    let error = "data: {\"error\":{\"code\":503,\"message\":\"unavailable\",\"status\":\"UNAVAILABLE\"}}\n\n";
    for (bytes, stop, ending) in [
        (
            format!("{content}{terminal}"),
            false,
            AdapterEnding::Terminal,
        ),
        (content.to_owned(), false, AdapterEnding::Eof { after: 1 }),
        (
            format!("{content}data: {{"),
            false,
            AdapterEnding::PartialFrame {
                byte_count: 7,
                after: 1,
            },
        ),
        (
            error.to_owned(),
            false,
            AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "http".into(),
                status: Some(503),
                retryable: true,
            },
        ),
        (format!("{content}{terminal}"), true, AdapterEnding::Dropped),
    ] {
        let trace = observed_stream(&bytes, stop).await;
        let events: Vec<_> = trace
            .observations
            .iter()
            .map(|o| {
                let Action::Adapter { observation } = &o.action else {
                    panic!("adapter fact")
                };
                assert_eq!(observation.attempt, Some(1));
                &observation.event
            })
            .collect();
        assert_eq!(
            events[0],
            &AdapterEvent::Started {
                method: "POST".into(),
                route: "/models/{model}:streamGenerateContent".into()
            }
        );
        assert_eq!(events[1], &AdapterEvent::Response { status: 200 });
        assert_eq!(events.last().unwrap(), &&AdapterEvent::Finished { ending });
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AdapterEvent::Finished { .. }))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn corrupt_frame_is_evidence_separate_from_recovery_or_consumer_drop() {
    let corrupt = "data: {\n\n";
    let terminal = "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"index\":0}]}\n\n";
    for (stop, ending) in [
        (false, AdapterEnding::Terminal),
        (true, AdapterEnding::Dropped),
    ] {
        let trace = observed_stream(&format!("{corrupt}{terminal}"), stop).await;
        let events: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| {
                let Action::Adapter { observation } = &o.action else {
                    return None;
                };
                Some(&observation.event)
            })
            .collect();
        assert_eq!(events[2], &AdapterEvent::Corrupt { frame: 1 });
        assert_eq!(events.last().unwrap(), &&AdapterEvent::Finished { ending });
        assert_eq!(events.len(), if stop { 4 } else { 6 });
        assert!(!serde_json::to_string(&trace).unwrap().contains("data:"));
    }
}

#[tokio::test]
async fn provider_terminal_does_not_hide_partial_transport_eof() {
    let trace = observed_stream(
        concat!(
            "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"index\":0}]}\n\n",
            "data: {",
        ),
        false,
    )
    .await;
    let events: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|o| {
            let Action::Adapter { observation } = &o.action else {
                return None;
            };
            Some(&observation.event)
        })
        .collect();
    assert_eq!(
        events[3],
        &AdapterEvent::TransportEof {
            after: 1,
            partial_bytes: 7
        }
    );
    assert_eq!(
        events[4],
        &AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        }
    );
    assert_eq!(events.len(), 5);
}

#[tokio::test]
async fn empty_unary_rejection_preserves_optional_usage_before_failure() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _, test_utils::RecordingHttpClient,
    };
    for (metadata, expected) in [
        (
            serde_json::json!({"promptTokenCount": 7, "totalTokenCount": 9}),
            Some(AdapterUsage {
                input_tokens: Some(7),
                total_tokens: Some(9),
                ..AdapterUsage::default()
            }),
        ),
        (
            serde_json::json!({"candidatesTokenCount": 0, "promptTokenCount": -1}),
            Some(AdapterUsage {
                output_tokens: Some(0),
                ..AdapterUsage::default()
            }),
        ),
        (serde_json::Value::Null, None),
    ] {
        let body = serde_json::json!({"candidates": [], "usageMetadata": metadata}).to_string();
        let http = RecordingHttpClient::new(body);
        let client = crate::providers::gemini::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .unwrap();
        let model = client.completion_model("gemini-test");
        let request = model.completion_request("hello").build();
        let plain_error = model.completion(request.clone()).await.unwrap_err();
        let log = Arc::new(ObservationLog::default());
        let context = Some(AdapterContext::new(
            log.clone(),
            Subject::default(),
            "empty-call",
        ));
        let error = model
            .completion_with_context(request, context)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), plain_error.to_string());
        assert_eq!(http.requests()[0], http.requests()[1]);
        let trace = log.trace();
        let usage: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| match &o.action {
                Action::Adapter {
                    observation:
                        AdapterObservation {
                            event: AdapterEvent::Usage { usage },
                            ..
                        },
                } => Some(usage.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(usage, expected.into_iter().collect::<Vec<_>>());
        assert!(matches!(&trace.observations.last().unwrap().action,
            Action::Adapter { observation } if observation.event == AdapterEvent::Finished { ending: AdapterEnding::Error { boundary: AdapterErrorBoundary::Decode, kind: "response".into(), status: None, retryable: false } }
        ));
        let raw_log = Arc::new(ObservationLog::default());
        let raw_request = model.completion_request("hello").build();
        let context = Some(AdapterContext::new(
            raw_log.clone(),
            Subject::default(),
            "raw-empty-call",
        ));
        let raw = model
            .raw_completion_with_context(raw_request, context)
            .await
            .unwrap();
        assert!(
            raw.candidates.is_empty(),
            "the raw API must retain its decode-only contract"
        );
        assert!(
            matches!(&raw_log.trace().observations.last().unwrap().action,
                Action::Adapter { observation } if observation.event == AdapterEvent::Finished { ending: AdapterEnding::Decoded }
            )
        );
    }
}

#[tokio::test]
async fn streamed_usage_snapshots_keep_missing_counts_and_failed_attempt_usage() {
    let bytes = concat!(
        "data: {\"usageMetadata\":{\"promptTokenCount\":5,\"totalTokenCount\":5}}\n\n",
        "data: {\"usageMetadata\":{\"candidatesTokenCount\":0}}\n\n",
        "data: {\"error\":{\"code\":503,\"message\":\"unavailable\"}}\n\n",
    );
    let trace = observed_stream(bytes, false).await;
    let usage: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Adapter {
                observation:
                    AdapterObservation {
                        event: AdapterEvent::Usage { usage },
                        ..
                    },
            } => Some(usage.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        usage,
        [
            AdapterUsage {
                input_tokens: Some(5),
                total_tokens: Some(5),
                ..AdapterUsage::default()
            },
            AdapterUsage {
                output_tokens: Some(0),
                ..AdapterUsage::default()
            },
        ]
    );
    assert!(matches!(&trace.observations.last().unwrap().action,
        Action::Adapter { observation } if matches!(observation.event, AdapterEvent::Finished { ending: AdapterEnding::Error { status: Some(503), .. } })
    ));
}

#[tokio::test]
async fn streaming_http_rejection_preserves_usage_and_the_original_error() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _,
        test_utils::HttpErrorStreamingClient,
    };
    use futures::StreamExt;
    let client = crate::providers::gemini::Client::builder().api_key("synthetic-sensitive-body")
        .http_client(HttpErrorStreamingClient::new(http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"synthetic-sensitive-body"},"usageMetadata":{"promptTokenCount":3}}"#))
        .build().unwrap();
    let model = client.completion_model("gemini-test");
    let request = model.completion_request("hello").build();
    let mut plain = model.stream(request.clone()).await.unwrap();
    let plain_error = plain.next().await.unwrap().unwrap_err();
    assert!(plain.next().await.is_none());
    let log = Arc::new(ObservationLog::default());
    let context = Some(AdapterContext::new(
        log.clone(),
        Subject::default(),
        "rejected-stream",
    ));
    let mut stream = model.stream_with_context(request, context).await.unwrap();
    let error = stream.next().await.unwrap().unwrap_err();
    assert_eq!(error.to_string(), plain_error.to_string());
    assert!(stream.next().await.is_none());
    drop(stream);
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 5);
    assert!(matches!(&trace.observations[2].action,
        Action::Adapter { observation } if observation.event == AdapterEvent::Usage { usage: AdapterUsage { input_tokens: Some(3), ..AdapterUsage::default() } }
    ));
    assert!(matches!(&trace.observations[4].action,
        Action::Adapter { observation } if matches!(observation.event, AdapterEvent::Finished { ending: AdapterEnding::Error { status: Some(429), retryable: true, .. } })
    ));
    assert!(
        !serde_json::to_string(&trace)
            .unwrap()
            .contains("synthetic-sensitive-body")
    );
}

#[tokio::test]
async fn provider_metadata_and_headers_are_scrubbed_before_observation() {
    use crate::{
        client::CompletionClient, completion::CompletionModel as _, test_utils::RecordingHttpClient,
    };
    let secret = "synthetic-credential-12345";
    let body = serde_json::json!({
        "candidates": [{"finishReason": "MAX_TOKENS", "finishMessage": secret}],
        "modelVersion": "gemini-test", "responseId": secret,
        "error": {"code": "FUTURE_STATUS", "status": "RESOURCE_EXHAUSTED", "message": secret}
    })
    .to_string();
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", "2".parse().unwrap());
    headers.insert("x-request-id", secret.parse().unwrap());
    headers.insert(
        "x-ratelimit-reset-tokens",
        "x".repeat(4096).parse().unwrap(),
    );
    headers.insert("set-cookie", secret.parse().unwrap());
    headers.insert("authorization", secret.parse().unwrap());
    headers.insert("x-private-detail", secret.parse().unwrap());
    for returned_response in [false, true] {
        let http = if returned_response {
            RecordingHttpClient::with_error_response_headers(
                http::StatusCode::TOO_MANY_REQUESTS,
                body.clone(),
                headers.clone(),
            )
        } else {
            RecordingHttpClient::with_error_headers(
                http::StatusCode::TOO_MANY_REQUESTS,
                body.clone(),
                headers.clone(),
            )
        };
        let client = crate::providers::gemini::Client::builder()
            .api_key(secret)
            .http_client(http)
            .build()
            .unwrap();
        let model = client.completion_model("gemini-test");
        let log = Arc::new(ObservationLog::default());
        let request = model.completion_request("hello").build();
        let context = Some(AdapterContext::new(
            log.clone(),
            Subject::default(),
            "metadata-call",
        ));
        assert!(
            model
                .completion_with_context(request, context)
                .await
                .unwrap_err()
                .is_retryable()
        );
        let trace = log.trace();
        let facts: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| {
                let Action::Adapter { observation } = &o.action else {
                    return None;
                };
                Some(observation)
            })
            .collect();
        assert_eq!(facts.len(), 5);
        let captured = facts[1]
            .analysis
            .as_ref()
            .unwrap()
            .headers
            .as_ref()
            .unwrap();
        assert_eq!(captured.len(), 3);
        assert_eq!(captured["retry-after"], "2");
        assert_eq!(captured["x-request-id"], "[redacted]");
        assert_eq!(captured["x-ratelimit-reset-tokens"], "[truncated]");
        assert_eq!(
            facts[2].event,
            AdapterEvent::Provider {
                verdict: AdapterVerdict {
                    finish_reason: Some("MAX_TOKENS".into()),
                    detail: Some("[redacted]".into()),
                    model: Some("gemini-test".into()),
                    ..AdapterVerdict::default()
                }
            }
        );
        assert_eq!(
            facts[2].analysis.as_ref().unwrap().response_id.as_deref(),
            Some("[redacted]")
        );
        assert_eq!(
            facts[3].event,
            AdapterEvent::ErrorEnvelope {
                error: AdapterErrorEnvelope {
                    code: Some("FUTURE_STATUS".into()),
                    status: Some("RESOURCE_EXHAUSTED".into()),
                    message: Some("[redacted]".into())
                }
            }
        );
        let serialized = serde_json::to_string(&trace).unwrap();
        assert!(!serialized.contains(secret));
        assert!(!serialized.contains("set-cookie"));
        let mut changed = trace.clone();
        for o in &mut changed.observations {
            if let Action::Adapter { observation } = &mut o.action {
                observation.analysis = None;
            }
        }
        assert!(matches!(
            crate::observe::compare(&trace, &changed),
            crate::observe::Comparison::Equal
        ));
        if let Action::Adapter { observation } = &mut changed.observations[1].action {
            observation.event = AdapterEvent::Response { status: 503 };
        }
        assert!(matches!(
            crate::observe::compare(&trace, &changed),
            crate::observe::Comparison::Diverged(_)
        ));
    }
}

#[test]
fn scrubbing_controls_cannot_reconstitute_a_request_credential() {
    let secrets = vec!["credential-value".into()];
    assert_eq!(
        super::scrub::text("credential-\nvalue", &secrets),
        "[redacted]"
    );
    assert_eq!(
        super::scrub::text("https://example.invalid/?key=escaped%20value", &[]),
        "[redacted]"
    );
    assert_eq!(super::scrub::text(&"🦀".repeat(200), &[]), "[truncated]");
}

#[test]
fn url_credentials_are_scrubbed_in_messages_and_allowlisted_header_echoes() {
    for (uri, echoes) in [
        (
            "https://name%2Dpiece:opaque%2Dvalue@example.invalid/v1",
            vec![
                "name-piece",
                "name%2dpiece",
                "opaque-value",
                "opaque%2Dvalue",
            ],
        ),
        (
            "https://alice:opaque%252Dvalue@example.invalid/v1",
            vec!["opaque%2Dvalue", "opaque%252Dvalue"],
        ),
        (
            "https://example.invalid/v1?API_KEY=opaque%2Dvalue",
            vec!["opaque-value", "opaque%2dvalue"],
        ),
        (
            "/v1?%61ccess_token=opaque%2Dvalue",
            vec!["opaque-value", "opaque%2Dvalue"],
        ),
        (
            "/v1?key=opaque+value",
            vec!["opaque value", "opaque+value", "opaque%20value"],
        ),
        (
            "https://opaque+name:opaque%2Bvalue@example.invalid/v1",
            vec!["opaque+name", "opaque+value"],
        ),
    ] {
        let request = http::Request::builder().uri(uri).body(()).unwrap();
        let secrets = super::scrub::request_secrets(&request);
        for echo in echoes {
            assert_eq!(
                super::scrub::text(echo, &secrets),
                "[redacted]",
                "{uri}: {echo}"
            );
            let mut headers = http::HeaderMap::new();
            headers.insert("request-id", echo.parse().unwrap());
            assert_eq!(
                super::scrub::headers(&headers, &secrets)["request-id"],
                "[redacted]"
            );
        }
    }
    let secrets = super::diagnostic_url_secrets("/v1?search=ordinary-value&key=");
    assert!(secrets.is_empty());
    assert_eq!(
        super::scrub::text("ordinary%20value", &secrets),
        "ordinary%20value"
    );
}

#[test]
fn diagnostic_comparison_normalizes_both_sides_without_persisting_decoded_text() {
    for (secret, value) in [
        ("opaque-\nvalue", "opaque-value"),
        ("opaque-value", "opaque%2dvalue"),
        ("opaque-value", "opaque-%0Avalue"),
        ("opaque-%0Avalue", "opaque-value"),
        ("opaque%2Dvalue", "opaque%252Dvalue"),
        ("opaque%\n2Dvalue", "opaque-value"),
    ] {
        assert_eq!(super::scrub::text(value, &[secret.into()]), "[redacted]");
    }
    assert_eq!(super::scrub::text("ordinary", &["\n".into()]), "ordinary");
    assert_eq!(super::scrub::text("ordinary%2", &[]), "ordinary%2");
    assert_eq!(super::scrub::text("%61pi_key=opaque", &[]), "[redacted]");
    assert_eq!(super::scrub::text(&"%41".repeat(200), &[]), "[truncated]");
}

#[tokio::test]
async fn optional_response_ids_do_not_create_semantic_stream_events() {
    let content = serde_json::json!({
        "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "index": 0}]
    });
    let terminal = "data: {\"candidates\":[{\"finishReason\":\"STOP\",\"index\":0}]}\n\n";
    for (suffix, stop) in [("", false), (terminal, false), (terminal, true)] {
        let baseline = observed_stream(&format!("data: {content}\n\n{suffix}"), stop).await;
        for id in ["response-one", "response-two", "test-key"] {
            let mut identified = content.clone();
            identified["responseId"] = id.into();
            let actual = observed_stream(&format!("data: {identified}\n\n{suffix}"), stop).await;
            assert_eq!(
                crate::observe::compare(&baseline, &actual),
                crate::observe::Comparison::Equal
            );
            let ids: Vec<_> = actual
                .observations
                .iter()
                .filter_map(|o| {
                    let Action::Adapter { observation } = &o.action else {
                        return None;
                    };
                    observation.analysis.as_ref()?.response_id.as_deref()
                })
                .collect();
            assert_eq!(ids, [if id == "test-key" { "[redacted]" } else { id }]);
        }
    }
    // Actual insertion/removal must agree, including EOF and corruption positions.
    let baseline = observed_stream("", false).await;
    let actual = observed_stream("data: {\"responseId\":\"response-only\"}\n\n", false).await;
    assert_eq!(
        crate::observe::compare(&baseline, &actual),
        crate::observe::Comparison::Equal
    );
    assert!(actual.observations.iter().any(|o| matches!(
        &o.action, Action::Adapter { observation }
        if observation.analysis.as_ref().and_then(|a| a.response_id.as_deref()) == Some("response-only")
    )));
    for suffix in ["", "data: {broken\n\n", "data: partial"] {
        let baseline = observed_stream(&format!("data: {content}\n\n{suffix}"), false).await;
        let actual = observed_stream(
            &format!("data: {{\"responseId\":\"first\"}}\n\ndata: {content}\n\ndata: {{\"responseId\":\"last\"}}\n\n{suffix}"),
            false,
        ).await;
        assert_eq!(
            crate::observe::compare(&baseline, &actual),
            crate::observe::Comparison::Equal
        );
    }
    // Empty, unknown, invalid-ID and meaningful frames must still advance positions.
    for extra in [
        "{}",
        "{\"unknown\":1}",
        "{\"responseId\":42}",
        "{\"responseId\":\"a\",\"responseId\":\"b\"}",
        "{\"usageMetadata\":{\"totalTokenCount\":1}}",
    ] {
        let actual = observed_stream(&format!("data: {extra}\n\n"), false).await;
        assert_ne!(
            crate::observe::compare(&baseline, &actual),
            crate::observe::Comparison::Equal
        );
        assert!(actual.observations.iter().all(|fact| !matches!(
            &fact.action, Action::Adapter { observation }
            if observation.event == AdapterEvent::Corrupt { frame: 0 }
        )));
    }
}
