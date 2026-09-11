use std::{sync::Mutex, task::Context};

use futures::{StreamExt, executor::block_on, task::noop_waker_ref};

use super::*;

#[derive(Default)]
struct Seen {
    outcomes: Vec<Result<Outcome, ErrorReport>>,
    events: usize,
    discard_events: bool,
}

struct Observer(Arc<Mutex<Seen>>);

impl Observe for Observer {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.0.lock().expect("seen").outcomes.push(outcome.clone());
    }

    fn keep_events(&self) -> bool {
        !self.0.lock().expect("seen").discard_events
    }

    fn event(&mut self, _: &StreamEvent) {
        self.0.lock().expect("seen").events += 1;
    }

    fn discard(&mut self, _: &str) {}
    fn patch(&mut self, _: &EffectKind) {}
}

#[test]
fn resolved_stream_preserves_original_response_for_outcome_only_replay() {
    use crate::message::{AssistantContent, DocumentSourceKind, Image};

    for keep_events in [false, true] {
        let seen = Arc::new(Mutex::new(Seen {
            discard_events: !keep_events,
            ..Seen::default()
        }));
        let mut response = CompletionResponse::new(
            vec![
                AssistantContent::text("generated image"),
                AssistantContent::Image(Image {
                    data: DocumentSourceKind::base64("aW1hZ2U="),
                    ..Image::default()
                }),
            ],
            Default::default(),
            "test",
        );
        response.message_id = Some("message".into());
        response.response_id = Some("response".into());
        response.provider_request_id = Some("request".into());
        response.model = Some("image-model".into());
        let expected = serde_json::to_value(&response).expect("response JSON");
        let reply = Reply::Outcome(Ok(Outcome::Completion(response))).observed(
            true,
            Some(Observed {
                observer: Box::new(Observer(seen.clone())),
                told: false,
            }),
            None,
        );
        let delivered = block_on(reply.into_stream().collect::<Vec<_>>());
        assert!(
            delivered
                .iter()
                .any(|item| matches!(item, Ok(StreamEvent::Unknown(_))))
        );
        let seen = seen.lock().expect("seen");
        assert_eq!(seen.outcomes.len(), 1);
        let Ok(Outcome::Completion(recorded)) = &seen.outcomes[0] else {
            panic!("expected completed response");
        };
        assert_eq!(
            serde_json::to_value(recorded).expect("recorded JSON"),
            expected
        );
        assert_eq!(
            serde_json::to_value(events_from_response(recorded)).expect("replay events"),
            serde_json::to_value(&delivered).expect("delivered events"),
            "outcome-only replay must reconstruct the same image-bearing stream"
        );
        assert_eq!(seen.events, if keep_events { delivered.len() } else { 0 });
    }
}

#[test]
fn dropping_a_pending_deferred_handler_records_cancellation_and_closes_the_resolver() {
    struct External(Arc<Mutex<Option<Resolver>>>);
    impl Serve for External {
        type Family = crate::effect::family::Dynamic;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: crate::effect::HandlerKey::from("external"),
                family: crate::effect::FamilyDescriptor::Custom {
                    kind: "external".into(),
                },
                layers: vec![],
            }
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            let (resolver, answer) = deferred();
            *self.0.lock().expect("slot") = Some(resolver);
            Reply::Outcome(answer.await)
        }
    }
    let slot = Arc::new(Mutex::new(None));
    let handler = ErasedHandler::new(External(slot.clone()));
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut serving = handler.handle(
        EffectKind::Custom {
            kind: Arc::from("external"),
            payload: serde_json::Value::Null,
        },
        Dispatch::new(EffectId::from_raw(1), false).with_observer(Box::new(Observer(seen.clone()))),
    );
    assert!(
        serving
            .as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref()))
            .is_pending()
    );
    let resolver = slot.lock().expect("slot").take().expect("published");
    drop(serving);
    assert!(resolver.is_closed());
    assert_eq!(
        resolver.resolve(Ok(Outcome::Custom {
            payload: serde_json::Value::Null
        })),
        Err(SinkClosed)
    );
    let seen = seen.lock().expect("seen");
    assert_eq!(seen.outcomes.len(), 1);
    assert_eq!(
        seen.outcomes[0].as_ref().expect_err("cancelled").kind,
        ErrorKind::Cancelled
    );
}

#[test]
fn dropping_a_backpressured_writer_does_not_record_its_unpulled_final() {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished_in_writer = finished.clone();
    let reply = Reply::written(move |mut writer| async move {
        writer.text("prefix").await.expect("consumer present");
        writer
            .finish(StreamFinal::new("test", Default::default()))
            .await
            .expect("consumer present");
        finished_in_writer.store(true, std::sync::atomic::Ordering::SeqCst);
    })
    .observed(
        true,
        Some(Observed {
            observer: Box::new(Observer(seen.clone())),
            told: false,
        }),
        None,
    );
    let mut stream = reply.into_stream();
    assert!(matches!(
        stream
            .as_mut()
            .poll_next(&mut Context::from_waker(noop_waker_ref())),
        Poll::Ready(Some(Ok(_)))
    ));
    assert!(!finished.load(std::sync::atomic::Ordering::SeqCst));
    assert!(seen.lock().expect("seen").outcomes.is_empty());
    drop(stream);
    let seen = seen.lock().expect("seen");
    assert_eq!(seen.events, 1, "only the pulled prefix was observed");
    assert_eq!(seen.outcomes.len(), 1);
    assert_eq!(
        seen.outcomes[0].as_ref().expect_err("cancelled").kind,
        ErrorKind::Cancelled
    );
}

#[test]
fn dropping_the_writer_without_finishing_is_truncation() {
    let reply = Reply::written(|mut writer| async move {
        writer.text("prefix").await.expect("open");
    });
    assert_eq!(
        block_on(reply.into_outcome()).expect_err("truncated"),
        stream_truncated()
    );
}

#[test]
fn response_reemission_preserves_local_tool_ids_without_provider_provenance() {
    use crate::message::{AssistantContent, ToolCall, ToolCallId, ToolFunction};
    let mut calls = ["local-call", "tool-00", "tool-0", "wire-call"]
        .into_iter()
        .map(|id| {
            AssistantContent::ToolCall(ToolCall::new(
                ToolCallId::new(id).expect("nonempty local id"),
                ToolFunction::new("local".into(), serde_json::json!({"id": id})),
            ))
        })
        .collect::<Vec<_>>();
    let mut provider_call = ToolCall::new(
        ToolCallId::new("local-provider").expect("local provider call id"),
        ToolFunction::new("provider".into(), serde_json::json!({"x": 1})),
    );
    provider_call.provider = crate::message::ProviderCallId::new("wire-call")
        .map(|provider| provider.with_item_id("wire-item"));
    provider_call.signature = Some("signature".into());
    provider_call.additional_params = Some(serde_json::json!({"metadata": true}));
    calls.push(AssistantContent::ToolCall(provider_call));
    let response = CompletionResponse::new(calls.clone(), Default::default(), "local");
    let mut accumulator = crate::streaming::BlockAccumulator::new();
    let mut published = Vec::new();
    let events: Vec<Result<StreamEvent, ErrorReport>> = serde_json::from_value(
        serde_json::to_value(events_from_response(&response)).expect("serialize events"),
    )
    .expect("deserialize events");
    for event in events {
        let event = event.expect("response reemits");
        if let Some((_, content)) = accumulator.apply(&event).expect("event folds") {
            published.push(content);
        }
    }
    assert_eq!(
        published, calls,
        "completed events preserve local identities"
    );
    assert_eq!(
        accumulator.finish(),
        calls,
        "final response preserves local identities"
    );
}

#[test]
fn writer_execution_outlives_its_final_until_the_owned_future_finishes() {
    let (release, wait) = futures::channel::oneshot::channel::<()>();
    let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = completed.clone();
    let mut stream = Reply::written(move |writer| async move {
        writer
            .finish(StreamFinal::new("writer", Default::default()))
            .await
            .expect("open");
        wait.await.expect("released");
        finished.store(true, std::sync::atomic::Ordering::SeqCst);
    })
    .into_stream();
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(
        stream.as_mut().poll_next(&mut cx),
        std::task::Poll::Ready(Some(Ok(StreamEvent::Final(_))))
    ));
    assert!(
        stream.as_mut().poll_next(&mut cx).is_pending(),
        "Final must not end the owned writing future"
    );
    assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
    release.send(()).expect("writer remains alive");
    assert!(matches!(
        stream.as_mut().poll_next(&mut cx),
        std::task::Poll::Ready(None)
    ));
    assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn terminal_items_carry_the_original_answer_in_one_observer_call() {
    use crate::message::{AssistantContent, DocumentSourceKind, Image};
    type Observation = (
        Result<StreamEvent, ErrorReport>,
        Option<Result<Outcome, ErrorReport>>,
    );
    struct AtomicObserver(Arc<Mutex<Vec<Observation>>>);
    impl Observe for AtomicObserver {
        fn outcome(&mut self, _: &Result<Outcome, ErrorReport>) {
            panic!("a terminal item must carry its answer in stream_item");
        }
        fn keep_events(&self) -> bool {
            true
        }
        fn event(&mut self, _: &StreamEvent) {
            panic!("separate event callback");
        }
        fn stream_error(&mut self, _: &ErrorReport) {
            panic!("separate error callback");
        }
        fn stream_item(
            &mut self,
            item: &Result<StreamEvent, ErrorReport>,
            outcome: Option<&Result<Outcome, ErrorReport>>,
        ) {
            self.0
                .lock()
                .expect("observations")
                .push((item.clone(), outcome.cloned()));
        }
        fn discard(&mut self, _: &str) {}
        fn patch(&mut self, _: &EffectKind) {}
    }
    let response = CompletionResponse::new(
        vec![AssistantContent::Image(Image {
            data: DocumentSourceKind::base64("aW1hZ2U="),
            ..Image::default()
        })],
        Default::default(),
        "image-provider",
    );
    let original = Ok(Outcome::Completion(response));
    let error = ErrorReport::new(ErrorKind::Response, "first error");
    let final_event = Ok(StreamEvent::Final(StreamFinal::new(
        "test",
        Default::default(),
    )));
    let after = Ok(StreamEvent::Unknown(crate::streaming::UnknownPayload::new(
        serde_json::json!({"after": true}),
    )));
    let terminal_answer = StreamTap::new()
        .observe(&final_event)
        .expect("terminal folds");
    let cases = [
        (Reply::Outcome(original.clone()), original),
        (
            Reply::Stream(Box::pin(futures::stream::iter(vec![
                final_event.clone(),
                after.clone(),
                Err(error.clone()),
            ]))),
            terminal_answer,
        ),
        (
            Reply::Stream(Box::pin(futures::stream::iter(vec![
                Err(error.clone()),
                final_event,
                after,
            ]))),
            Err(error),
        ),
    ];
    for (reply, expected) in cases {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let reply = reply.observed(
            true,
            Some(Observed {
                observer: Box::new(AtomicObserver(calls.clone())),
                told: false,
            }),
            None,
        );
        let delivered = block_on(reply.into_stream().collect::<Vec<_>>());
        let calls = calls.lock().expect("observations");
        assert_eq!(calls.len(), delivered.len());
        let answers: Vec<_> = calls
            .iter()
            .filter_map(|(item, answer)| answer.as_ref().map(|answer| (item, answer)))
            .collect();
        assert_eq!(answers.len(), 1);
        assert!(matches!(answers[0].0, Ok(StreamEvent::Final(_)) | Err(_)));
        assert_eq!(
            serde_json::to_value(answers[0].1).expect("answer"),
            serde_json::to_value(expected).expect("expected")
        );
    }
}

struct ProviderObserver(crate::observe::AdapterContext);

#[test]
fn explicit_dispatch_context_wins_in_both_installation_orders_and_through_layers() {
    use crate::observe::{AdapterContext, ObservationLog, Subject};

    let sink = Arc::new(ObservationLog::default());
    for explicit_first in [false, true] {
        let explicit = AdapterContext::new(sink.clone(), Subject::default(), "explicit");
        let observed = AdapterContext::new(sink.clone(), Subject::default(), "observer");
        let dispatch = Dispatch::new(EffectId::from_raw(1), false);
        let mut dispatch = if explicit_first {
            dispatch
                .with_adapter_context(explicit)
                .with_observer(Box::new(ProviderObserver(observed)))
        } else {
            dispatch
                .with_observer(Box::new(ProviderObserver(observed)))
                .with_adapter_context(explicit)
        };
        assert_eq!(dispatch.adapter_context().unwrap().operation(), "explicit");
        let inner = dispatch.inner(None);
        assert_eq!(inner.adapter_context().unwrap().operation(), "explicit");
        assert!(sink.trace().observations.is_empty());
    }
}

#[test]
fn replacing_observer_replaces_only_observer_derived_provider_context() {
    use crate::observe::{AdapterContext, ObservationLog, Subject};
    let sink = Arc::new(ObservationLog::default());
    for explicit in [false, true] {
        let mut dispatch = Dispatch::new(EffectId::from_raw(1), false);
        if explicit {
            dispatch = dispatch.with_adapter_context(AdapterContext::new(
                sink.clone(),
                Subject::default(),
                "caller",
            ));
        }
        for operation in ["first", "replacement"] {
            dispatch = dispatch.with_observer(Box::new(ProviderObserver(AdapterContext::new(
                sink.clone(),
                Subject::default(),
                operation,
            ))));
            assert_eq!(
                dispatch.adapter_context().unwrap().operation(),
                if explicit { "caller" } else { operation }
            );
        }
        dispatch =
            dispatch.with_observer(Box::new(Observer(Arc::new(Mutex::new(Seen::default())))));
        assert_eq!(
            dispatch
                .adapter_context()
                .as_ref()
                .map(|context| context.operation()),
            explicit.then_some("caller")
        );
    }
}

impl Observe for ProviderObserver {
    fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        Some(self.0.clone())
    }
    fn outcome(&mut self, _: &Result<Outcome, ErrorReport>) {}
    fn keep_events(&self) -> bool {
        false
    }
    fn event(&mut self, _: &StreamEvent) {}
    fn discard(&mut self, _: &str) {}
    fn patch(&mut self, _: &EffectKind) {}
}

#[tokio::test]
async fn provider_context_survives_inner_dispatch_and_explicit_call_context_wins() {
    use crate::{
        client::CompletionClient,
        completion::CompletionModel as _,
        observe::{Action, AdapterContext, ObservationLog, Subject},
        test_utils::RecordingHttpClient,
    };
    let body = r#"{"candidates":[{"content":{"parts":[{"text":"pong"}],"role":"model"},"finishReason":"STOP"}]}"#;
    let client = crate::providers::gemini::Client::builder()
        .api_key("key")
        .http_client(RecordingHttpClient::new(body))
        .build()
        .unwrap();
    let model = client.completion_model("gemini-test");
    let handler = crate::serve::adapters::CompletionAdapter::new("gemini-test", model.clone());
    let bus_log = Arc::new(ObservationLog::default());
    let direct_log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(bus_log.clone(), Subject::default(), "bus-operation");
    for explicit in [false, true] {
        let mut dispatch = Dispatch::new(EffectId::from_raw(1), false)
            .with_observer(Box::new(ProviderObserver(context.clone())));
        let request = model.completion_request("hello").build();
        if explicit {
            dispatch = dispatch.with_adapter_context(AdapterContext::new(
                direct_log.clone(),
                Subject::default(),
                "explicit-operation",
            ));
        }
        let reply = Serve::serve(
            &handler,
            EffectKind::Completion {
                request,
                stream: false,
            },
            dispatch.inner(None),
        )
        .await;
        assert!(reply.into_outcome().await.is_ok());
    }
    for (log, operation) in [
        (bus_log, "bus-operation"),
        (direct_log, "explicit-operation"),
    ] {
        let trace = log.trace();
        assert_eq!(trace.observations.len(), 4);
        assert!(trace.observations.iter().all(|o| matches!(&o.action,
            Action::Adapter { observation } if observation.operation == operation && observation.attempt == Some(1)
        )));
    }
}
