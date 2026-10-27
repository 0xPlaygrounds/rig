use std::{sync::Mutex, task::Context};

use futures::{StreamExt, executor::block_on, task::noop_waker_ref};

use super::*;
use crate::completion::CompletionResponse;
use crate::operation::Finish;
use crate::streaming::{Item, Part, Relayed, StreamEvent, UnknownPayload};

#[derive(Default)]
struct Seen {
    outcomes: Vec<Result<Answered, ErrorReport>>,
    context: Option<crate::observe::AdapterContext>,
}

struct Observer(Arc<Mutex<Seen>>);

impl Observe for Observer {
    fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        self.0.lock().expect("seen").context.clone()
    }

    fn origin(&mut self, _origin: &crate::message::Origin) {}
    fn outcome(&mut self, outcome: Result<Answered, ErrorReport>) {
        self.0.lock().expect("seen").outcomes.push(outcome);
    }

    fn keep_events(&self) -> bool {
        true
    }

    fn event(&mut self, _: &Item<StreamEvent>) {}

    fn discard(&mut self, _: &str) {}
    fn patch(&mut self, _: &EffectKind) {}
}

/// An observer's adapter context reaches the handler unless the caller
/// set one, whichever was installed first.
#[test]
fn an_observer_never_overwrites_the_callers_adapter_context() {
    use crate::observe::{AdapterContext, ObservationLog, Subject};
    let log = Arc::new(ObservationLog::default());
    let context = |operation: &str| AdapterContext::new(log.clone(), Subject::default(), operation);
    let observer = || {
        let seen = Seen {
            context: Some(context("observer")),
            ..Seen::default()
        };
        Box::new(Observer(Arc::new(Mutex::new(seen))))
    };
    let operation = |dispatch: Dispatch| {
        dispatch
            .adapter_context()
            .map(|it| it.operation().to_owned())
    };
    let observed = Dispatch::new(EffectId::from_raw(1), false).with_observer(observer());
    assert_eq!(operation(observed).as_deref(), Some("observer"));
    let first = Dispatch::new(EffectId::from_raw(2), false).with_adapter_context(context("caller"));
    assert_eq!(
        operation(first.with_observer(observer())).as_deref(),
        Some("caller")
    );
}

#[test]
fn a_written_reply_names_its_origin_before_its_first_item() {
    let origin = crate::message::Origin::new("example.api", "example", "example-1");
    let Reply::Stream(stream) = Reply::written(origin.clone(), |mut writer| async move {
        writer.text("prefix").await.expect("open");
    }) else {
        panic!("a written reply is a stream");
    };
    let first: Vec<_> = block_on(stream.take(2).collect());
    assert!(
        matches!(first.as_slice(), [Ok(Relayed::Origin(sent)), Ok(Relayed::Item(_))] if *sent == origin),
        "{first:?}"
    );
}

#[test]
fn restating_stored_float_calls_preserves_native_currentness() {
    use crate::message::{AssistantContent, CallId, Origin, ToolCall, ToolFunction, ToolName};
    use serde_json::json;

    for edited in [false, true] {
        let original = 0.100_000_000_000_000_03_f64;
        let mut block = AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_1"),
            ToolFunction::new(
                ToolName::new("measure").expect("tool name"),
                json!({"value": original}),
            ),
        ))
        .with_native(json!({"id": "fc_1", "signature": "keep"}));
        if edited && let AssistantContent::ToolCall(call) = &mut block {
            call.function.arguments.insert(
                "value".into(),
                json!(f64::from_bits(original.to_bits() + 1)),
            );
        }
        let block: AssistantContent =
            serde_json::from_str(&serde_json::to_string(&block).expect("save call"))
                .expect("load call");
        let response = Arc::new(CompletionResponse::new(
            vec![block.clone()],
            crate::completion::Usage::default(),
            Origin::new("test", "test", "test"),
            json!({}),
        ));
        let items: Vec<_> = block_on(
            Reply::Outcome(Ok(Answered::Completion(response.clone())))
                .into_stream()
                .collect(),
        );
        let ended = items
            .iter()
            .find_map(|item| match item.as_ref().ok()? {
                Relayed::Item(Item::Event(StreamEvent::End { content, .. })) => Some(content),
                _ => None,
            })
            .expect("ended call");
        let done = items
            .iter()
            .find_map(|item| match item.as_ref().ok()? {
                Relayed::Done(response) => Some(response),
                _ => None,
            })
            .expect("final response");
        assert_eq!(ended, &block);
        assert_eq!(done, &response);
        assert_eq!(ended.native_item().is_some(), !edited);
        assert_eq!(
            done.choice
                .first()
                .and_then(AssistantContent::native_item)
                .is_some(),
            !edited
        );
    }
}

/// A writer relaying a provider's reply reports its request id and document
/// in place of a transport; an empty id is no id.
#[test]
fn a_writer_reports_the_request_id_and_document_of_the_reply_it_relays() {
    let finished = |request_id: &'static str| {
        let items: Vec<_> = block_on(
            Reply::written(
                crate::message::Origin::new("relay", "relay", "relay"),
                move |mut writer| async move {
                    writer.request_id(request_id);
                    writer.raw(serde_json::json!({"id": "body"}));
                    writer.finish(Finish::default()).await.expect("open");
                },
            )
            .into_stream()
            .collect(),
        );
        let Some(Ok(Relayed::Done(done))) = items.into_iter().last() else {
            panic!("the response ends the stream");
        };
        done
    };
    let response = finished("req-1");
    assert_eq!(response.provider(), "relay");
    assert_eq!(response.provider_request_id.as_deref(), Some("req-1"));
    assert_eq!(response.raw, serde_json::json!({"id": "body"}));
    assert_eq!(finished("").provider_request_id, None);
}

#[test]
fn writer_execution_outlives_its_final_until_the_owned_future_finishes() {
    let (release, wait) = futures::channel::oneshot::channel::<()>();
    let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = completed.clone();
    let mut stream = Reply::written(
        crate::message::Origin::new("writer", "writer", "writer"),
        move |writer| async move {
            writer.finish(Finish::default()).await.expect("open");
            wait.await.expect("released");
            finished.store(true, std::sync::atomic::Ordering::SeqCst);
        },
    )
    .into_stream();
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(
        stream.as_mut().poll_next(&mut cx),
        std::task::Poll::Ready(Some(Ok(Relayed::Origin(_))))
    ));
    assert!(matches!(
        stream.as_mut().poll_next(&mut cx),
        std::task::Poll::Ready(Some(Ok(Relayed::Done(_))))
    ));
    assert!(
        stream.as_mut().poll_next(&mut cx).is_pending(),
        "the response must not end the owned writing future"
    );
    assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
    release.send(()).expect("writer remains alive");
    assert!(matches!(
        stream.as_mut().poll_next(&mut cx),
        std::task::Poll::Ready(None)
    ));
    assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
}

/// Recording is consumer-invisible: whatever shape a handler answers in —
/// a completed response, a setup failure, a stream with an in-band error,
/// one cut short before its terminal, one with a frame after it — the
/// items a streaming consumer receives and the outcome a unary consumer
/// receives are the same with an observer attached and without one, and
/// the observer is told exactly one outcome.
#[test]
fn an_observer_never_changes_what_the_consumer_receives() {
    use crate::message::{AssistantContent, DocumentSourceKind, Image};

    type Shape = fn() -> Reply;
    fn terminal() -> Result<Relayed, ErrorReport> {
        Ok(Relayed::Done(Arc::new(CompletionResponse::new(
            vec![AssistantContent::text("body")],
            Default::default(),
            crate::message::Origin::new("test.api", "test", ""),
            serde_json::json!({}),
        ))))
    }
    fn text(fragment: &str) -> Result<Relayed, ErrorReport> {
        Ok(text_item(fragment))
    }
    fn items(items: Vec<Result<Relayed, ErrorReport>>) -> Reply {
        Reply::Stream(Box::pin(futures::stream::iter(items)))
    }
    let shapes: [(&str, Shape); 6] = [
        ("a completed response with an image", || {
            Reply::Outcome(Ok(Answered::Completion(
                CompletionResponse::new(
                    vec![
                        AssistantContent::text("done"),
                        AssistantContent::Image(Image {
                            data: DocumentSourceKind::base64("aW1hZ2U="),
                            ..Image::default()
                        }),
                    ],
                    Default::default(),
                    crate::message::Origin::new("test.api", "test", ""),
                    serde_json::json!({}),
                )
                .into(),
            )))
        }),
        ("a setup failure", || {
            Reply::Outcome(Err(ErrorReport::new(ErrorKind::Provider, "refused")))
        }),
        ("an in-band error before the terminal", || {
            items(vec![
                Err(ErrorReport::new(ErrorKind::Response, "mid-stream")),
                text("after"),
                terminal(),
            ])
        }),
        ("a stream cut short before its terminal", || {
            items(vec![text("prefix")])
        }),
        ("a frame after the terminal", || {
            items(vec![
                text("body"),
                terminal(),
                Ok(Relayed::Item(Item::Unknown(UnknownPayload::new(
                    serde_json::json!({
                        "late": true
                    }),
                )))),
            ])
        }),
        ("a stream that ends at once", || items(Vec::new())),
    ];
    for (name, shape) in shapes {
        for streaming in [true, false] {
            let seen = Arc::new(Mutex::new(Seen::default()));
            let observed = Observed {
                observer: Box::new(Observer(seen.clone())),
                told: false,
            };
            if streaming {
                assert_eq!(
                    block_on(
                        shape()
                            .observed(true, None, None)
                            .into_stream()
                            .collect::<Vec<_>>(),
                    ),
                    block_on(
                        shape()
                            .observed(true, Some(observed), None)
                            .into_stream()
                            .collect::<Vec<_>>(),
                    ),
                    "{name}, streaming"
                );
            } else {
                assert_eq!(
                    serde_json::to_value(block_on(
                        shape().observed(false, None, None).into_outcome(),
                    ))
                    .expect("serde"),
                    serde_json::to_value(block_on(
                        shape().observed(false, Some(observed), None).into_outcome(),
                    ))
                    .expect("serde"),
                    "{name}, unary"
                );
            }
            assert_eq!(
                seen.lock().expect("seen").outcomes.len(),
                1,
                "{name}, streaming: {streaming}: one outcome told"
            );
        }
    }
}

/// A text fragment of the reply's first part.
fn text_item(fragment: &str) -> Relayed {
    Relayed::Item(Item::Event(StreamEvent::Text {
        part: Part::new(0),
        text: fragment.to_owned(),
    }))
}

/// A response the origin folded, carrying `text`.
fn done(text: &str) -> Result<Relayed, ErrorReport> {
    Ok(Relayed::Done(Arc::new(CompletionResponse::new(
        vec![crate::message::AssistantContent::text(text)],
        Default::default(),
        crate::message::Origin::new("test.api", "local", ""),
        serde_json::Value::Null,
    ))))
}

/// A tap yields one outcome, the first: an error item is the outcome, and
/// nothing observed after it yields another.
#[test]
fn the_tap_yields_only_its_first_outcome() {
    let items: Vec<Result<Relayed, ErrorReport>> = vec![
        Err(ErrorReport::new(ErrorKind::Provider, "reset")),
        Ok(text_item("late")),
        done("late"),
    ];
    let mut tap = StreamTap::new();
    let outcomes: Vec<_> = items.iter().filter_map(|item| tap.observe(item)).collect();
    assert_eq!(outcomes.len(), 1, "{outcomes:?}");
    assert!(matches!(outcomes.first(), Some(Err(report)) if report.message == "reset"));
}

/// An unavailable handler keeps the descriptor it was given and answers
/// every effect, unary or streamed, `HandlerUnavailable` with its reason.
#[test]
fn an_unavailable_handler_answers_every_effect_with_its_reason() {
    let descriptor = crate::effect::HandlerDescriptor::tool("lookup", "", serde_json::json!({}));
    let handler = ErasedHandler::unavailable(descriptor.clone(), "answered from the batch API");
    assert_eq!(handler.descriptor(), descriptor);
    let kinds = [
        EffectKind::Completion {
            request: crate::completion::CompletionRequest::new("hi").into(),
            stream: true,
        },
        EffectKind::ToolCall {
            name: "lookup".to_owned(),
            args: "{}".to_owned(),
        },
    ];
    for (at, kind) in (1..).zip(kinds) {
        let streams = kind.streams();
        let dispatch = Dispatch::new(EffectId::from_raw(at), streams);
        let answer = block_on(async { handler.handle(kind, dispatch).await.into_outcome().await });
        assert!(
            matches!(
                &answer,
                Err(report) if report.kind == ErrorKind::HandlerUnavailable
                    && report.message == "answered from the batch API"
            ),
            "{answer:?}"
        );
    }
}

/// A model's handler says which options of a completion its model would
/// leave out, through every wrapper a host puts around it (layers, stream
/// maps, sharing, erasing again), and sends nothing to say so; a tool call
/// has nothing to leave out.
#[test]
fn a_model_handler_names_what_its_model_leaves_out() {
    struct Pass;
    impl crate::serve::Intercept for Pass {
        fn name(&self) -> String {
            "pass".to_owned()
        }
        async fn before(&self, _: EffectId, _: &EffectKind) -> crate::serve::BeforeServe {
            crate::serve::BeforeServe::Proceed
        }
        async fn after(
            &self,
            _: EffectId,
            _: &EffectKind,
            _: &Result<Answered, ErrorReport>,
        ) -> crate::serve::AfterServe {
            crate::serve::AfterServe::Keep
        }
    }
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let model = crate::catalog::Catalog::builtin()
        .connect_with(
            "anthropic/claude-fable-5",
            crate::providers::registry::ConnectOptions::new()
                .api_key("sk-test")
                .http(http.clone()),
        )
        .expect("connects");
    let handler = ErasedHandler::new(crate::serve::adapters::ModelAdapter::new(
        "anthropic/claude-fable-5",
        model,
    ))
    .layered(Pass)
    .map_stream("same", |_| |relayed| vec![relayed]);
    // Shared and erased again, as a host registers one handler twice.
    let handler = ErasedHandler::new(Arc::new(ErasedHandler::new(handler)));
    let request = crate::completion::CompletionRequest::new("hi")
        .temperature(0.2)
        .top_p(0.9);
    let left_out = |request: &crate::completion::CompletionRequest| -> Vec<String> {
        let kind = EffectKind::Completion {
            request: request.clone().into(),
            stream: true,
        };
        handler
            .check(&kind)
            .iter()
            .map(|refusal| refusal.option.to_string())
            .collect()
    };
    assert_eq!(left_out(&request), ["top_p", "temperature"]);
    // The next call, a message longer, is answered as the last was; one
    // that changes an option is checked again.
    let next = request
        .clone()
        .message(crate::completion::Message::user("and then?"));
    assert_eq!(left_out(&next), ["top_p", "temperature"]);
    assert_eq!(left_out(&next.temperature(None)), ["top_p"]);
    assert!(http.requests().is_empty(), "a check sends nothing");
    let call = EffectKind::ToolCall {
        name: "lookup".to_owned(),
        args: "{}".to_owned(),
    };
    assert!(handler.check(&call).is_empty());
}

/// A tool's handler, and a wrapper around it, says its key without building
/// its descriptor, and the key is the descriptor's.
#[test]
fn a_tool_handler_and_its_wrappers_have_their_descriptors_key() {
    let tool = ErasedHandler::new(crate::serve::adapters::ToolAdapter::new(
        crate::tool::builtin::ThinkTool,
    ));
    let mapped = tool.clone().map_stream("same", |_| |item| vec![item]);
    for handler in [tool, mapped] {
        assert_eq!(Serve::key(&handler), handler.descriptor().key);
        assert_eq!(handler.key(), &crate::effect::tool_key("think"));
    }
}
