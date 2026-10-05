use std::{sync::Mutex, task::Context};

use futures::{StreamExt, executor::block_on, task::noop_waker_ref};

use super::*;
use crate::completion::CompletionResponse;
use crate::operation::Finish;
use crate::streaming::{Item, Part, Relayed, StreamEvent, UnknownPayload};

#[derive(Default)]
struct Seen {
    outcomes: Vec<Result<Outcome, ErrorReport>>,
    events: usize,
    discard_events: bool,
}

struct Observer(Arc<Mutex<Seen>>);

impl Observe for Observer {
    fn origin(&mut self, _origin: &crate::message::Origin) {}
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.0.lock().expect("seen").outcomes.push(outcome.clone());
    }

    fn keep_events(&self) -> bool {
        !self.0.lock().expect("seen").discard_events
    }

    fn event(&mut self, _: &Item<StreamEvent>) {
        self.0.lock().expect("seen").events += 1;
    }

    fn discard(&mut self, _: &str) {}
    fn patch(&mut self, _: &EffectKind) {}
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
        Ok(Relayed::Done(Box::new(CompletionResponse::new(
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
            Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
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
            ))))
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
    Ok(Relayed::Done(Box::new(CompletionResponse::new(
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
