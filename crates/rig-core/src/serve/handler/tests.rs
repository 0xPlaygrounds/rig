use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use futures::{FutureExt, StreamExt, executor::block_on, task::noop_waker_ref};

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

    fn discard(&mut self) {}
    fn patch(&mut self, _: &EffectKind) {}
}

#[test]
fn resolved_stream_preserves_original_response_for_outcome_only_replay() {
    use crate::message::{AssistantContent, DocumentSourceKind, Image};

    for keep_events in [false, true] {
        let (events, receiver) = mpsc::channel(16);
        let seen = Arc::new(Mutex::new(Seen {
            discard_events: !keep_events,
            ..Seen::default()
        }));
        let sink = OutcomeSink::stream(EffectId::from_raw(1), events)
            .with_observer(Box::new(Observer(seen.clone())));
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
        block_on(sink.resolve(Ok(Outcome::Completion(response))));
        let delivered = block_on(receiver.collect::<Vec<_>>());
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
fn ancestor_cancelled_detached_unary_resolve_records_and_delivers_cancellation() {
    let (reply, receiver) = oneshot::channel();
    let marker = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(Seen::default()));
    let sink = OutcomeSink::unary(EffectId::from_raw(1), reply)
        .with_cancel(marker.clone())
        .with_observer(Box::new(Observer(seen.clone())))
        .detach();
    marker.store(true, Ordering::SeqCst);
    assert!(sink.is_closed());
    block_on(sink.resolve(Ok(Outcome::Custom {
        payload: serde_json::json!("late success"),
    })));
    let report = block_on(receiver)
        .expect("answered")
        .expect_err("ancestor cancelled");
    assert_eq!(report.kind, ErrorKind::Cancelled);
    let seen = seen.lock().expect("seen");
    assert_eq!(seen.outcomes.len(), 1);
    assert_eq!(
        seen.outcomes[0]
            .as_ref()
            .expect_err("recorded cancellation")
            .kind,
        ErrorKind::Cancelled
    );
}

#[test]
fn ancestor_cancelled_detached_stream_rejects_and_does_not_record_late_events() {
    let (events, mut receiver) = mpsc::channel(4);
    let marker = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut sink = OutcomeSink::stream(EffectId::from_raw(1), events)
        .with_cancel(marker.clone())
        .with_observer(Box::new(Observer(seen.clone())))
        .detach();
    marker.store(true, Ordering::SeqCst);
    assert!(sink.is_closed());
    let result = block_on(sink.send(Ok(StreamEvent::Final(StreamFinal::new(
        "test",
        Default::default(),
    )))));
    drop(sink);
    assert_eq!(result, Err(SinkClosed));
    let report = block_on(receiver.next())
        .expect("cancellation item")
        .expect_err("cancelled");
    assert_eq!(report.kind, ErrorKind::Cancelled);
    assert!(block_on(receiver.next()).is_none());
    let seen = seen.lock().expect("seen");
    assert_eq!(
        seen.events, 0,
        "a rejected Final is not evidence of completion"
    );
    assert_eq!(seen.outcomes.len(), 1);
    assert_eq!(
        seen.outcomes[0]
            .as_ref()
            .expect_err("recorded cancellation")
            .kind,
        ErrorKind::Cancelled
    );
}

#[test]
fn ancestor_cancelled_sink_is_not_ready_for_more_output() {
    let (events, _receiver) = mpsc::channel(4);
    let marker = Arc::new(AtomicBool::new(true));
    let mut sink = OutcomeSink::stream(EffectId::from_raw(1), events).with_cancel(marker);
    let mut cx = Context::from_waker(noop_waker_ref());
    assert_eq!(sink.poll_ready(&mut cx), Poll::Ready(Err(SinkClosed)));
}

struct FullStream {
    sink: OutcomeSink,
    receiver: mpsc::Receiver<Result<StreamEvent, ErrorReport>>,
    marker: Arc<AtomicBool>,
    seen: Arc<Mutex<Seen>>,
}

impl FullStream {
    fn new() -> Self {
        let (mut events, receiver) = mpsc::channel(0);
        let (prefix_events, mut prefix_receiver) = mpsc::channel(4);
        let mut writer = OutcomeSink::stream(EffectId::from_raw(1), prefix_events).writer();
        block_on(writer.text("prefix")).expect("writer creates the prefix");
        let prefix = block_on(prefix_receiver.next()).expect("text block start");
        events
            .try_send(prefix)
            .expect("fill the sender's reserved slot");
        let marker = Arc::new(AtomicBool::new(false));
        let seen = Arc::new(Mutex::new(Seen::default()));
        let sink = OutcomeSink::stream(EffectId::from_raw(1), events)
            .with_cancel(marker.clone())
            .with_observer(Box::new(Observer(seen.clone())));
        Self {
            sink,
            receiver,
            marker,
            seen,
        }
    }
}

#[test]
fn backpressured_send_rechecks_cancellation_before_publishing_final() {
    let FullStream {
        mut sink,
        mut receiver,
        marker,
        seen,
    } = FullStream::new();
    let mut cx = Context::from_waker(noop_waker_ref());
    let mut send = Box::pin(sink.send(Ok(StreamEvent::Final(StreamFinal::new(
        "test",
        Default::default(),
    )))));
    assert!(send.poll_unpin(&mut cx).is_pending());
    assert!(
        seen.lock().expect("seen").outcomes.is_empty(),
        "unsent Final must not record success"
    );
    marker.store(true, Ordering::SeqCst);
    assert!(block_on(receiver.next()).expect("prefix").is_ok());
    assert_eq!(send.poll_unpin(&mut cx), Poll::Ready(Err(SinkClosed)));
    drop(send);
    drop(sink);
    let report = block_on(receiver.next())
        .expect("cancellation")
        .expect_err("no late Final");
    assert_eq!(report.kind, ErrorKind::Cancelled);
    assert!(block_on(receiver.next()).is_none());
}

#[test]
fn backpressured_resolve_rechecks_cancellation_before_publishing_completion() {
    let FullStream {
        sink,
        mut receiver,
        marker,
        seen,
    } = FullStream::new();
    let mut cx = Context::from_waker(noop_waker_ref());
    let response = CompletionResponse::new(
        vec![crate::message::AssistantContent::text("late response")],
        Default::default(),
        "test",
    );
    let mut resolve = sink.resolve(Ok(Outcome::Completion(response)));
    assert!(resolve.poll_unpin(&mut cx).is_pending());
    assert!(
        seen.lock().expect("seen").outcomes.is_empty(),
        "unsent response must not record success"
    );
    marker.store(true, Ordering::SeqCst);
    assert!(block_on(receiver.next()).expect("prefix").is_ok());
    assert!(resolve.poll_unpin(&mut cx).is_ready());
    let report = block_on(receiver.next())
        .expect("cancellation")
        .expect_err("no late response");
    assert_eq!(report.kind, ErrorKind::Cancelled);
    assert_eq!(
        seen.lock().expect("seen").outcomes[0]
            .as_ref()
            .expect_err("recorded cancellation")
            .kind,
        ErrorKind::Cancelled
    );
}

#[test]
fn cancellation_terminal_survives_a_full_stream_buffer() {
    let FullStream {
        sink,
        mut receiver,
        marker,
        seen,
    } = FullStream::new();
    marker.store(true, Ordering::SeqCst);
    drop(sink);
    assert!(block_on(receiver.next()).expect("prefix").is_ok());
    let report = block_on(receiver.next())
        .expect("cancellation terminal after buffered prefix")
        .expect_err("cancelled");
    assert_eq!(report.kind, ErrorKind::Cancelled);
    assert!(block_on(receiver.next()).is_none());
    assert_eq!(
        seen.lock().expect("seen").outcomes[0]
            .as_ref()
            .expect_err("recorded cancellation")
            .kind,
        ErrorKind::Cancelled
    );
}
