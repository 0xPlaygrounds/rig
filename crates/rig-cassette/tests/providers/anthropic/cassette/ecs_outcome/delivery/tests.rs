use super::*;
use futures::{FutureExt, Stream};
use rig::{
    completion::{CompletionResponse, Usage},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, ToolCall, ToolFunction, ToolName},
    streaming::Transcript,
};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    task::{Context, Poll},
};

struct Tracked {
    items: VecDeque<Result<Relayed, ErrorReport>>,
    polls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}

impl Stream for Tracked {
    type Item = Result<Relayed, ErrorReport>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Poll::Ready(self.items.pop_front())
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

/// `events` as a stream relays them, with `errors` at their positions.
fn relayed(
    events: serde_json::Value,
    errors: &[(usize, &str)],
) -> Vec<Result<Relayed, ErrorReport>> {
    let mut items: Vec<_> = Transcript::parse_prefix(events)
        .expect("a stream in order")
        .into_items()
        .into_iter()
        .map(|item| Ok(Relayed::Item(item)))
        .collect();
    for (at, message) in errors {
        items.insert(*at, Err(ErrorReport::new(ErrorKind::Provider, *message)));
    }
    items
}

fn event(value: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"item": "event", "value": value})
}

/// A tool call, with an error after its start.
fn items() -> Vec<Result<Relayed, ErrorReport>> {
    let call = ToolCall::from_wire(
        "real-call",
        ToolFunction::new(
            ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    relayed(
        serde_json::json!([
            event(serde_json::json!({"event": "start", "part": 0, "kind": "tool_call"})),
            event(serde_json::json!({"event": "arguments", "part": 0, "json": "{}"})),
            event(serde_json::json!({
                "event": "end",
                "part": 0,
                "content": AssistantContent::ToolCall(call),
            })),
        ]),
        &[(1, "preserved error")],
    )
}

#[tokio::test]
async fn boundary_pauses_before_polling_and_release_preserves_every_item() {
    let expected = items();
    let polls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let release = Arc::new(Semaphore::new(0));
    let mut stream = gate_events(
        Box::pin(Tracked {
            items: expected.clone().into(),
            polls: polls.clone(),
            dropped: dropped.clone(),
        }),
        DeltaBoundary::Tool,
        release.clone(),
    );
    let mut observed = Vec::new();
    for _ in 0..3 {
        observed.push(stream.next().await.expect("prefix item"));
    }
    assert!(
        stream.next().now_or_never().is_none(),
        "no EOF or later item at the boundary"
    );
    assert_eq!(
        polls.load(Ordering::SeqCst),
        3,
        "no polling beyond the delivered delta"
    );
    assert!(
        !dropped.load(Ordering::SeqCst),
        "paused provider is still owned"
    );
    release.add_permits(1);
    observed.extend(stream.collect::<Vec<_>>().await);
    assert_eq!(observed, expected);
    assert!(dropped.load(Ordering::SeqCst));
}

/// An always-ready upstream exposes the gate's boundary without transport timing.
#[tokio::test]
async fn text_boundary_pauses_before_polling_and_release_preserves_every_item() {
    let call = ToolCall::from_wire(
        "real-call",
        ToolFunction::new(
            ToolName::new("add").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let mut expected = relayed(
        serde_json::json!([
            event(serde_json::json!({"event": "start", "part": 0, "kind": "tool_call"})),
            event(serde_json::json!({"event": "arguments", "part": 0, "json": "{}"})),
            event(serde_json::json!({
                "event": "end",
                "part": 0,
                "content": AssistantContent::ToolCall(call),
            })),
            event(serde_json::json!({"event": "start", "part": 1, "kind": "text"})),
            event(serde_json::json!({"event": "text", "part": 1, "text": "first"})),
            event(serde_json::json!({"event": "text", "part": 1, "text": " second"})),
            event(serde_json::json!({
                "event": "end",
                "part": 1,
                "content": AssistantContent::text("first second"),
            })),
        ]),
        &[(1, "preserved error"), (6, "error after text")],
    );
    expected.push(Ok(Relayed::Done(Box::new({
        let mut response = CompletionResponse::new(
            Vec::new(),
            Usage::default(),
            "anthropic",
            serde_json::json!({"stop_reason": "end_turn"}),
        );
        response.message_id = Some("message".into());
        response
    }))));
    let polls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let release = Arc::new(Semaphore::new(0));
    let mut stream = gate_events(
        Box::pin(Tracked {
            items: expected.clone().into(),
            polls: polls.clone(),
            dropped: dropped.clone(),
        }),
        DeltaBoundary::Text,
        release.clone(),
    );
    let mut observed = Vec::new();
    for _ in 0..6 {
        observed.push(
            stream
                .next()
                .now_or_never()
                .expect("tool deltas do not pause the text gate")
                .expect("prefix through the first text delta"),
        );
    }
    assert!(
        stream.next().now_or_never().is_none(),
        "no EOF or later item at the text boundary"
    );
    assert_eq!(
        polls.load(Ordering::SeqCst),
        6,
        "no upstream poll after the first text delta"
    );
    assert!(
        !dropped.load(Ordering::SeqCst),
        "paused provider is still owned"
    );
    release.add_permits(1);
    observed.extend(
        stream
            .collect::<Vec<_>>()
            .now_or_never()
            .expect("one release drains all remaining items without another pause"),
    );
    assert_eq!(observed, expected);
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn cancelling_a_paused_stream_drops_the_provider() {
    let dropped = Arc::new(AtomicBool::new(false));
    let mut stream = gate_events(
        Box::pin(Tracked {
            items: items().into(),
            polls: Arc::default(),
            dropped: dropped.clone(),
        }),
        DeltaBoundary::Tool,
        Arc::new(Semaphore::new(0)),
    );
    for _ in 0..3 {
        stream.next().await.expect("prefix item").ok();
    }
    assert!(stream.next().now_or_never().is_none());
    drop(stream);
    assert!(dropped.load(Ordering::SeqCst));
}
