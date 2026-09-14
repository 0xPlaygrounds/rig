use super::*;
use futures::{FutureExt, Stream};
use rig::{
    error::{ErrorKind, ErrorReport},
    streaming::{BlockId, BlockKind},
};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    task::{Context, Poll},
};

struct Tracked {
    items: VecDeque<Result<StreamEvent, ErrorReport>>,
    polls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}

impl Stream for Tracked {
    type Item = Result<StreamEvent, ErrorReport>;
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

fn items() -> Vec<Result<StreamEvent, ErrorReport>> {
    let id = BlockId::Wire("real-call".into());
    vec![
        Ok(StreamEvent::BlockStart {
            id: id.clone(),
            kind: BlockKind::ToolCall,
        }),
        Err(ErrorReport::new(ErrorKind::Provider, "preserved error")),
        Ok(StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::ToolName {
                name: "tool".into(),
            },
        }),
        Ok(StreamEvent::BlockDelta {
            id,
            delta: Delta::ToolArguments {
                arguments: "{}".into(),
            },
        }),
    ]
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
    assert_eq!(
        serde_json::to_value(observed).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
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
        Arc::new(Semaphore::new(0)),
    );
    for _ in 0..3 {
        stream.next().await.expect("prefix item").ok();
    }
    assert!(stream.next().now_or_never().is_none());
    drop(stream);
    assert!(dropped.load(Ordering::SeqCst));
}
