use super::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Waker},
};

#[test]
fn cancelled_prefix_stops_before_fallback_and_retains_ownership() {
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    for count in [0, 1, 3] {
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let counter = polls.clone();
        let owned = Dropped(dropped.clone());
        let source = Box::pin(futures::stream::poll_fn(move |cx| {
            let _owned = &owned;
            let poll = counter.fetch_add(1, Ordering::SeqCst);
            if poll == 0 {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            assert!(
                poll <= count,
                "the cancelled prefix must not poll the synthesized fallback"
            );
            Poll::Ready(Some(Ok(StreamEvent::Unknown(
                rig_core::streaming::UnknownPayload::new(serde_json::Value::Null),
            ))))
        }));
        let mut limited = cancelled_prefix(source, count);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(limited.as_mut().poll_next(&mut cx).is_pending());
        for _ in 0..count {
            assert!(matches!(
                limited.as_mut().poll_next(&mut cx),
                Poll::Ready(Some(Ok(_)))
            ));
        }
        for _ in 0..3 {
            assert!(limited.as_mut().poll_next(&mut cx).is_pending());
        }
        assert_eq!(
            polls.load(Ordering::SeqCst),
            if count == 0 { 0 } else { count + 1 }
        );
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(limited);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
