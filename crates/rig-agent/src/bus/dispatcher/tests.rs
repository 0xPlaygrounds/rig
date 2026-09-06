use super::*;
use futures::{
    FutureExt,
    task::{ArcWake, waker},
};

struct QueueProbe {
    shared: std::sync::Weak<Shared>,
    called: AtomicBool,
    unlocked: AtomicBool,
}
impl ArcWake for QueueProbe {
    fn wake_by_ref(this: &Arc<Self>) {
        this.called.store(true, Ordering::SeqCst);
        let shared = this.shared.upgrade().expect("live dispatcher");
        this.unlocked
            .store(shared.queue.try_lock().is_ok(), Ordering::SeqCst);
    }
}

/// Reentrant executors can inspect the dispatcher synchronously from wake.
#[test]
fn enqueue_wakes_the_driver_after_releasing_the_queue_lock() {
    let (dispatcher, _registrar, mut driver) = super::super::Bus::channel();
    let probe = Arc::new(QueueProbe {
        shared: Arc::downgrade(&dispatcher.shared),
        called: AtomicBool::new(false),
        unlocked: AtomicBool::new(false),
    });
    let wake = waker(probe.clone());
    let mut cx = Context::from_waker(&wake);
    assert!(driver.poll_unpin(&mut cx).is_pending());
    let mut pending = dispatcher.dispatch(
        &HandlerKey::from("custom"),
        EffectKind::Custom {
            kind: "test".into(),
            payload: serde_json::json!(null),
        },
    );
    assert!(pending.poll_unpin(&mut cx).is_pending());
    assert!(probe.called.load(Ordering::SeqCst));
    assert!(
        probe.unlocked.load(Ordering::SeqCst),
        "driver wake may call Dispatcher::buffered without deadlocking"
    );
}
