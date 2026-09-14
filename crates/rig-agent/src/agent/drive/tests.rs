//! The driver protocol between runs: whoever is awaiting drives, and a run
//! that stops driving hands the bus to the runs still waiting on it.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures::{StreamExt, task::noop_waker_ref};

use super::*;
use crate::bus::Bus;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    serve::{Dispatch, Reply, Serve},
};

/// A waker that counts its wakes: another run's, registered on the bus.
struct Counting(AtomicUsize);

impl Wake for Counting {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn custom() -> EffectKind {
    EffectKind::Custom {
        kind: Arc::from("test"),
        payload: serde_json::Value::Null,
    }
}

/// Answers at once.
struct Handler;

impl Serve for Handler {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Ok(Outcome::Custom {
            payload: serde_json::Value::Null,
        }))
    }
}

/// An owned agent bus serving `k`, and another run's waker registered on
/// it — a run that found the driver lock taken and is waiting to drive.
fn bus_with_a_waiting_run() -> (AgentBus, Arc<Counting>) {
    let policy = ServingPolicy::default();
    let (dispatcher, registrar, mut driver) = Bus::channel_with(policy);
    driver.register("k", Handler).expect("a fresh key");
    let bus = AgentBus::owned(dispatcher, registrar, driver, "owner".to_owned(), policy);
    let waiting = Arc::new(Counting(AtomicUsize::new(0)));
    let slot = bus.wakers.slot();
    bus.wakers
        .register(slot, &Waker::from(Arc::clone(&waiting)));
    (bus, waiting)
}

/// A run that finished drove the bus last. Another run that registered
/// its waker and found the lock taken may have buffered a command since
/// the finished run's last drain; nothing else polls the driver between
/// runs, so the finished run must wake the runs still registered, which
/// then take the lock and drive. (The obligation the loom model of the
/// protocol assumes of a run's end.) Exactly one wake, at the end: the
/// polls that serve the run's own dispatch wake nobody.
#[test]
fn a_finished_run_wakes_the_runs_that_registered_while_it_drove() {
    let (bus, waiting) = bus_with_a_waiting_run();
    let key = HandlerKey::from("k");
    let mut run = bus.drive(futures::stream::once(
        bus.dispatcher().dispatch(&key, custom()),
    ));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(
        matches!(run.poll_next_unpin(&mut cx), Poll::Ready(Some(Ok(_)))),
        "the run served its own dispatch while it drove"
    );
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        0,
        "serving the run's own dispatch woke nobody"
    );
    assert!(matches!(run.poll_next_unpin(&mut cx), Poll::Ready(None)));
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        1,
        "the run that stopped driving woke the runs still registered, once"
    );
}

/// The same obligation for a run dropped while live: its answer landed
/// but its stream was never polled to its end, so `Drop` gives the driver
/// its last poll (nothing in flight to settle) and wakes the runs still
/// registered to take over the bus. Exactly one wake, from the drop.
#[test]
fn a_dropped_run_wakes_the_runs_that_registered_while_it_drove() {
    let (bus, waiting) = bus_with_a_waiting_run();
    let key = HandlerKey::from("k");
    let mut run = bus.drive(futures::stream::once(
        bus.dispatcher().dispatch(&key, custom()),
    ));
    let mut cx = Context::from_waker(noop_waker_ref());
    assert!(matches!(
        run.poll_next_unpin(&mut cx),
        Poll::Ready(Some(Ok(_)))
    ));
    assert_eq!(waiting.0.load(Ordering::SeqCst), 0);
    drop(run);
    assert_eq!(
        waiting.0.load(Ordering::SeqCst),
        1,
        "the dropped run woke the runs still registered, once"
    );
}
