//! The agent's driver protocol under `loom`: the per-poll driver lock and
//! the bus-wide waker set. Run with `RUSTFLAGS="--cfg rig_loom" cargo test
//! -p rig-agent --lib --release loom_`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{
    sync::atomic::{AtomicUsize, Ordering as StdOrdering},
    task::{Wake, Waker},
};

use loom::{sync::Arc, thread};

use super::{WakerSet, try_lock};
use crate::sync::Mutex;

/// A waker that counts its wakes.
struct Counting(AtomicUsize);

impl Wake for Counting {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.fetch_add(1, StdOrdering::SeqCst);
    }
}

fn counting() -> (std::sync::Arc<Counting>, Waker) {
    let counter = std::sync::Arc::new(Counting(AtomicUsize::new(0)));
    let waker = Waker::from(std::sync::Arc::clone(&counter));
    (counter, waker)
}

/// Two runs on one bus: run A holds the driver lock for one poll while run
/// B, registered in the waker set, finds the lock taken; the driver's
/// progress under A's polls wakes B in every schedule, and B takes the
/// lock afterwards.
#[test]
fn loom_a_driven_run_that_finds_the_lock_taken_is_woken() {
    loom::model(|| {
        let wakers = std::sync::Arc::new(WakerSet::default());
        let lock = Arc::new(Mutex::new(0u32));
        let (a_flag, a_waker) = counting();
        let (b_flag, b_waker) = counting();
        let a_slot = wakers.slot();
        let b_slot = wakers.slot();
        wakers.register(a_slot, &a_waker);
        wakers.register(b_slot, &b_waker);

        let run_a = {
            let wakers = std::sync::Arc::clone(&wakers);
            let lock = Arc::clone(&lock);
            thread::spawn(move || {
                if let Some(guard) = try_lock(&lock) {
                    // Driver progress under A's poll wakes every registered run.
                    std::sync::Arc::clone(&wakers).wake_by_ref();
                    drop(guard);
                }
                thread::yield_now();
                // The run's end, as `Driven::finish` and `Drop for Driven`
                // do it: the runs still registered are woken to take over.
                std::sync::Arc::clone(&wakers).wake_by_ref();
            })
        };
        let run_b = {
            let lock = Arc::clone(&lock);
            thread::spawn(move || {
                loop {
                    match try_lock(&lock) {
                        Some(_guard) => break,
                        None => thread::yield_now(),
                    }
                }
            })
        };
        run_a.join().unwrap();
        run_b.join().unwrap();
        assert!(a_flag.0.load(StdOrdering::SeqCst) >= 1);
        assert!(
            b_flag.0.load(StdOrdering::SeqCst) >= 1,
            "a run that registered its waker is woken by the driving run"
        );
    });
}

/// A run that drops mid-flight unregisters its slot and wakes the survivors
/// with the bus-wide waker: the live run's waker fires, in every schedule.
#[test]
fn loom_a_dropped_run_wakes_the_survivors() {
    loom::model(|| {
        let wakers = std::sync::Arc::new(WakerSet::default());
        let (_a_flag, a_waker) = counting();
        let (b_flag, b_waker) = counting();
        let a_slot = wakers.slot();
        let b_slot = wakers.slot();
        wakers.register(b_slot, &b_waker);
        let dropper = {
            let wakers = std::sync::Arc::clone(&wakers);
            thread::spawn(move || {
                wakers.register(a_slot, &a_waker);
                wakers.unregister(a_slot);
                std::sync::Arc::clone(&wakers).wake_by_ref();
            })
        };
        let survivor = {
            let wakers = std::sync::Arc::clone(&wakers);
            thread::spawn(move || {
                wakers.register(b_slot, &b_waker);
            })
        };
        dropper.join().unwrap();
        survivor.join().unwrap();
        assert!(
            b_flag.0.load(StdOrdering::SeqCst) >= 1,
            "the survivor was woken"
        );
    });
}

/// Answers at once.
struct Instant;

impl rig_core::serve::Serve for Instant {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from("k"),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: "test".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        return rig_core::serve::Reply::Outcome(Ok(rig_core::effect::Outcome::Custom {
            payload: serde_json::Value::Null,
        }));
    }
}

/// The runtime itself, not a model of it: two runs on one agent bus, each
/// awaiting one dispatch, polled from two threads through [`Driven`]. A
/// run that finds the driver lock taken yields and must be woken to take
/// over — by driver progress under the other run's poll, or by that run's
/// end after its last drain. Whichever schedule the model picks, both runs
/// complete: no run is left with its command buffered and nobody driving.
/// (Fails against a `finish` that unregisters without waking: the run that
/// lost the lock race under the finishing run's last poll is never polled
/// again.)
#[test]
fn loom_two_contending_runs_both_complete() {
    use futures::StreamExt;
    use std::task::{Context, Poll};

    loom::model(|| {
        let (dispatcher, registrar, mut driver) = crate::bus::Bus::channel();
        driver.register("k", Instant).expect("a fresh key");
        let bus = std::sync::Arc::new(super::AgentBus::owned(
            dispatcher,
            registrar,
            driver,
            "owner".to_owned(),
            rig_core::serve::ServingPolicy::default(),
        ));
        let runs: Vec<_> = (0..2)
            .map(|_| {
                let bus = std::sync::Arc::clone(&bus);
                thread::spawn(move || {
                    let (wakes, waker) = counting();
                    let mut cx = Context::from_waker(&waker);
                    let key = rig_core::effect::HandlerKey::from("k");
                    let mut run = bus.drive(futures::stream::once(bus.dispatcher().dispatch(
                        &key,
                        rig_core::effect::EffectKind::Custom {
                            kind: std::sync::Arc::from("test"),
                            payload: serde_json::Value::Null,
                        },
                    )));
                    let mut answered = false;
                    let mut seen_wakes = 0;
                    // A run's executor: poll, and poll again only when woken.
                    for _ in 0..32 {
                        match run.poll_next_unpin(&mut cx) {
                            Poll::Ready(Some(outcome)) => answered = outcome.is_ok(),
                            Poll::Ready(None) => return answered,
                            Poll::Pending => {
                                let mut waited = 0;
                                while wakes.0.load(StdOrdering::SeqCst) == seen_wakes {
                                    waited += 1;
                                    assert!(
                                        waited < 64,
                                        "a run pending on the bus was never woken"
                                    );
                                    thread::yield_now();
                                }
                                seen_wakes = wakes.0.load(StdOrdering::SeqCst);
                            }
                        }
                    }
                    panic!("a run did not complete within its poll budget");
                })
            })
            .collect();
        for run in runs {
            assert!(run.join().unwrap(), "every run's dispatch was answered");
        }
    });
}
