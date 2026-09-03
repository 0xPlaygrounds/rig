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
use crate::agent::sync::Mutex;

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
                // The last poll, as `Drop for Driven` does it.
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
