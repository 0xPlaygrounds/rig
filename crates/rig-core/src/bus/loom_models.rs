//! The bus's wait/wake protocols under `loom`: every interleaving of a
//! small model, run with `RUSTFLAGS="--cfg rig_loom" cargo test -p rig-core
//! --lib --release loom_`. A model that fails is a bug in the protocol.
//!
//! `Waker`s are `std`'s (loom does not intercept `wake()`), so the models
//! observe wakes through a recording `Wake` impl; the reply halves are
//! `futures` channels, treated as data. The first two models are the
//! regression proofs for the races `fix(core)` closed in #2443 (a send after
//! the close; a registration between the driver's two drains): each fails
//! against the pre-fix code.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{
    sync::atomic::{AtomicBool, Ordering as StdOrdering},
    task::{Context, Wake, Waker},
};

use futures::channel::oneshot;
use loom::{sync::Arc, thread};

use super::{
    Bus, ErasedHandler, Serve,
    dispatcher::{Command, Enqueue, Reply, Shared},
    registrar::{Mailbox, Registration},
};
use crate::effect::{EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey};

/// A waker that records that it was woken.
struct Recording(AtomicBool);

impl Wake for Recording {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.store(true, StdOrdering::SeqCst);
    }
}

fn recording() -> (std::sync::Arc<Recording>, Waker) {
    let flag = std::sync::Arc::new(Recording(AtomicBool::new(false)));
    let waker = Waker::from(std::sync::Arc::clone(&flag));
    (flag, waker)
}

type Receiver = oneshot::Receiver<Result<crate::effect::Outcome, crate::error::ErrorReport>>;

fn command(id: u64) -> (Box<Command>, Receiver) {
    let (reply, receiver) = oneshot::channel();
    let (cancel_guard, cancel) = oneshot::channel();
    std::mem::forget(cancel_guard);
    (
        Box::new(Command {
            id: EffectId::from_raw(id),
            key: HandlerKey::from("k"),
            kind: EffectKind::Custom {
                kind: std::sync::Arc::from("m"),
                payload: serde_json::Value::Null,
            },
            reply: Reply::Unary(reply),
            span: tracing::Span::none(),
            cancel,
        }),
        receiver,
    )
}

/// A sender racing the driver's drop: every command is either taken by the
/// driver or failed with `BusClosed` — never buffered forever.
#[test]
fn loom_close_fails_what_the_driver_never_took() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(4, false));
        let sender = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let (cmd, receiver) = command(1);
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                // `Pending::poll`'s send stage.
                if shared.is_closed() {
                    return (true, receiver);
                }
                match shared.enqueue(cmd, &cx) {
                    Enqueue::Sent => (false, receiver),
                    Enqueue::Closed => (true, receiver),
                    Enqueue::Parked(_) | Enqueue::Refused(_) => panic!("neither"),
                }
            })
        };
        let closer = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || shared.mark_closed())
        };
        let (saw_closed, mut receiver) = sender.join().unwrap();
        closer.join().unwrap();
        assert_eq!(
            shared.buffered(),
            0,
            "a command was buffered after the close"
        );
        if !saw_closed {
            match receiver.try_recv() {
                Ok(Some(Err(report))) => {
                    assert_eq!(report.kind, crate::error::ErrorKind::BusClosed);
                }
                other => panic!("the buffered command was not failed: {other:?}"),
            }
        }
    });
}

struct Nothing;

impl Serve for Nothing {
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom { kind: "m".into() },
        }
    }

    async fn serve(&self, _kind: EffectKind, _sink: super::OutcomeSink) {}
}

/// A registration posted before a dispatch (program order on the
/// registering thread) is installed before that dispatch is served: the
/// driver takes the queue first and the mailbox second.
#[test]
fn loom_a_registration_before_a_dispatch_is_installed_first() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(4, false));
        let mailbox = Arc::new(Mailbox::new());
        let poster = {
            let shared = Arc::clone(&shared);
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || {
                mailbox.post(Registration::Install {
                    key: HandlerKey::from("k"),
                    handler: ErasedHandler::new(Nothing),
                });
                let (cmd, receiver) = command(1);
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                assert!(matches!(shared.enqueue(cmd, &cx), Enqueue::Sent));
                receiver
            })
        };
        let driver = {
            let shared = Arc::clone(&shared);
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || {
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let mut installed = false;
                let mut served = 0;
                while served < 1 {
                    // The driver's poll: the queue, then the mailbox, then
                    // serving what was taken.
                    let commands = shared.drain(&cx);
                    for registration in mailbox.drain(&cx) {
                        if let Registration::Install { .. } = registration {
                            installed = true;
                        }
                    }
                    for _command in commands {
                        assert!(installed, "a dispatch was served before its registration");
                        served += 1;
                    }
                    if served < 1 {
                        thread::yield_now();
                    }
                }
            })
        };
        let _receiver = poster.join().unwrap();
        driver.join().unwrap();
    });
}

/// Two senders against one driver with `command_capacity: 1`: the buffer
/// never holds more than one command, every parked sender is woken before
/// the model ends, and both commands are drained exactly once.
#[test]
fn loom_bound_is_never_exceeded_and_no_sender_is_lost() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(1, false));
        let mut senders = Vec::new();
        for id in 1..=2u64 {
            let shared = Arc::clone(&shared);
            senders.push(thread::spawn(move || {
                let (mut cmd, _receiver) = command(id);
                let (flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let mut parked = false;
                loop {
                    match shared.enqueue(cmd, &cx) {
                        Enqueue::Sent => break,
                        Enqueue::Parked(kept) => {
                            parked = true;
                            cmd = kept;
                            thread::yield_now();
                        }
                        Enqueue::Refused(_) | Enqueue::Closed => panic!("neither"),
                    }
                }
                (parked, flag)
            }));
        }
        let drained = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let mut drained = 0;
                while drained < 2 {
                    let batch = shared.drain(&cx);
                    assert!(batch.len() <= 1, "the bound is bus-wide: {}", batch.len());
                    drained += batch.len();
                    if drained < 2 {
                        thread::yield_now();
                    }
                }
                drained
            })
        };
        let results: Vec<_> = senders.into_iter().map(|s| s.join().unwrap()).collect();
        assert_eq!(drained.join().unwrap(), 2);
        for (parked, flag) in results {
            if parked {
                assert!(
                    flag.0.load(StdOrdering::SeqCst),
                    "a parked sender was woken by a drain"
                );
            }
        }
        assert_eq!(shared.buffered(), 0);
    });
}

/// A registration posted after the driver dropped keeps nothing: the
/// descriptor is data and stays, the handler goes.
#[test]
fn loom_registrar_after_close_keeps_nothing() {
    loom::model(|| {
        let (_dispatcher, registrar, driver) = Bus::channel();
        let poster = thread::spawn(move || {
            registrar
                .register("late", Nothing)
                .expect("descriptor published");
            registrar
        });
        drop(driver);
        let registrar = poster.join().unwrap();
        assert!(registrar.is_closed());
    });
}
