//! The bus's wait/wake protocols under `loom`: every interleaving of a
//! small model, run with `RUSTFLAGS="--cfg rig_loom" cargo test -p rig-agent
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

use rig_core::serve::{ErasedHandler, Serve};

use super::{
    Bus,
    dispatcher::{Command, Enqueue, Reply, Shared},
    registrar::{Mailbox, Registration},
};
use rig_core::effect::{EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey};

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

type Receiver = oneshot::Receiver<Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>>;

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
            parent: None,
            scope: None,
            context: None,
            published: None,
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
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy {
            command_capacity: 4,
            ..rig_core::serve::ServingPolicy::default()
        }));
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
                let parked = std::sync::Arc::new(futures::task::AtomicWaker::new());
                match shared.enqueue(cmd, &parked, &cx) {
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
                    assert_eq!(report.kind, rig_core::error::ErrorKind::BusClosed);
                }
                other => panic!("the buffered command was not failed: {other:?}"),
            }
        }
    });
}

struct Nothing;

impl Serve for Nothing {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom { kind: "m".into() },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _sink: rig_core::serve::OutcomeSink) {}
}

struct Tagged(&'static str);

impl Serve for Tagged {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom {
                kind: self.0.into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: rig_core::serve::OutcomeSink) {
        sink.resolve(Ok(rig_core::effect::Outcome::Custom {
            payload: serde_json::json!(self.0),
        }))
        .await;
    }
}

#[test]
fn loom_concurrent_registrations_publish_and_serve_the_same_handler() {
    use futures::FutureExt;

    loom::model(|| {
        let (dispatcher, registrar, mut driver) = Bus::channel();
        let other = registrar.clone();
        let first = thread::spawn(move || registrar.register("k", Tagged("first")));
        let second = thread::spawn(move || other.register("k", Tagged("second")));
        first.join().unwrap().expect("registered first");
        second.join().unwrap().expect("registered second");

        let key = HandlerKey::from("k");
        let described = dispatcher.descriptor(&key).expect("published");
        let FamilyDescriptor::Custom { kind } = described.family else {
            panic!("custom handler");
        };
        let mut pending = dispatcher.dispatch(
            &key,
            EffectKind::Custom {
                kind: kind.clone().into(),
                payload: serde_json::Value::Null,
            },
        );
        let (_flag, waker) = recording();
        let mut cx = Context::from_waker(&waker);
        assert!(pending.poll_unpin(&mut cx).is_pending());
        let _ = driver.poll_unpin(&mut cx);
        let std::task::Poll::Ready(Ok(rig_core::effect::Outcome::Custom { payload })) =
            pending.poll_unpin(&mut cx)
        else {
            panic!("registered handler must answer");
        };
        assert_eq!(payload, serde_json::json!(kind));
    });
}

/// A registration posted before a dispatch (program order on the
/// registering thread) is installed before that dispatch is served: the
/// driver takes the queue first and the mailbox second.
#[test]
fn loom_a_registration_before_a_dispatch_is_installed_first() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy {
            command_capacity: 4,
            ..rig_core::serve::ServingPolicy::default()
        }));
        let mailbox = Arc::new(Mailbox::new());
        let poster = {
            let shared = Arc::clone(&shared);
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || {
                mailbox
                    .register(&shared, HandlerKey::from("k"), ErasedHandler::new(Nothing))
                    .expect("register");
                let (cmd, receiver) = command(1);
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let parked = std::sync::Arc::new(futures::task::AtomicWaker::new());
                assert!(matches!(shared.enqueue(cmd, &parked, &cx), Enqueue::Sent));
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
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy {
            command_capacity: 1,
            ..rig_core::serve::ServingPolicy::default()
        }));
        let mut senders = Vec::new();
        for id in 1..=2u64 {
            let shared = Arc::clone(&shared);
            senders.push(thread::spawn(move || {
                let (mut cmd, _receiver) = command(id);
                let (flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let slot = std::sync::Arc::new(futures::task::AtomicWaker::new());
                let mut parked = false;
                loop {
                    match shared.enqueue(cmd, &slot, &cx) {
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

/// Counts polls of the handler and records opened by the driver: the two
/// must agree however a consumer's drop interleaves with the driver's poll.
#[derive(Clone)]
struct Counted {
    polled: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Serve for Counted {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("k"),
            family: FamilyDescriptor::Custom { kind: "m".into() },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: rig_core::serve::OutcomeSink) {
        self.polled.fetch_add(1, StdOrdering::SeqCst);
        // One yield before answering, the shape of any handler that does IO.
        let mut yielded = false;
        futures::future::poll_fn(|cx| {
            if yielded {
                std::task::Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
        })
        .await;
        sink.resolve(Ok(rig_core::effect::Outcome::Custom {
            payload: serde_json::Value::Null,
        }))
        .await;
    }
}

/// Records opened, and outcomes that were failures.
#[derive(Clone)]
struct Begun {
    begun: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    failed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl rig_core::serve::Recorder for Begun {
    fn tool_output(&self, _: EffectId, _: rig_core::tool::ToolResultContext) {}
    fn handlers(&self, _handlers: Vec<HandlerDescriptor>) {}
    fn begin(
        &self,
        _id: EffectId,
        _key: HandlerKey,
        _kind: EffectKind,
        _origin: rig_core::serve::Origin,
    ) {
        self.begun.fetch_add(1, StdOrdering::SeqCst);
    }
    fn discard(&self, _id: EffectId) {}
    fn patch(&self, _id: EffectId, _kind: EffectKind) {}
    fn keep_events(&self) -> bool {
        false
    }
    fn event(&self, _id: EffectId, _event: &rig_core::streaming::StreamEvent) {}
    fn resolve(
        &self,
        _id: EffectId,
        outcome: Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) {
        if outcome.is_err() {
            self.failed.fetch_add(1, StdOrdering::SeqCst);
        }
    }
}

/// A consumer dropping its dispatch racing the driver's poll: whichever
/// order the model picks, a record is opened only for a dispatch whose
/// handler was polled, no record ever resolves as the "handler dropped its
/// sink" failure (before the fix, a cancel that landed before the driver's
/// poll opened a record and then failed it that way), and the driver ends
/// with nothing in flight.
#[test]
fn loom_cancel_before_serve_never_polls_the_handler() {
    use futures::FutureExt;
    loom::model(|| {
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        let polled = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        driver
            .register(
                "k",
                Counted {
                    polled: polled.clone(),
                },
            )
            .expect("register");
        let begun = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        driver.record_to(Begun {
            begun: begun.clone(),
            failed: failed.clone(),
        });
        let mut pending = dispatcher.dispatch(
            &HandlerKey::from("k"),
            EffectKind::Custom {
                kind: std::sync::Arc::from("m"),
                payload: serde_json::Value::Null,
            },
        );
        {
            let (_flag, waker) = recording();
            let mut cx = Context::from_waker(&waker);
            assert!(pending.poll_unpin(&mut cx).is_pending(), "sent");
        }
        let consumer = thread::spawn(move || drop(pending));
        let driving = thread::spawn(move || {
            let (_flag, waker) = recording();
            let mut cx = Context::from_waker(&waker);
            for _ in 0..3 {
                let _ = driver.poll_unpin(&mut cx);
                thread::yield_now();
            }
            driver
        });
        consumer.join().unwrap();
        let mut driver = driving.join().unwrap();
        let (_flag, waker) = recording();
        let mut cx = Context::from_waker(&waker);
        let _ = driver.poll_unpin(&mut cx);
        assert_eq!(
            polled.load(StdOrdering::SeqCst),
            begun.load(StdOrdering::SeqCst),
            "a record without a handler poll, or a poll without a record"
        );
        assert!(polled.load(StdOrdering::SeqCst) <= 1);
        assert_eq!(
            failed.load(StdOrdering::SeqCst),
            0,
            "a cancelled dispatch was served and recorded as a handler failure"
        );
        assert_eq!(driver.in_flight(), 0);
        assert_eq!(dispatcher.buffered(), 0);
    });
}

/// The close for commands races a late enqueue from a `Pending` that
/// outlived its dispatcher: every interleaving ends with the command
/// either buffered on a bus still open for commands (the driver's next
/// poll takes it) or refused as `BusClosed` — never buffered on a bus
/// closed for commands, which nothing would ever take. The decision and
/// the store are one critical section with the enqueue's check.
#[test]
fn loom_close_for_commands_never_strands_a_late_enqueue() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy {
            command_capacity: 4,
            ..rig_core::serve::ServingPolicy::default()
        }));
        // No dispatcher is open: the sender is a `Pending` that outlived its.
        let sender = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let (cmd, _receiver) = command(1);
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let parked = std::sync::Arc::new(futures::task::AtomicWaker::new());
                match shared.enqueue(cmd, &parked, &cx) {
                    Enqueue::Sent => false,
                    Enqueue::Closed => true,
                    Enqueue::Parked(_) | Enqueue::Refused(_) => panic!("neither"),
                }
            })
        };
        let closer = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || shared.try_close_commands())
        };
        let refused = sender.join().unwrap();
        let closed = closer.join().unwrap();
        assert!(
            !(shared.commands_closed() && shared.buffered() > 0),
            "a command is buffered on a bus closed for commands (refused: {refused}, closed: {closed})"
        );
        if closed {
            assert!(
                refused,
                "the close saw an empty buffer, so the send came after it"
            );
        }
    });
}

/// Causality: a nested dispatch under serial serving is refused by its
/// chain, whichever thread enqueues it. A dispatch is in flight on `k`; a
/// command made from it, to `k`, is enqueued from a spawned thread while
/// another thread begins and ends an unrelated dispatch. Every interleaving
/// refuses the nested command — the old thread-id rule accepted it (and
/// hung) whenever the enqueuing thread was not the polling one.
#[test]
fn loom_a_nested_serial_dispatch_is_refused_from_any_thread() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy {
            command_capacity: 4,
            serial_per_handler: true,
            ..rig_core::serve::ServingPolicy::default()
        }));
        shared.dispatcher_opened();
        let _outer = shared
            .begin_in_flight(EffectId::from_raw(1), HandlerKey::from("k"), None)
            .ok()
            .expect("nothing cancelled");
        let nested = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let (mut cmd, _receiver) = command(2);
                cmd.parent = Some(EffectId::from_raw(1));
                let (_flag, waker) = recording();
                let cx = Context::from_waker(&waker);
                let parked = std::sync::Arc::new(futures::task::AtomicWaker::new());
                matches!(shared.enqueue(cmd, &parked, &cx), Enqueue::Refused(_))
            })
        };
        let sibling = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let _flag = shared
                    .begin_in_flight(EffectId::from_raw(3), HandlerKey::from("j"), None)
                    .ok()
                    .expect("nothing cancelled");
                shared.end_in_flight(EffectId::from_raw(3));
            })
        };
        let refused = nested.join().unwrap();
        sibling.join().unwrap();
        assert!(refused, "the nested dispatch queued behind its ancestor");
        assert_eq!(shared.buffered(), 0);
    });
}

/// Causality: a parent's cancel reaches a child that begins concurrently.
/// One thread cancels the descendants of dispatch 1; another begins
/// dispatch 2 as 1's child. Whichever order the model picks, the child is
/// either refused at `begin_in_flight` (the cancel came first) or flagged
/// (it was in flight when the cancel scanned) — never in flight unflagged,
/// which is what a separate check-then-insert would allow.
#[test]
fn loom_a_parent_cancel_reaches_a_child_that_begins_meanwhile() {
    loom::model(|| {
        let shared = Arc::new(Shared::new(rig_core::serve::ServingPolicy::default()));
        let _parent = shared
            .begin_in_flight(EffectId::from_raw(1), HandlerKey::from("k"), None)
            .ok()
            .expect("nothing cancelled");
        let cancelling = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || shared.cancel_descendants(EffectId::from_raw(1)))
        };
        let beginning = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                shared
                    .begin_in_flight(
                        EffectId::from_raw(2),
                        HandlerKey::from("j"),
                        Some(EffectId::from_raw(1)),
                    )
                    .ok()
            })
        };
        cancelling.join().unwrap();
        let child = beginning.join().unwrap();
        match child {
            None => {}
            Some(flag) => assert!(
                flag.is_set(),
                "a child in flight escaped its parent's cancel"
            ),
        }
    });
}
