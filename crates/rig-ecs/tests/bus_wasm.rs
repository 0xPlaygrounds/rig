//! The bus module executed on `wasm32-unknown-unknown`, once: every other
//! wasm claim about the crate is `cargo check`. A bare `World` and the
//! plugin's schedule, driven by the test — `bevy_app`'s runner on wasm is
//! frame-scheduled by the browser, so the app runner is not what a test
//! ticks — with a scripted handler on the single-threaded pool; a unary
//! effect resolves, a stream accumulates, a despawn cancels, and the
//! components a system holds are the same `Send + Sync` types as natively.
//!
//! Run with `cargo test -p rig-ecs --target wasm32-unknown-unknown --test
//! bus_wasm` under `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner`
//! (the CLI at the workspace lock file's `wasm-bindgen` version), as
//! rig-bus's `tests/wasm.rs` is.

#![cfg(target_arch = "wasm32")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::{
        CompletionRequest, CompletionResponse, Message, ModelRef, ProviderCapabilities, Usage,
    },
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{Dispatch, Reply, Serve, ServingPolicy},
    streaming::StreamFinal,
};
use rig_ecs::bus::{
    Bus, EffectOutcome, Handlers, InFlight, PendingEffect, Streamed, run_to_quiescence,
};
use wasm_bindgen_test::wasm_bindgen_test;

/// A `!Send` handler, honestly: an `Rc` counter, as a browser provider
/// client would hold `!Send` state.
struct BrowserModel {
    served: Rc<Cell<usize>>,
    sends: Arc<AtomicUsize>,
    /// Deltas before the terminal record; `usize::MAX` streams until the
    /// consumer goes.
    cap: usize,
}

impl Serve for BrowserModel {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("browser"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                self.served.set(self.served.get() + 1);
                let response = CompletionResponse::new(
                    vec![AssistantContent::text("hello from the browser")],
                    Usage::new(),
                    "browser",
                );
                Reply::Outcome(Ok(Outcome::Completion(response)))
            }
            EffectKind::Completion { stream: true, .. } => {
                let sends = self.sends.clone();
                let cap = self.cap;
                let local = Rc::clone(&self.served);
                Reply::written(move |mut out| async move {
                    let _local = local; // The returned stream itself is !Send.
                    loop {
                        if out.text("tick ").await.is_err() {
                            return;
                        }
                        if sends.fetch_add(1, Ordering::SeqCst) + 1 >= cap {
                            break;
                        }
                    }
                    let _ = out.finish(StreamFinal::new("browser", Usage::new())).await;
                })
            }
            other => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!("cannot serve {}", other.name()),
            ))),
        }
    }
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
        observation: None,
    }
}

fn app() -> App {
    let mut app = App::new();
    Bus::with_policy(ServingPolicy::default())
        .ambiguity_detection(LogLevel::Error)
        .install(app.world_mut());
    app.finish();
    app.cleanup();
    app
}

/// One host pass, then let the browser run its queued executor microtasks.
async fn tick(app: &mut App) {
    run_to_quiescence(app.world_mut());
    rig_core::wasm_compat::sleep(std::time::Duration::from_millis(1)).await;
}

#[wasm_bindgen_test]
async fn a_unary_effect_resolves_on_the_browser_pool() {
    let served = Rc::new(Cell::new(0));
    let sends = Arc::new(AtomicUsize::new(0));
    let mut app = app();
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register(
                "model",
                BrowserModel {
                    served: Rc::clone(&served),
                    sends: Arc::clone(&sends),
                    cap: 5,
                },
            )
            .expect("a fresh key")
    })
    .expect("a bus");
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        ))
        .id();
    for _ in 0..100 {
        tick(&mut app).await;
        if app.world().get::<EffectOutcome>(effect).is_some() {
            break;
        }
    }
    let outcome = app
        .world()
        .get::<EffectOutcome>(effect)
        .expect("answered within a hundred ticks");
    assert!(outcome.0.is_ok(), "{:?}", outcome.0);
    assert_eq!(served.get(), 1);
    assert!(app.world().get::<InFlight>(effect).is_none());
}

#[wasm_bindgen_test]
async fn a_stream_accumulates_and_a_despawn_cancels() {
    let served = Rc::new(Cell::new(0));
    let sends = Arc::new(AtomicUsize::new(0));
    let mut app = app();
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register(
                "model",
                BrowserModel {
                    served: Rc::clone(&served),
                    sends: Arc::clone(&sends),
                    cap: 5,
                },
            )
            .expect("a fresh key")
    })
    .expect("a bus");
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(
            "model",
            EffectKind::Completion {
                request: request(),
                stream: true,
            },
        ))
        .id();
    for _ in 0..100 {
        tick(&mut app).await;
        if app.world().get::<EffectOutcome>(effect).is_some() {
            break;
        }
    }
    let streamed = app.world().get::<Streamed>(effect).expect("folded");
    assert_eq!(streamed.text, "tick tick tick tick tick ");
    assert!(streamed.outcome.as_ref().is_some_and(Result::is_ok));

    // An endless stream on its own key: it cannot finish, so it is in
    // flight until despawned, and its sends stop where they were.
    let endless_sends = Arc::new(AtomicUsize::new(0));
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register(
                "endless",
                BrowserModel {
                    served: Rc::clone(&served),
                    sends: Arc::clone(&endless_sends),
                    cap: usize::MAX,
                },
            )
            .expect("a fresh key")
    })
    .expect("a bus");
    let second = app
        .world_mut()
        .spawn(PendingEffect::new(
            "endless",
            EffectKind::Completion {
                request: request(),
                stream: true,
            },
        ))
        .id();
    for _ in 0..3 {
        tick(&mut app).await;
    }
    assert!(
        app.world().get::<InFlight>(second).is_some(),
        "an endless stream stays in flight"
    );
    assert!(
        app.world()
            .get::<Streamed>(second)
            .is_some_and(|streamed| !streamed.events.is_empty()),
        "and lands per tick"
    );
    app.world_mut().despawn(second);
    tick(&mut app).await;
    let sent = endless_sends.load(Ordering::SeqCst);
    for _ in 0..10 {
        tick(&mut app).await;
    }
    assert_eq!(
        endless_sends.load(Ordering::SeqCst),
        sent,
        "nothing sent after the despawn"
    );
}

#[wasm_bindgen_test]
fn the_components_are_send_sync_on_wasm_too() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<PendingEffect>();
    assert_send_sync::<EffectOutcome>();
    assert_send_sync::<Streamed>();
    assert_send_sync::<InFlight>();
}

#[wasm_bindgen_test]
async fn local_streams_drop_on_marker_removal_scheduled_despawn_replacement_and_shutdown() {
    use rig_ecs::bus::effect::{Executions, Streaming};
    struct Local(Rc<Cell<usize>>);
    impl Drop for Local {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let drops = Rc::new(Cell::new(0));
    let stream = || {
        let local = Local(drops.clone());
        Box::pin(futures::stream::poll_fn(move |_| {
            let _local = &local;
            std::task::Poll::Pending
        })) as rig_core::streaming::StreamEvents
    };
    let mut world = World::new();
    Bus::default().install(&mut world);
    async fn dropped(drops: &Cell<usize>, expected: usize) {
        for _ in 0..1000 {
            if drops.get() == expected {
                return;
            }
            // Yield to the browser event loop, not only this Rust test task.
            rig_core::wasm_compat::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert_eq!(
            drops.get(),
            expected,
            "cancelled local worker was not dropped"
        );
    }
    let (streaming, task) = Streaming::spawn(stream(), 1);
    let entity = world
        .spawn((
            InFlight {
                key: "local".into(),
            },
            streaming,
        ))
        .id();
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    let (streaming, task) = Streaming::spawn(stream(), 1);
    world.entity_mut(entity).insert(streaming);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    dropped(&drops, 1).await;
    world.entity_mut(entity).remove::<InFlight>();
    dropped(&drops, 2).await;
    let (streaming, task) = Streaming::spawn(stream(), 1);
    world.entity_mut(entity).insert((
        InFlight {
            key: "local".into(),
        },
        streaming,
    ));
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    let mut schedule = Schedule::default();
    schedule.add_systems(move |mut commands: Commands| {
        commands.entity(entity).despawn();
    });
    schedule.run(&mut world);
    dropped(&drops, 3).await;
    assert!(world.non_send::<Executions>().streams.is_empty());
    let (streaming, task) = Streaming::spawn(stream(), 1);
    let entity = world
        .spawn((
            InFlight {
                key: "local".into(),
            },
            streaming,
        ))
        .id();
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    drop(world);
    dropped(&drops, 4).await;
}

#[wasm_bindgen_test]
fn a_local_writer_keeps_post_final_work_alive_until_resume_or_cancellation() {
    for resume in [false, true] {
        let local = Rc::new(Cell::new(false));
        let finished = local.clone();
        let (release, wait) = futures::channel::oneshot::channel::<()>();
        let mut stream = Reply::written(move |writer| async move {
            writer
                .finish(StreamFinal::new("local", Usage::new()))
                .await
                .unwrap();
            wait.await.unwrap();
            finished.set(true);
        })
        .into_stream();
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(matches!(
            stream.as_mut().poll_next(&mut cx),
            std::task::Poll::Ready(Some(Ok(rig_core::streaming::StreamEvent::Final(_))))
        ));
        assert!(stream.as_mut().poll_next(&mut cx).is_pending());
        if resume {
            release.send(()).unwrap();
            assert!(matches!(
                stream.as_mut().poll_next(&mut cx),
                std::task::Poll::Ready(None)
            ));
            assert!(local.get());
        } else {
            drop(stream);
            assert!(release.send(()).is_err());
            assert!(!local.get());
            assert_eq!(Rc::strong_count(&local), 1);
        }
    }
}

#[wasm_bindgen_test]
async fn cancellation_reaches_setup_unary_fold_idle_and_full_queue_without_host_ticks() {
    #[derive(Clone, Copy)]
    enum Stage {
        Setup,
        Unary,
        Idle,
        Full,
    }
    struct LocalDrop(Rc<Cell<bool>>);
    impl Drop for LocalDrop {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    struct Parked {
        stage: Stage,
        entered: Rc<Cell<bool>>,
        produced: Rc<Cell<usize>>,
        dropped: Rc<Cell<bool>>,
    }
    impl Serve for Parked {
        type Family = rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: "parked".into(),
                family: FamilyDescriptor::Completion {
                    model: ModelRef::new("local"),
                    capabilities: ProviderCapabilities::default(),
                },
                layers: Vec::new(),
            }
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            let owned = LocalDrop(self.dropped.clone());
            if matches!(self.stage, Stage::Setup) {
                self.entered.set(true);
                std::future::pending::<()>().await;
            }
            let entered = self.entered.clone();
            let produced = self.produced.clone();
            let full = matches!(self.stage, Stage::Full);
            Reply::Stream(Box::pin(futures::stream::poll_fn(move |_| {
                let _owned = &owned;
                entered.set(true);
                if full {
                    produced.set(produced.get() + 1);
                    std::task::Poll::Ready(Some(Ok(rig_core::streaming::StreamEvent::Unknown(
                        rig_core::streaming::UnknownPayload::new(serde_json::Value::Null),
                    ))))
                } else {
                    std::task::Poll::Pending
                }
            })))
        }
    }
    for stage in [Stage::Setup, Stage::Unary, Stage::Idle, Stage::Full] {
        let entered = Rc::new(Cell::new(false));
        let produced = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let mut app = app();
        app.world_mut()
            .resource_mut::<rig_ecs::bus::Policy>()
            .0
            .stream_capacity = 2;
        Handlers::with(app.world_mut(), |handlers| {
            handlers.register(
                "parked",
                Parked {
                    stage,
                    entered: entered.clone(),
                    produced: produced.clone(),
                    dropped: dropped.clone(),
                },
            )
        })
        .unwrap()
        .unwrap();
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(
                "parked",
                EffectKind::Completion {
                    request: request(),
                    stream: !matches!(stage, Stage::Unary),
                },
            ))
            .id();
        for _ in 0..100 {
            tick(&mut app).await;
            if entered.get() {
                break;
            }
        }
        assert!(entered.get(), "owned execution must begin");
        // No more host collection: the worker either parks in the source or
        // fills its actual private queue. Neither may need a later spawn to stop.
        rig_core::wasm_compat::sleep(std::time::Duration::from_millis(1)).await;
        if matches!(stage, Stage::Full) {
            let delivered = app.world().get::<Streamed>(effect).unwrap().events.len();
            assert_eq!(
                produced.get() - delivered,
                3,
                "two shared slots and one reserved slot"
            );
        }
        let before = produced.get();
        app.world_mut().despawn(effect);
        for _ in 0..100 {
            if dropped.get() {
                break;
            }
            rig_core::wasm_compat::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert!(
            dropped.get(),
            "cancellation must reach the pending future without a host tick"
        );
        assert_eq!(produced.get(), before, "cancelled work must not advance");
    }
}
