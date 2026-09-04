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
    serve::{OutcomeSink, Serve, ServingPolicy},
    streaming::StreamFinal,
};
use rig_ecs::bus::{
    BusPlugin, EffectOutcome, Handlers, InFlight, PendingEffect, Streamed, run_to_quiescence,
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

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                self.served.set(self.served.get() + 1);
                let response = CompletionResponse::new(
                    vec![AssistantContent::text("hello from the browser")],
                    Usage::new(),
                    "browser",
                );
                sink.resolve(Ok(Outcome::Completion(response))).await;
            }
            EffectKind::Completion { stream: true, .. } => {
                let mut out = sink.writer();
                loop {
                    if out.text("tick ").await.is_err() {
                        return;
                    }
                    if self.sends.fetch_add(1, Ordering::SeqCst) + 1 >= self.cap {
                        break;
                    }
                }
                let _ = out.finish(StreamFinal::new("browser", Usage::new())).await;
            }
            other => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("cannot serve {}", other.name()),
                )))
                .await;
            }
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
    }
}

fn app() -> App {
    // The plugin initialises the IO pool it spawns on; the test ticks the
    // pools the way `bevy_app`'s `TaskPoolPlugin` would, which needs the
    // other two initialised as well.
    bevy_tasks::ComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let mut app = App::new();
    app.add_plugins(
        BusPlugin::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
    );
    app.finish();
    app.cleanup();
    app
}

/// One pass of the plugin's runner, then a yield so the single-threaded
/// pool's tasks advance: on wasm the executor is ticked between frames.
async fn tick(app: &mut App) {
    run_to_quiescence(app.world_mut());
    bevy_tasks::futures_lite::future::yield_now().await;
    bevy_tasks::tick_global_task_pools_on_main_thread();
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
