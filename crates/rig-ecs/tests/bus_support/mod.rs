//! Shared by the `bus_*` suites: a scripted model, an app with the plugin,
//! and a wall-clock tick guard. Nothing agent-shaped.

#![allow(dead_code, reason = "each suite uses the part of the support it needs")]

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
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
use rig_ecs::bus::{BusPlugin, Handlers};

/// A hang is a failure, never a wait.
pub const GUARD: Duration = Duration::from_secs(10);

/// What a scripted model observed.
#[derive(Default)]
pub struct Counters {
    /// Unary dispatches the handler entered.
    pub unary_started: AtomicUsize,
    /// Unary dispatches the handler answered.
    pub unary_served: AtomicUsize,
    /// Streams that saw their consumer go.
    pub stream_cancelled: AtomicUsize,
    /// Stream deltas sent.
    pub stream_sends: AtomicUsize,
    /// While set, a dispatch stays in flight inside the handler, parked (not
    /// spinning) until released.
    pub hold: Hold,
}

/// A gate a handler parks on: `hold()` closes it, `release()` opens it and
/// wakes every parked future.
#[derive(Default)]
pub struct Hold {
    closed: AtomicBool,
    wakers: std::sync::Mutex<Vec<std::task::Waker>>,
}

impl Hold {
    pub fn hold(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn release(&self) {
        self.closed.store(false, Ordering::SeqCst);
        for waker in self.wakers.lock().expect("wakers").drain(..) {
            waker.wake();
        }
    }

    pub fn is_held(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Park until released; resolves at once when open.
    pub async fn wait(&self) {
        std::future::poll_fn(|cx| {
            if !self.is_held() {
                return std::task::Poll::Ready(());
            }
            self.wakers.lock().expect("wakers").push(cx.waker().clone());
            if !self.is_held() {
                return std::task::Poll::Ready(());
            }
            std::task::Poll::Pending
        })
        .await;
    }
}

/// Deltas one stream emits before its terminal record.
pub const STREAM_CAP: usize = 200;

/// A scripted completion handler: answers unary dispatches with a fixed
/// text once the hold is released, streams one text delta per poll until
/// the cap or the consumer goes, and counts what it observed.
pub struct MockModel {
    pub counters: Arc<Counters>,
    /// The text a unary answer carries.
    pub text: String,
    /// Deltas a stream emits before its terminal record.
    pub cap: usize,
}

impl MockModel {
    pub fn new(counters: &Arc<Counters>) -> Self {
        Self {
            counters: Arc::clone(counters),
            text: "hello from the world".to_owned(),
            cap: STREAM_CAP,
        }
    }

    pub fn saying(counters: &Arc<Counters>, text: &str) -> Self {
        Self {
            text: text.to_owned(),
            ..Self::new(counters)
        }
    }

    /// A model whose stream never ends on its own.
    pub fn endless(counters: &Arc<Counters>) -> Self {
        Self {
            cap: usize::MAX,
            ..Self::new(counters)
        }
    }
}

impl Serve for MockModel {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("mock"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                self.counters.unary_started.fetch_add(1, Ordering::SeqCst);
                self.counters.hold.wait().await;
                self.counters.unary_served.fetch_add(1, Ordering::SeqCst);
                let response = CompletionResponse::new(
                    vec![AssistantContent::text(&self.text)],
                    Usage::new(),
                    "mock",
                );
                sink.resolve(Ok(Outcome::Completion(response))).await;
            }
            EffectKind::Completion { stream: true, .. } => {
                let mut out = sink.writer();
                loop {
                    self.counters.hold.wait().await;
                    if out.text("tick ").await.is_err() {
                        self.counters
                            .stream_cancelled
                            .fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                    let sent = self.counters.stream_sends.fetch_add(1, Ordering::SeqCst) + 1;
                    if sent >= self.cap {
                        let _ = out.finish(StreamFinal::new("mock", Usage::new())).await;
                        return;
                    }
                }
            }
            other => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("mock model cannot serve {}", other.name()),
                )))
                .await;
            }
        }
    }
}

/// A completion request with one user message.
pub fn request() -> CompletionRequest {
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

/// A unary completion effect.
pub fn completion() -> EffectKind {
    EffectKind::Completion {
        request: request(),
        stream: false,
    }
}

/// A streaming completion effect.
pub fn streaming() -> EffectKind {
    EffectKind::Completion {
        request: request(),
        stream: true,
    }
}

/// An app with the plugin under `policy`, ambiguity detection at error
/// level, plugins finished.
pub fn app_with(policy: ServingPolicy) -> App {
    let mut app = App::new();
    app.add_plugins(BusPlugin::with_policy(policy).ambiguity_detection(LogLevel::Error));
    app.finish();
    app.cleanup();
    app
}

/// [`app_with`] under the default policy.
pub fn app() -> App {
    app_with(ServingPolicy::default())
}

/// [`app_with`] under serial serving.
pub fn serial_app() -> App {
    app_with(ServingPolicy {
        serial_per_handler: true,
        ..ServingPolicy::default()
    })
}

/// Register `handler` under `key` from outside a system.
pub fn register(app: &mut App, key: &str, handler: impl Serve + 'static) -> Entity {
    Handlers::with(app.world_mut(), |handlers| handlers.register(key, handler))
        .expect("the world has a bus")
        .expect("a fresh key")
}

/// Tick the app until `done` holds, or fail after [`GUARD`]. Returns the
/// ticks taken.
pub fn tick_until(app: &mut App, what: &str, mut done: impl FnMut(&mut World) -> bool) -> usize {
    let start = Instant::now();
    let mut ticks = 0;
    loop {
        app.update();
        ticks += 1;
        if done(app.world_mut()) {
            return ticks;
        }
        assert!(
            start.elapsed() < GUARD,
            "{what}: not done after {ticks} ticks and {:?}",
            start.elapsed()
        );
        std::thread::yield_now();
    }
}

/// Tick the app `n` times.
pub fn tick(app: &mut App, n: usize) {
    for _ in 0..n {
        app.update();
    }
}

/// The text of a unary completion outcome.
pub fn text_of(outcome: &Result<Outcome, ErrorReport>) -> String {
    match outcome {
        Ok(Outcome::Completion(response)) => response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.clone()),
                AssistantContent::Reasoning(_)
                | AssistantContent::Image(_)
                | AssistantContent::ToolCall(_) => None,
            })
            .collect(),
        other => panic!("not a completion: {other:?}"),
    }
}
