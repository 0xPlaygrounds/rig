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
    serve::{Dispatch, Reply, Serve, ServingPolicy},
    streaming::{Item, Relayed, StreamEvent, Transcript},
};
use rig_ecs::{
    bus::{BusPlugin, EffectOutcome, PendingEffect},
    checkpoint::{Checkpoint, save_world},
};

/// A hang is a failure, never a wait.
pub const GUARD: Duration = Duration::from_secs(10);

/// The world's checkpoint, taken through its wire form.
pub fn checkpoint(app: &mut App) -> Checkpoint {
    let saved = save_world(app.world_mut()).expect("the world saves");
    Checkpoint::from_json(&saved.to_json().expect("serde")).expect("serde")
}

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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                self.counters.unary_started.fetch_add(1, Ordering::SeqCst);
                self.counters.hold.wait().await;
                self.counters.unary_served.fetch_add(1, Ordering::SeqCst);
                Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
                    vec![AssistantContent::text(&self.text)],
                    Usage::default(),
                    "mock",
                    serde_json::json!({ "provider": "mock" }),
                ))))
            }
            EffectKind::Completion { stream: true, .. } => {
                let counters = self.counters.clone();
                let cap = self.cap;
                Reply::written(move |mut out| async move {
                    let mut guard = StreamGuard {
                        counters: counters.clone(),
                        finished: false,
                    };
                    loop {
                        counters.hold.wait().await;
                        if out.text("tick ").await.is_err() {
                            return;
                        }
                        let sent = counters.stream_sends.fetch_add(1, Ordering::SeqCst) + 1;
                        if sent >= cap {
                            out.raw(serde_json::json!({ "provider": "mock" }));
                            guard.finished = out
                                .finish(
                                    "mock",
                                    rig_core::operation::Finish {
                                        usage: Usage::default(),
                                        ..rig_core::operation::Finish::default()
                                    },
                                )
                                .await
                                .is_ok();
                            return;
                        }
                    }
                })
            }
            other => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!("mock model cannot serve {}", other.name()),
            ))),
        }
    }
}

struct StreamGuard {
    counters: Arc<Counters>,
    finished: bool,
}
impl Drop for StreamGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.counters
                .stream_cancelled
                .fetch_add(1, Ordering::SeqCst);
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

/// An app with the bus installed under `policy`, ambiguity detection at
/// error level, the runner in `Update`, and the crate's types registered so
/// a checkpoint covers the bus.
pub fn app_with(policy: ServingPolicy) -> App {
    let mut app = App::new();
    app.add_plugins(BusPlugin::with_policy(policy).ambiguity_detection(LogLevel::Error));
    app.add_plugins(rig_cassette::ecs::ReplayPlugin);
    rig_ecs::checkpoint::register_types(app.world_mut());
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

pub use crate::run_support::register;

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

/// An app serving `model` with a scripted handler, and its counters: the
/// prologue of most bus suites.
pub fn served() -> (App, Entity, std::sync::Arc<Counters>) {
    let counters = std::sync::Arc::new(Counters::default());
    let mut app = app();
    let model = register(&mut app, "model", MockModel::new(&counters));
    (app, model, counters)
}

/// Spawn a unary completion effect on `model`'s key and tick until it is
/// answered. Returns the effect entity.
pub fn answered(app: &mut App, what: &str) -> Entity {
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(app, what, |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    effect
}

/// A relayed stream item, as a handler's stream carries it.
pub type Relay = Result<Relayed, ErrorReport>;

/// The items of a stream whose events are `events`, in order. A stream's
/// parts are the driver's to number, so a scripted one is read back from
/// its transcript.
pub fn relayed(events: serde_json::Value) -> Vec<Relay> {
    Transcript::parse_prefix(events)
        .expect("a stream in order")
        .into_items()
        .into_iter()
        .map(|item| Ok(Relayed::Item(item)))
        .collect()
}

/// A stream event of part `part`, as its JSON.
pub fn event(value: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"item": "event", "value": value})
}

/// The opening of a text part at `part`.
pub fn start_text(part: u32) -> serde_json::Value {
    event(serde_json::json!({"event": "start", "part": part, "kind": "text"}))
}

/// A text fragment of part `part`.
pub fn text_event(part: u32, text: &str) -> serde_json::Value {
    event(serde_json::json!({"event": "text", "part": part, "text": text}))
}

/// The one text part `text`: its start, its fragment, its end.
pub fn text_part(part: u32, text: &str) -> Vec<serde_json::Value> {
    vec![
        start_text(part),
        text_event(part, text),
        event(serde_json::json!({
            "event": "end",
            "part": part,
            "content": AssistantContent::text(text),
        })),
    ]
}

/// The text fragment `text` of part 0, cut from a stream that opened it.
pub fn text_item(text: &str) -> Relay {
    relayed(serde_json::json!([start_text(0), text_event(0, text)]))
        .pop()
        .expect("the fragment")
}

/// The opening of text part 0.
pub fn text_start_item() -> Relay {
    relayed(serde_json::json!([start_text(0)]))
        .pop()
        .expect("the start")
}

/// The provider's end of a reply from `provider`, carrying no content.
pub fn done(provider: &str) -> Relay {
    Ok(Relayed::Done(Box::new(CompletionResponse::new(
        Vec::new(),
        Usage::default(),
        provider,
        serde_json::json!({}),
    ))))
}

/// Whether `item` is a text fragment.
pub fn is_text(item: &Relay) -> Option<&str> {
    match item {
        Ok(Relayed::Item(Item::Event(StreamEvent::Text { text, .. }))) => Some(text),
        _ => None,
    }
}
