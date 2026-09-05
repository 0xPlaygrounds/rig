//! Shared by the `run_*` suites: a request-capturing model, a tool handler
//! that is never called, an app with both plugins, and the tick guard.

#![allow(dead_code, reason = "each suite uses the part of the support it needs")]

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::{CompletionRequest, CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{OutcomeSink, Serve, ServingPolicy},
};
use rig_ecs::{
    agent::{
        AdditionalParams, DefaultMaxTurns, InvalidCalls, MaxTokens, MaxTurns, Output, Owner,
        Preamble, Temperature, ToolChoiceSpec, UsesModel,
    },
    bus::{BusPlugin, Handlers},
    systems::AgentPlugin,
};

pub const GUARD: Duration = Duration::from_secs(10);

/// A model that keeps every request it is asked and answers a fixed text.
pub struct Capturing {
    pub label: String,
    pub requests: Arc<Mutex<Vec<CompletionRequest>>>,
    pub answer: String,
}

impl Capturing {
    pub fn new(label: &str, answer: &str) -> (Self, Arc<Mutex<Vec<CompletionRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                label: label.to_owned(),
                requests: Arc::clone(&requests),
                answer: answer.to_owned(),
            },
            requests,
        )
    }
}

impl Serve for Capturing {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new(self.label.as_str()),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { request, .. } => {
                self.requests.lock().expect("requests").push(request);
                let response = CompletionResponse::new(
                    vec![AssistantContent::text(&self.answer)],
                    Usage::new(),
                    "capturing",
                );
                sink.resolve(Ok(Outcome::Completion(response))).await;
            }
            other => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("a model cannot serve {}", other.name()),
                )))
                .await;
            }
        }
    }
}

/// A tool handler that is advertised and never called.
pub struct NeverCalled {
    pub name: String,
}

impl Serve for NeverCalled {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.name.as_str()),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: format!("the {} tool", self.name),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        sink.resolve(Err(ErrorReport::new(
            ErrorKind::Internal,
            "a tool advertised and never called was called",
        )))
        .await;
    }
}

/// An app with both plugins, ambiguity detection at error level.
pub fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        BusPlugin::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
    ));
    app.finish();
    app.cleanup();
    app
}

/// Register `handler` under `key` from outside a system.
pub fn register(app: &mut App, key: &str, handler: impl Serve + 'static) -> Entity {
    Handlers::with(app.world_mut(), |handlers| handlers.register(key, handler))
        .expect("the world has a bus")
        .expect("a fresh key")
}

/// An agent entity over `model`, with a preamble and defaults.
pub fn spawn_agent(world: &mut World, owner: &str, model: Entity) -> Entity {
    world
        .spawn((
            Owner(owner.to_owned()),
            Preamble(Some("You are terse.".to_owned())),
            Temperature(None),
            MaxTokens(None),
            AdditionalParams(None),
            ToolChoiceSpec(None),
            Output::default(),
            DefaultMaxTurns(None),
            MaxTurns(1),
            InvalidCalls::default(),
            UsesModel(model),
        ))
        .id()
}

/// Tick the app until `done` holds, or fail after [`GUARD`].
pub fn tick_until(app: &mut App, what: &str, mut done: impl FnMut(&mut World) -> bool) {
    let start = Instant::now();
    loop {
        app.update();
        if done(app.world_mut()) {
            return;
        }
        assert!(start.elapsed() < GUARD, "{what}: not done within {GUARD:?}");
        std::thread::yield_now();
    }
}

/// The text parts of a request's user messages, in order.
pub fn texts(request: &CompletionRequest) -> Vec<String> {
    request
        .chat_history
        .iter()
        .map(|message| match message {
            rig_core::message::Message::System { content } => format!("system:{content}"),
            rig_core::message::Message::User { content } => format!(
                "user:{}",
                content
                    .iter()
                    .filter_map(|part| match part {
                        rig_core::message::UserContent::Text(text) => Some(text.text.clone()),
                        rig_core::message::UserContent::ToolResult(_)
                        | rig_core::message::UserContent::Image(_)
                        | rig_core::message::UserContent::Audio(_)
                        | rig_core::message::UserContent::Video(_)
                        | rig_core::message::UserContent::Document(_) => None,
                    })
                    .collect::<String>()
            ),
            rig_core::message::Message::Assistant { content, .. } => format!(
                "assistant:{}",
                content
                    .iter()
                    .filter_map(|part| match part {
                        AssistantContent::Text(text) => Some(text.text.clone()),
                        AssistantContent::ToolCall(_)
                        | AssistantContent::Reasoning(_)
                        | AssistantContent::Image(_) => None,
                    })
                    .collect::<String>()
            ),
        })
        .collect()
}

/// A completion model that never answers: the dispatch stays in flight
/// for as long as the world lives.
pub struct NeverAnswers {
    pub label: String,
}

impl Serve for NeverAnswers {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new(self.label.as_str()),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        futures::future::pending::<()>().await;
        drop(sink);
    }
}

/// A model that answers a script: one assistant turn per request, in
/// order, then a fixed text.
pub struct Scripted {
    pub label: String,
    pub turns: Mutex<std::collections::VecDeque<Vec<AssistantContent>>>,
    pub requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl Scripted {
    pub fn new(
        label: &str,
        turns: Vec<Vec<AssistantContent>>,
    ) -> (Self, Arc<Mutex<Vec<CompletionRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                label: label.to_owned(),
                turns: Mutex::new(turns.into()),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }
}

impl Serve for Scripted {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new(self.label.as_str()),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { request, .. } => {
                self.requests.lock().expect("requests").push(request);
                let choice = self
                    .turns
                    .lock()
                    .expect("turns")
                    .pop_front()
                    .unwrap_or_else(|| vec![AssistantContent::text("done")]);
                let response = CompletionResponse::new(choice, Usage::new(), "scripted");
                sink.resolve(Ok(Outcome::Completion(response))).await;
            }
            other => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("a model cannot serve {}", other.name()),
                )))
                .await;
            }
        }
    }
}

/// A tool call the model makes, as the script's assistant part.
pub fn call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::tool_call(id, name, arguments)
}

/// A tool that adds `x` and `y`, counting how many calls were in flight
/// at once.
pub struct Adder {
    pub name: String,
    pub in_flight: Arc<std::sync::atomic::AtomicUsize>,
    pub peak: Arc<std::sync::atomic::AtomicUsize>,
    pub hold: Option<Arc<Mutex<Option<futures::channel::oneshot::Receiver<()>>>>>,
}

impl Adder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            in_flight: Arc::default(),
            peak: Arc::default(),
            hold: None,
        }
    }
}

impl Serve for Adder {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.name.as_str()),
            family: FamilyDescriptor::Tool {
                name: "add".to_owned(),
                description: "adds x and y".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        use std::sync::atomic::Ordering;
        let EffectKind::ToolCall { args, .. } = kind else {
            sink.resolve(Err(ErrorReport::new(ErrorKind::Request, "a tool call")))
                .await;
            return;
        };
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let parsed: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
        let sum = parsed["x"].as_i64().unwrap_or(0) + parsed["y"].as_i64().unwrap_or(0);
        // Let a sibling start before answering, so concurrency shows.
        for _ in 0..3 {
            bevy_tasks::futures_lite::future::yield_now().await;
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        sink.resolve(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::json(
                serde_json::json!(sum),
            )),
        }))
        .await;
    }
}
