//! Shared by the `run_*` suites: a request-capturing model, a tool handler
//! that is never called, an app with both plugins, and the tick guard.

#![allow(dead_code, reason = "each suite uses the part of the support it needs")]

use rig_core::serve::Dispatch;
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
    message::{AssistantContent, Message},
    serve::{Serve, ServingPolicy},
};
use rig_ecs::{
    agent::{
        AdditionalParams, DefaultMaxTurns, Failed, InvalidCalls, MaxTokens, MaxTurns, Output,
        Owner, Preamble, Settled, Temperature, ToolChoiceSpec, UsesModel, Utterance,
        content::parts::read_message,
    },
    bus::{BusPlugin, Handlers, PendingEffect},
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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        match kind {
            EffectKind::Completion { request, .. } => {
                self.requests.lock().expect("requests").push(request);
                let response = CompletionResponse::new(
                    vec![AssistantContent::text(&self.answer)],
                    Usage::default(),
                    "capturing",
                    serde_json::json!({ "provider": "capturing" }),
                );
                rig_core::serve::Reply::Outcome(Ok(Outcome::Completion(response)))
            }
            other => rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!("a model cannot serve {}", other.name()),
            ))),
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

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::Internal,
            "a tool advertised and never called was called",
        )))
    }
}

/// An app with the bus and the agent installed, ambiguity detection at
/// error level, the runner in `Update`.
pub fn app() -> App {
    let mut app = App::new();
    app.add_plugins(
        rig_ecs::RigPlugin::with_policy(ServingPolicy::default())
            .ambiguity_detection(LogLevel::Error),
    );
    app.add_plugins(rig_cassette::ecs::ReplayPlugin);
    app.finish();
    app.cleanup();
    app
}

#[path = "../bus_support/registration.rs"]
mod registration;
pub use registration::register;

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

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        std::future::pending().await
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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        match kind {
            EffectKind::Completion { request, .. } => {
                self.requests.lock().expect("requests").push(request);
                let choice = self
                    .turns
                    .lock()
                    .expect("turns")
                    .pop_front()
                    .unwrap_or_else(|| vec![AssistantContent::text("done")]);
                let response = CompletionResponse::new(
                    choice,
                    Usage::default(),
                    "scripted",
                    serde_json::json!({}),
                );
                rig_core::serve::Reply::Outcome(Ok(Outcome::Completion(response)))
            }
            other => rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!("a model cannot serve {}", other.name()),
            ))),
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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        use std::sync::atomic::Ordering;
        let EffectKind::ToolCall { args, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Request,
                "a tool call",
            )));
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
        rig_core::serve::Reply::Outcome(Ok(Outcome::ToolResult {
            result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::json(
                serde_json::json!(sum),
            )),
        }))
    }
}

/// The requests a registered model kept, shared with the test.
pub type RequestsSeen = Arc<Mutex<Vec<CompletionRequest>>>;

/// A [`Scripted`] model registered under `key`, and an agent over it.
pub fn scripted_agent(
    app: &mut App,
    key: &str,
    turns: Vec<Vec<AssistantContent>>,
) -> (Entity, RequestsSeen) {
    let (model, requests) = Scripted::new(key, turns);
    let model = register(app, key, model);
    (spawn_agent(app.world_mut(), "t", model), requests)
}

/// A [`Capturing`] model registered under `key`, labelled `label` in its
/// own descriptor, and an agent over it.
pub fn capturing_agent(
    app: &mut App,
    key: &str,
    label: &str,
    answer: &str,
) -> (Entity, RequestsSeen) {
    let (model, requests) = Capturing::new(label, answer);
    let model = register(app, key, model);
    (spawn_agent(app.world_mut(), "t", model), requests)
}

/// Tick until `run` settled or failed, or fail after [`GUARD`].
pub fn ended(app: &mut App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
}

/// A bare world with the bus and the agent installed, an open completion
/// handler under `model`, and one agent over it.
pub fn open_model_world() -> (World, Entity) {
    let mut world = World::new();
    BusPlugin::with_policy(ServingPolicy::default()).install(&mut world);
    AgentPlugin::install(&mut world);
    rig_ecs::checkpoint::register_types(&mut world);
    let model = Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "model",
            FamilyDescriptor::Completion {
                model: ModelRef::new("model"),
                capabilities: ProviderCapabilities::default(),
            },
        )
    })
    .unwrap()
    .unwrap();
    let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
    (world, agent)
}

/// The one utterance `ChildOf` `run`.
pub fn first_utterance(world: &mut World, run: Entity) -> Entity {
    world
        .query_filtered::<(Entity, &ChildOf), With<Utterance>>()
        .iter(world)
        .find(|(_, parent)| parent.parent() == run)
        .unwrap()
        .0
}

/// The utterances `ChildOf` `run`, in sibling (`Children`) order.
pub fn utterances_of(world: &mut World, run: Entity) -> Vec<Entity> {
    world
        .get::<Children>(run)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter(|child| world.get::<Utterance>(*child).is_some())
        .collect()
}

/// The fresh, uncached render of `run`'s history, as the DTOs the graph
/// reconstructs.
pub fn graph_messages(world: &mut World, run: Entity) -> Vec<Message> {
    utterances_of(world, run)
        .into_iter()
        .map(|entity| read_message(world, entity).unwrap().to_message())
        .collect()
}

/// The completion requests folded under `run`, by turn order (the run's
/// sibling order).
pub fn requests(world: &mut World, run: Entity) -> Vec<CompletionRequest> {
    let turns: Vec<Entity> = world
        .get::<Children>(run)
        .into_iter()
        .flat_map(|children| children.iter())
        .collect();
    let mut found: Vec<_> = world
        .query::<(&PendingEffect, &ChildOf)>()
        .iter(world)
        .filter_map(|(effect, parent)| match &effect.kind {
            EffectKind::Completion { request, .. } => {
                let position = turns.iter().position(|turn| *turn == parent.parent())?;
                Some((position, request.clone()))
            }
            _ => None,
        })
        .collect();
    found.sort_by_key(|(position, _)| *position);
    found.into_iter().map(|(_, request)| request).collect()
}
