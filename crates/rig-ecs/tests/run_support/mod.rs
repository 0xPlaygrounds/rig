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
