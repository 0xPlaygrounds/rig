//! Shared by the `run_*` suites: a request-capturing model, a tool handler
//! that is never called, an app with both plugins, and the tick guard.

#![allow(dead_code, reason = "each suite uses the part of the support it needs")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "test support fails immediately when a fixture invariant is violated"
)]

use rig_core::serve::Dispatch;
use std::sync::{Arc, Mutex};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    completion::CompletionRequest,
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{Serve, ServingPolicy},
};
use rig_ecs::{
    RigPlugin,
    agent::{
        AdditionalParams, DefaultMaxTurns, InvalidCalls, MaxTokens, MaxTurns, Output, Owner,
        Preamble, Temperature, ToolChoiceSpec, UsesModel,
    },
    bus::Bus,
};

pub const GUARD: std::time::Duration = std::time::Duration::from_secs(10);

/// An app with the bus and the agent installed, ambiguity detection at
/// error level, the runner in `Update`.
pub fn app() -> App {
    let mut app = App::new();
    app.add_plugins(RigPlugin {
        bus: Bus::with_policy(ServingPolicy::default()).ambiguity_detection(LogLevel::Error),
    });
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
