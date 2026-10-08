//! The agent's building blocks: one component per thing that can differ
//! between agents, and the relationship that ties calls in flight to their
//! agent.

use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::prelude::*;
use rig_core::{completion::Effort, message::Message};
use serde::{Deserialize, Serialize};

use super::session::ReflectSave;

/// Marks an agent. Spawning it alone gives a working agent: every other
/// building block is required with its default.
#[derive(Component, Debug, Default)]
#[require(
    AgentId = AgentId::fresh(),
    Conversation,
    ModelChoice,
    EffortChoice,
    SystemPrompt,
    Workdir,
    ToolAccess,
    AgentStatus
)]
pub struct Agent;

/// The agent's stable id, used in effects, logs and saved state, where an
/// [`Entity`] would not survive a restart.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
pub struct AgentId(pub Arc<str>);

impl AgentId {
    /// A new id, unique on this machine: `agent-` and the current time in
    /// nanoseconds, in hex.
    pub fn fresh() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default();
        Self(format!("agent-{nanos:x}").into())
    }
}

/// The conversation, oldest message first. The system prompt is not in it.
/// Reflected as one opaque value saved through rig's own serde format.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Clone, Serialize, Deserialize, Save)]
#[serde(transparent)]
pub struct Conversation(pub Vec<Message>);

/// The chosen model as a catalog reference such as `deepseek/deepseek-flash`,
/// or `None` before one is picked.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Default, Save)]
pub struct ModelChoice(pub Option<String>);

/// How much the model reasons. `Level` is a named effort; on a model that
/// takes a token budget instead, it maps to a budget. Reflected as an opaque
/// value because rig's [`Effort`] is serde, not reflect.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[reflect(opaque)]
#[reflect(Component, Default, Clone, Serialize, Deserialize, Save)]
pub enum EffortChoice {
    /// Whatever the model does when a request names nothing.
    #[default]
    Default,
    /// Reasoning turned off.
    Off,
    /// A named level.
    Level(Effort),
}

impl EffortChoice {
    /// The name `/effort` takes and shows.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Off => "off",
            Self::Level(effort) => effort.as_str(),
        }
    }
}

/// The instructions sent ahead of the conversation.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Save)]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        Self(
            "You are a coding agent working in a software project. Use the tools to read, \
             search and edit files and to run shell commands. Read before you edit, keep \
             changes small, and answer concisely."
                .to_owned(),
        )
    }
}

/// The directory the agent's tools work in. Relative tool paths resolve
/// against it.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Save)]
pub struct Workdir(pub PathBuf);

impl Default for Workdir {
    fn default() -> Self {
        Self(std::env::current_dir().unwrap_or_default())
    }
}

/// Which registered tools the agent may call.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default, Save)]
pub enum ToolAccess {
    /// Every registered tool.
    #[default]
    All,
    /// Only the tools with these names.
    Only(Vec<String>),
}

impl ToolAccess {
    /// Whether the tool named `name` is allowed.
    pub fn allows(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.iter().any(|allowed| allowed == name),
        }
    }
}

/// What the agent is doing.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AgentStatus {
    /// Waiting for input.
    #[default]
    Idle,
    /// A model call is starting or streaming.
    Thinking,
    /// Tool calls are running.
    Tools,
}

/// The reply streaming in, present only while a model call runs.
#[derive(Component, Clone, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct Draft {
    /// Answer text so far.
    pub text: String,
    /// Reasoning text so far.
    pub reasoning: String,
}

/// Asks the loop to send the conversation to the model on the next frame.
#[derive(Component, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct NeedsReply;

/// A call in flight, tied to the agent that made it. Despawning the agent
/// despawns its calls, which cancels their work.
#[derive(Component, Debug)]
#[relationship(relationship_target = Calls)]
pub struct CallOf(pub Entity);

/// The calls in flight of an agent.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct Calls(Vec<Entity>);
