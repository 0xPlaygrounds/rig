//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{Message, Reasoning};
use rig_core::message::ToolCall;
use serde::{Deserialize, Serialize};

use super::session::ReflectSaved;

/// Marks an agent. Spawning it adds every per-agent component with its
/// default, including a fresh [`AgentId`].
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(
    AgentId,
    Conversation,
    ModelChoice,
    Effort,
    SystemPrompt,
    ToolAccess,
    AgentStatus
)]
pub struct Agent;

/// The agent's stable id, used in saved state, effect scopes and logs.
/// `Entity` ids are not stable across a restart; this one is.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Component)]
pub struct AgentId(pub String);

impl Default for AgentId {
    fn default() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

/// The conversation: every message sent to and received from the model.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct Conversation(pub Vec<Message>);

/// The chosen catalog model, as `vendor/model`, or `None` until one is
/// picked.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct ModelChoice(pub Option<String>);

/// The reasoning setting sent with each request, or `None` for the
/// provider's default.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct Effort(pub Option<Reasoning>);

/// The system prompt sent first in every request.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        Self(
            "You are a coding agent working in the user's current directory. Use the tools \
             to read, search, edit and write files and to run shell commands. Read a file \
             before you edit it. Keep answers short."
                .to_owned(),
        )
    }
}

/// Which registered tools the agent may call.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub enum ToolAccess {
    /// Every registered tool.
    #[default]
    All,
    /// Only the tools named.
    Only(Vec<String>),
}

impl ToolAccess {
    /// Whether the tool `name` may be called.
    pub fn allows(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.iter().any(|allowed| allowed == name),
        }
    }
}

/// What the agent is doing. Not saved: a restored agent is idle.
#[derive(Component, Reflect, Clone, Copy, Default, Debug, PartialEq, Eq)]
#[reflect(Component)]
pub enum AgentStatus {
    /// Waiting for input.
    #[default]
    Idle,
    /// A model call is starting or streaming.
    Thinking,
    /// Tool calls are running.
    RunningTools,
}

/// The agent needs a model call: `AgentSystems::Start` sends the
/// conversation.
#[derive(Component, Default)]
#[component(storage = "SparseSet")]
pub struct NeedsCompletion;

/// Work in flight that belongs to an agent. Despawning the agent despawns
/// its calls, and dropping a call's task cancels it.
#[derive(Component, Debug)]
#[relationship(relationship_target = Calls)]
pub struct CallOf(pub Entity);

/// The calls an agent owns.
#[derive(Component, Debug)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct Calls(Vec<Entity>);

/// Text streamed so far by an in-flight model call, for views.
#[derive(Component, Default)]
pub struct Partial {
    /// Answer text.
    pub text: String,
    /// Reasoning text.
    pub reasoning: String,
}

/// A tool call of the model's last reply, running or finished.
#[derive(Component)]
pub struct ToolCallRun {
    /// The call's position in the reply.
    pub index: usize,
    /// The call.
    pub call: ToolCall,
    /// The result, once the tool finished.
    pub result: Option<rig_core::message::ToolResult>,
}

/// Send `text` to the agent: a slash command when it starts with `/`,
/// otherwise a user message that starts a turn.
#[derive(EntityEvent, Clone, Debug)]
pub struct Submit {
    /// The agent.
    pub entity: Entity,
    /// The text typed.
    pub text: String,
}

/// Stop the agent's running turn.
#[derive(EntityEvent, Clone, Debug)]
pub struct Interrupt {
    /// The agent.
    pub entity: Entity,
}

/// Choose the agent's model by catalog reference (`vendor/model`).
#[derive(EntityEvent, Clone, Debug)]
pub struct SetModel {
    /// The agent.
    pub entity: Entity,
    /// The catalog reference.
    pub model: String,
}

/// Choose the agent's reasoning setting; `None` is the provider default.
#[derive(EntityEvent, Clone, Debug)]
pub struct SetEffort {
    /// The agent.
    pub entity: Entity,
    /// The setting.
    pub effort: Option<Reasoning>,
}

/// A line for the user, shown by views and logged.
#[derive(Message, Clone, Debug)]
pub struct Notice(pub String);

/// An agent's turn ended: answered, failed or interrupted.
#[derive(Message, Clone, Copy, Debug)]
pub struct TurnFinished {
    /// The agent.
    pub agent: Entity,
}

/// What a view should let the user pick from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickKind {
    /// A model from [`available_models`](crate::core::models::available_models).
    Model,
    /// A reasoning setting from [`effort_options`](crate::core::models::effort_options).
    Effort,
}

/// Asks a view to open a picker for the agent. The view answers with
/// [`SetModel`] or [`SetEffort`].
#[derive(Message, Clone, Copy, Debug)]
pub struct PickRequest {
    /// The agent.
    pub agent: Entity,
    /// What to pick.
    pub kind: PickKind,
}
