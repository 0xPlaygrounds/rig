//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::Task;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{Message, Reasoning};
use rig_core::effect::EffectId;
use rig_core::message::{ToolCall, ToolResult};
use rig_core::serve::ErasedHandler;
use serde::{Deserialize, Serialize};

use super::save::ReflectSaved;

/// Marks an agent. Spawning it adds every per-agent component with its
/// default, including a fresh [`AgentId`]. An agent has no [`ModelChoice`]
/// until one is picked.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(AgentId, Conversation, Effort, SystemPrompt, ToolAccess, AgentStatus)]
pub struct Agent;

/// The agent's stable id, used in saved state, effect scopes and logs.
/// `Entity` ids are not stable across a restart; this one is. It never
/// changes after spawn.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
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

/// The chosen catalog model, as `vendor/model`. It never changes in place:
/// choosing another model inserts a new one, and each insert rebuilds the
/// agent's [`Connection`].
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Clone, Saved)]
pub struct ModelChoice(pub String);

/// The connected model of an agent with a [`ModelChoice`]: its catalog
/// entry and the effect handler every model call is dispatched to. Built
/// from the environment once per choice, and not saved: restoring the
/// choice rebuilds it.
#[derive(Component, Clone)]
pub struct Connection {
    /// The model's catalog entry.
    pub spec: &'static ModelSpec,
    /// The model as an effect handler.
    pub handler: ErasedHandler,
}

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

/// Where a tool call of the last reply is. The calls of one reply run one
/// at a time, in order: two edits of one file in a reply would otherwise
/// both read the original text, and one would be lost.
pub enum ToolState {
    /// Waiting for the reply's earlier calls.
    Queued,
    /// Running; dropping the task cancels it.
    Running(Task<ToolResult>),
    /// Finished.
    Done(ToolResult),
}

/// A tool call of the model's last reply.
#[derive(Component)]
pub struct ToolCallRun {
    /// The call's position in the reply.
    pub index: usize,
    /// The call.
    pub call: ToolCall,
    /// The effect id of the model call that asked for it.
    pub parent: EffectId,
    /// Where it is.
    pub state: ToolState,
}

impl ToolCallRun {
    /// The result, once the tool finished.
    pub fn result(&self) -> Option<&ToolResult> {
        match &self.state {
            ToolState::Done(result) => Some(result),
            ToolState::Queued | ToolState::Running(_) => None,
        }
    }
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

/// How a [`Notice`] is shown and logged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoticeLevel {
    /// Information.
    #[default]
    Info,
    /// Something failed or was refused.
    Error,
}

/// A line for the user, shown by views and logged.
#[derive(Message, Clone, Debug)]
pub struct Notice {
    /// The agent it is about, or `None` when it is about the whole app.
    pub agent: Option<Entity>,
    /// The text.
    pub text: String,
    /// Whether it reports a failure.
    pub level: NoticeLevel,
}

impl Notice {
    /// Information about `agent`, or about the whole app with `None`.
    pub fn info(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            text: text.into(),
            level: NoticeLevel::Info,
        }
    }

    /// A failure or refusal concerning `agent`, or the whole app with
    /// `None`.
    pub fn error(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            text: text.into(),
            level: NoticeLevel::Error,
        }
    }
}

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
