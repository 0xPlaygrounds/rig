//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{Message, Reasoning};
use rig_core::effect::EffectId;
use rig_core::message::ToolCall;
use rig_core::serve::ErasedHandler;
use serde::{Deserialize, Serialize};

use super::compaction::Compacted;
use super::inbox::Inbox;
use super::recovery::Recovery;
use super::save::ReflectSaved;
use super::tools::Touch;
use super::usage::{Spending, TurnSpending};

/// Marks an agent. Spawning it adds every per-agent component with its
/// default, including a fresh [`AgentId`]. An agent has no [`ModelChoice`]
/// until one is picked.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(
    AgentId,
    Compacted,
    Conversation,
    Effort,
    Inbox,
    Spending,
    SystemPrompt,
    ToolAccess
)]
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
/// Requests leave out the ones its agent's
/// [`Compacted`] replaced with a summary.
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
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Default,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    Saved
)]
pub struct Effort(pub Option<Reasoning>);

/// The agent's own part of its system prompt: who it is and how it works.
/// Each request's prompt adds the rules of the tools the agent is offered
/// and the app's [`PromptSection`](super::prompt::PromptSection)s, such as
/// the project's instructions and the environment.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize, Saved)]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        Self(
            "You are rig, a coding agent. You work in the user's project from their terminal: \
             you read and search code, edit files and run commands with the tools you are \
             given, and answer questions about the code.\n\
             \n\
             - Work in the working directory named below, unless the user says otherwise.\n\
             - Read a file before you edit it. Change what was asked, in the style of the \
             code around it, and nothing else.\n\
             - After a change, check it when you can: build it, run the tests, or run the \
             code.\n\
             - When something fails, read the error and fix the cause. Ask the user when the \
             request is unclear or you are blocked, rather than guess.\n\
             - Do not undo changes you did not make, and do not run commands that delete \
             work, rewrite history or reach outside the project unless the user asked.\n\
             - Keep answers short. Say what you changed and what is left, and name files by \
             their path."
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

/// A running turn of the agent it names: from a user message to the reply
/// that ends it. At most one per agent. Despawning the turn stops it and
/// cancels its calls; its end removes the agent's [`ActiveTurn`]. The turn
/// sums its model calls' usage in a [`TurnSpending`] and counts its
/// retries in a [`Recovery`].
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = ActiveTurn)]
#[require(TurnSpending, Recovery)]
pub struct TurnOf(pub Entity);

/// The agent's running turn. An agent with it is busy.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = TurnOf, linked_spawn)]
pub struct ActiveTurn(Entity);

impl ActiveTurn {
    /// The turn entity.
    pub fn turn(&self) -> Entity {
        self.0
    }
}

/// A model or tool call of the turn it names. Despawning the turn despawns
/// its calls, which cancels their tasks.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Calls)]
pub struct CallOf(pub Entity);

/// The calls of a turn, in the order they were spawned: the tool calls of
/// a reply in the reply's order.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
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

/// A tool call of the model's last reply. It is [`Queued`] while an
/// earlier call of the reply touches what it touches (see
/// [`Footprint`](super::tools::Footprint)), then runs as a
/// [`Running<ToolResult>`](super::calls::Running) and ends with a
/// [`Done<ToolResult>`](super::calls::Done). Calls that only read run side
/// by side; two edits of one file run in order, or one would be lost.
#[derive(Component)]
pub struct ToolCallRun {
    /// The call.
    pub call: ToolCall,
    /// The effect id of the model call that asked for it.
    pub parent: EffectId,
    /// What it touches.
    pub(crate) touch: Touch,
}

/// A tool call waiting for earlier calls of its reply that touch what it
/// touches.
#[derive(Component, Default)]
#[component(storage = "SparseSet")]
pub struct Queued;

/// Send `text` to the agent: a slash command when it starts with `/`,
/// otherwise a user message that starts a turn, or that steers the running
/// one (see [`Inbox`]). An image named as `@path` goes with it when the
/// model reads images.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Submit {
    /// The agent.
    pub entity: Entity,
    /// The text typed.
    pub text: String,
}

/// Send the agent's conversation to its model again as it stands: after a
/// turn that failed on a transient error and kept the user's message, or
/// one that stopped before the model answered the last tool results.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Retry {
    /// The agent.
    pub entity: Entity,
}

/// Compact the agent's conversation now: its older messages are replaced
/// in requests by a summary the model writes, focused on `focus` when it is
/// not empty. Refused while a turn runs.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Compact {
    /// The agent.
    pub entity: Entity,
    /// What the summary should keep above all; may be empty.
    pub focus: String,
}

/// Ask the views to show the agent and send what is typed to it, such as
/// a subagent picked with `/agents`. The core does nothing with it.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Focus {
    /// The agent.
    pub entity: Entity,
}

/// Stop the agent's running turn.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Interrupt {
    /// The agent.
    pub entity: Entity,
}

/// Choose the agent's model by catalog reference (`vendor/model`).
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetModel {
    /// The agent.
    pub entity: Entity,
    /// The catalog reference.
    pub model: String,
}

/// Choose the agent's reasoning setting; `None` is the provider default.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetEffort {
    /// The agent.
    pub entity: Entity,
    /// The setting.
    pub effort: Effort,
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

/// An agent's turn ended: answered, failed or interrupted. Written when its
/// [`ActiveTurn`] goes away, however the turn entity was despawned.
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
    /// An earlier session to resume; the view answers with the host's
    /// `SwitchSession`.
    Session,
    /// An agent to show, from [`roster`](crate::core::subagents::roster);
    /// the view answers with [`Focus`].
    Agent,
}

/// Asks a view to open a picker for the agent. The view answers with
/// what the [`PickKind`] names.
#[derive(Message, Clone, Copy, Debug)]
pub struct PickRequest {
    /// The agent.
    pub agent: Entity,
    /// What to pick.
    pub kind: PickKind,
}
