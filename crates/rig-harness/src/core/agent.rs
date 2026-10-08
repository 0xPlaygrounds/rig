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
use super::inbox::{Inbox, Origin, RequestId};
use super::recovery::Recovery;
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

/// The agent's stable id, used in its log's name, effect scopes and logs.
/// `Entity` ids are not stable across a restart; this one is. It never
/// changes after spawn.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[component(immutable)]
#[reflect(Component)]
#[serde(transparent)]
pub struct AgentId(pub String);

impl Default for AgentId {
    fn default() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl AgentId {
    /// The first characters of the id, enough to tell agents apart.
    pub fn short(&self) -> &str {
        self.0.get(..8).unwrap_or(&self.0)
    }
}

/// The agent that spawned this one, such as the agent whose tool call
/// started it. Despawning that agent despawns this one. The core gives it
/// no other meaning: a restore links the agents again by their logs'
/// headers, views list agents by it, and [`TurnEnded`] travels up it.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = Spawned)]
pub struct SpawnedBy(pub Entity);

/// The agents this one spawned, in the order it spawned them.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = SpawnedBy, linked_spawn)]
pub struct Spawned(Vec<Entity>);

/// The effect the agent's model calls are recorded under, such as the
/// tool call that asked it for the work it does now, so the effect log
/// nests that work under the call. An agent without one records its model
/// calls at the top level. Not saved: effect ids do not outlive a run.
#[derive(Component, Clone, Copy, Debug)]
pub struct EffectParent(pub EffectId);

/// The conversation: every message sent to and received from the model,
/// and where each delivered text came from when it is not the user's own.
/// Requests leave out the messages its agent's [`Compacted`] replaced with
/// a summary. Messages are added only through the
/// [`SessionLog`](super::journal::SessionLog), which logs each one.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize)]
pub struct Conversation {
    messages: Vec<Message>,
    /// The origin of each content item that is not the user's own or the
    /// model's, by message and item index, in the order they were added.
    origins: Vec<(usize, usize, Origin)>,
}

impl Conversation {
    /// The messages, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Where content item `item` of message `message` came from, when it
    /// was delivered by an agent or a plugin; `None` for the user's own
    /// text, the model's messages and tool results.
    pub fn origin(&self, message: usize, item: usize) -> Option<&Origin> {
        self.origins
            .iter()
            .find(|(at, index, _)| *at == message && *index == item)
            .map(|(.., origin)| origin)
    }

    /// The messages, to change in place, such as clearing old tool outputs.
    /// Nothing can be added or taken out through it.
    pub(crate) fn messages_mut(&mut self) -> &mut [Message] {
        &mut self.messages
    }

    /// Adds `message`, whose content came from `origin`: a user message
    /// goes into the last message when that is the user's too, such as the
    /// tool results the model waits for, so user and model keep taking
    /// turns. Whether it went into the last one.
    pub(in crate::core) fn append(&mut self, message: Message, origin: Option<Origin>) -> bool {
        let (at, first) = match (self.messages.last(), &message) {
            (Some(Message::User { content }), Message::User { .. }) => {
                (self.messages.len().saturating_sub(1), content.len())
            }
            _ => (self.messages.len(), 0),
        };
        let items = match &message {
            Message::User { content } => content.len(),
            _ => 0,
        };
        if let Some(origin) = origin {
            self.origins
                .extend((first..first + items).map(|item| (at, item, origin.clone())));
        }
        match (self.messages.last_mut(), message) {
            (Some(Message::User { content }), Message::User { content: added }) => {
                content.extend(added);
                true
            }
            (_, message) => {
                self.messages.push(message);
                false
            }
        }
    }

    /// Takes out the last message.
    pub(in crate::core) fn retract(&mut self) -> Option<Message> {
        let message = self.messages.pop()?;
        let len = self.messages.len();
        self.origins.retain(|(at, ..)| *at < len);
        Some(message)
    }
}

/// The chosen catalog model, as `vendor/model`. It never changes in place:
/// choosing another model inserts a new one, and each insert rebuilds the
/// agent's [`Connection`] and logs the agent's settings.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Clone)]
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
/// provider's default. It never changes in place: each insert logs the
/// agent's settings.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[component(immutable)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct Effort(pub Option<Reasoning>);

/// The agent's own part of its system prompt: who it is and how it works.
/// Each request's prompt adds the rules of the tools the agent is offered
/// and the app's [`PromptSection`](super::prompt::PromptSection)s, such as
/// the project's instructions and the environment.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize)]
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
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize)]
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
#[require(TurnSpending, Recovery, TurnRequest)]
pub struct TurnOf(pub Entity);

/// On a turn: the request of the latest message delivered to it that
/// carried one, which its [`TurnEnded`] names.
#[derive(Component, Clone, Debug, Default)]
pub struct TurnRequest(pub Option<RequestId>);

/// On a turn about to be despawned: how it ended. A turn despawned without
/// one, such as by [`Interrupt`] or on exit, ended [`TurnOutcome::Stopped`].
#[derive(Component, Clone, Debug)]
pub(crate) struct Ending(pub(crate) TurnOutcome);

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
/// [`Running<ToolResult>`](super::calls::Running), or stays open for an
/// open tool, and ends with a [`ToolOutput`](super::tools::ToolOutput).
/// Calls that only read run side by side; two edits of one file run in
/// order, or one would be lost.
#[derive(Component, Clone)]
pub struct ToolCallRun {
    /// The call.
    pub call: ToolCall,
    /// The effect id of the model call that asked for it; `None` for a
    /// call that a restart runs again.
    pub parent: Option<EffectId>,
    /// What it touches.
    pub(crate) touch: Touch,
}

/// A tool call waiting for earlier calls of its reply that touch what it
/// touches.
#[derive(Component, Default)]
#[component(storage = "SparseSet")]
pub struct Queued;

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

/// The user chose the agent's model or reasoning setting with [`SetModel`]
/// or [`SetEffort`], and it took: the agent's [`ModelChoice`] and [`Effort`]
/// hold the choice. Restoring a session never sends it.
#[derive(EntityEvent, Clone, Debug)]
pub struct SettingsChosen {
    /// The agent.
    pub entity: Entity,
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

/// How a turn ended.
#[derive(Clone, Debug)]
pub enum TurnOutcome {
    /// The model answered: its last message, which ended the turn.
    Answered(Message),
    /// The turn failed, for the reason given.
    Failed(String),
    /// The turn was stopped before an answer, by the user or on exit.
    Stopped,
}

/// An agent's turn ended, however its turn entity went away. Triggered on
/// the agent once it is idle, then on each agent it was
/// [`SpawnedBy`] up the chain: an observer's `entity` is the agent seeing
/// it and `original_event_target()` the agent whose turn ended.
/// Not triggered for a turn the app's exit stops: the restart carries
/// that one on.
#[derive(EntityEvent, Clone, Debug)]
#[entity_event(propagate = &'static SpawnedBy, auto_propagate)]
pub struct TurnEnded {
    /// The agent seeing the event.
    pub entity: Entity,
    /// How the turn ended.
    pub outcome: TurnOutcome,
    /// The request the turn answered, if a delivered message carried one.
    pub request: Option<RequestId>,
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
    /// An agent to show, from [`roster`];
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

/// One agent in [`roster`]: how deep it is and a line describing it.
#[derive(Clone, Debug)]
pub struct RosterEntry {
    /// The agent.
    pub agent: Entity,
    /// 0 for an agent nothing spawned, 1 for the agents it spawned, and so
    /// on.
    pub depth: usize,
    /// Its title, model, state and cost, indented by depth.
    pub label: String,
}

/// What [`roster`] reads of each agent.
pub type RosterQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AgentId,
        Option<&'static Name>,
        Option<&'static SpawnedBy>,
        Option<&'static Spawned>,
        Option<&'static ModelChoice>,
        Has<ActiveTurn>,
        &'static Spending,
    ),
    With<Agent>,
>;

/// Every agent as a tree: the agents nothing spawned, by id, each followed
/// by the agents it spawned in the order it spawned them. A spawned agent
/// is titled by its [`Name`].
pub fn roster(agents: &RosterQuery) -> Vec<RosterEntry> {
    let mut roots: Vec<(Entity, &AgentId)> = agents
        .iter()
        .filter(|(_, _, _, of, ..)| of.is_none_or(|of| !agents.contains(of.0)))
        .map(|(entity, id, ..)| (entity, id))
        .collect();
    roots.sort_by(|a, b| a.1.0.cmp(&b.1.0));
    let several = roots.len() > 1;
    let total = agents.iter().count();
    let mut stack: Vec<(Entity, usize)> = roots.iter().rev().map(|(root, _)| (*root, 0)).collect();
    let mut entries = Vec::new();
    while let Some((agent, depth)) = stack.pop() {
        // A relationship loop cannot happen, but a bound costs nothing.
        if entries.len() >= total {
            break;
        }
        let Ok((_, id, name, of, spawned, model, busy, spent)) = agents.get(agent) else {
            continue;
        };
        let title = match (name, of) {
            (Some(name), Some(_)) => name.as_str().to_owned(),
            _ if several => format!("agent {}", id.short()),
            _ => "main agent".to_owned(),
        };
        let mut label = format!(
            "{}{title} · {} · {}",
            "  ".repeat(depth),
            model.map_or("no model", |model| model.0.as_str()),
            if busy { "working" } else { "idle" }
        );
        if let Some(cost) = spent.cost_label() {
            label.push_str(&format!(" · {cost}"));
        }
        entries.push(RosterEntry {
            agent,
            depth,
            label,
        });
        for child in spawned.into_iter().flat_map(|spawned| spawned.iter().rev()) {
            stack.push((child, depth + 1));
        }
    }
    entries
}
