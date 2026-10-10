//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{Message, Usage};
use rig_core::effect::EffectId;
use rig_core::message::{ToolCall, UserContent};
use serde::{Deserialize, Serialize};

use super::inbox::{Inbox, Origin};
use super::journal::ReflectSaved;
use super::model::Effort;
use super::tools::Footprint;
use super::turn::Recovery;

/// Marks an agent. Spawning it adds every per-agent component with its
/// default, including a fresh [`AgentId`]. An agent has no
/// [`ModelChoice`](super::model::ModelChoice) until one is picked.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(
    AgentId,
    Conversation,
    Effort,
    Inbox,
    LastUsage,
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
/// headers, and views list agents by it.
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
/// Requests leave out the messages its agent's [`Condensed`] replaced with
/// a summary. It changes only through a
/// [`Commit`](super::journal::Commit), which logs each change.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize)]
pub struct Conversation {
    messages: Vec<Message>,
    /// The origin of each content item that is not the user's own or the
    /// model's, by message and item index, in the order they were added.
    origins: Vec<(usize, usize, Origin)>,
    /// Why the user's last message was left unanswered, until a message
    /// is added or taken out.
    #[serde(default)]
    halted: Option<Halt>,
    /// The `seq` of the log record that began each message, while the
    /// session is logged.
    #[serde(skip)]
    seqs: Vec<u64>,
}

/// The text that goes between a request the user stopped and the next
/// message, which joins it, so the model does not carry the request out.
pub const STOPPED: &str = "[The request above was stopped by the user before it was answered. \
                           Do not carry it out unless asked again.]";

/// Why the user's last message is left without an answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Halt {
    /// The user stopped the turn answering it.
    Stopped,
    /// It was left for the next turn, such as a note to an idle agent or
    /// a request whose turn failed: the next message joins it.
    #[default]
    Kept,
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
    /// Nothing can be added or taken out through it, and nothing changed
    /// through it is logged: a restored session has the messages as they
    /// were added.
    pub fn messages_mut(&mut self) -> &mut [Message] {
        &mut self.messages
    }

    /// Whether the model owes an answer: the last message is the user's,
    /// and it was not halted.
    pub(crate) fn awaits_model(&self) -> bool {
        self.halted.is_none() && matches!(self.messages.last(), Some(Message::User { .. }))
    }

    /// Leaves the user's last message unanswered, for `reason`. Whether it
    /// was halted now: not when the model owes no answer.
    pub(crate) fn halt(&mut self, reason: Halt) -> bool {
        let halts = self.awaits_model();
        if halts {
            self.halted = Some(reason);
        }
        halts
    }

    /// Asks the model again for an answer to the user's last message, also
    /// a halted one; whether the last message is the user's. Not logged: a
    /// restore reads a halt after a halt as asked again in between.
    pub(crate) fn resume(&mut self) -> bool {
        self.halted = None;
        self.awaits_model()
    }

    /// The `seq` of the log record that began message `message`, if it was
    /// logged.
    pub(crate) fn seq(&self, message: usize) -> Option<u64> {
        self.seqs.get(message).copied()
    }

    /// Adds `message`, whose content came from `origin` and which the log
    /// recorded as `seq`: a user message goes into the last message when
    /// that is the user's too, such as the tool results the model waits
    /// for, so user and model keep taking turns. After [`STOPPED`] when
    /// that is a request the user stopped.
    pub(crate) fn append(&mut self, message: Message, origin: Option<Origin>, seq: Option<u64>) {
        let stopped = self.halted.take() == Some(Halt::Stopped);
        let (at, first) = match (self.messages.last(), &message) {
            (Some(Message::User { content }), Message::User { .. }) => (
                self.messages.len().saturating_sub(1),
                content.len() + usize::from(stopped),
            ),
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
                if stopped {
                    content.push(UserContent::text(STOPPED));
                }
                content.extend(added);
            }
            (_, message) => {
                self.messages.push(message);
                self.seqs.extend(seq);
            }
        }
    }

    /// Takes out the last message.
    pub(crate) fn retract(&mut self) -> Option<Message> {
        let message = self.messages.pop()?;
        self.halted = None;
        let len = self.messages.len();
        self.origins.retain(|(at, ..)| *at < len);
        self.seqs.truncate(len);
        Some(message)
    }
}

/// What requests send in place of the conversation's older messages, such
/// as a compaction's summary: `summary`, as one user text, in place of the
/// first `upto` messages, which stay in the [`Conversation`] for views. It
/// never changes in place: inserting another logs it, and a restored
/// session starts from the first message it kept. An agent without one
/// sends every message.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Condensed {
    /// How many of the conversation's first messages `summary` replaces.
    pub upto: usize,
    /// The text sent in their place.
    pub summary: String,
}

impl Condensed {
    /// The messages requests send as they are.
    pub fn live<'a>(&self, messages: &'a [Message]) -> &'a [Message] {
        messages
            .get(self.upto.min(messages.len())..)
            .unwrap_or_default()
    }

    /// The messages a request sends: the summary, then the live messages.
    /// The summary goes into the first live message when that is the
    /// user's, so user and assistant messages still alternate.
    pub fn request(&self, messages: &[Message]) -> Vec<Message> {
        let mut live = self.live(messages).to_vec();
        let summary = UserContent::text(self.summary.clone());
        match live.first_mut() {
            Some(Message::User { content }) => content.insert(0, summary),
            _ => live.insert(
                0,
                Message::User {
                    content: vec![summary],
                },
            ),
        }
        live
    }
}

/// The usage of the agent's last model reply that reported its tokens,
/// `None` before one did. A plugin that shrinks what requests send, as
/// compaction does,
/// replaces it with its estimate of the tokens the next request sends, in
/// `total_tokens`. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LastUsage(pub Option<Usage>);

impl LastUsage {
    /// The tokens the last reply read and wrote, which the next request
    /// sends again, when it reported them.
    pub fn context(&self) -> Option<u64> {
        self.0.as_ref()?.context_tokens()
    }
}

/// The agent's own part of its system prompt: who it is and how it works.
/// Each request's prompt adds the rules of the tools the agent is offered
/// and the app's [`PromptSection`](super::prompt::PromptSection)s, such as
/// the project's instructions and the environment. Saved with the
/// session, as `null` when it is the default, so a restored agent gets the
/// default of the build that restores it.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(from = "Option<String>", into = "Option<String>")]
pub struct SystemPrompt(pub String);

impl From<SystemPrompt> for Option<String> {
    fn from(prompt: SystemPrompt) -> Self {
        (prompt.0 != SystemPrompt::default().0).then_some(prompt.0)
    }
}

impl From<Option<String>> for SystemPrompt {
    fn from(prompt: Option<String>) -> Self {
        prompt.map_or_else(Self::default, Self)
    }
}

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
             their path.\n\
             - Give that answer once, at the end of the turn: no running commentary or \
             interim summaries between tool calls."
                .to_owned(),
        )
    }
}

/// Which registered tools the agent may call. Saved with the session.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
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
/// counts its retries in a [`Recovery`].
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = ActiveTurn)]
#[require(Recovery)]
pub struct TurnOf(pub Entity);

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
/// earlier call of the reply holds it back (see
/// [`Footprint`]), then runs as a
/// [`Running<ToolResult>`](super::calls::Running), or stays open for an
/// open tool, and ends with a [`ToolOutput`](super::tools::ToolOutput).
/// Read-only calls run side by side; every other call runs alone, in order.
#[derive(Component, Clone)]
pub struct ToolCallRun {
    /// The call.
    pub call: ToolCall,
    /// The effect id of the model call that asked for it; `None` for a
    /// call that a restart runs again.
    pub parent: Option<EffectId>,
    /// Whether it runs beside others, from its tool.
    pub(crate) footprint: Footprint,
}

/// A tool call waiting for earlier calls of its reply to finish.
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

/// Stop the agent's running turn.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Interrupt {
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

/// A line for the user, shown by views.
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

/// The agents, each with whether it is a subagent, for [`primary`].
pub type PrimaryQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static AgentId, Has<SpawnedBy>), With<Agent>>;

/// Where an agent sorts among the agents of a session: the ones the user
/// started first, each group by id. The first is the agent the user talks
/// to when none is named.
pub fn primary_order(spawned: bool, id: &AgentId) -> (bool, &str) {
    (spawned, id.0.as_str())
}

/// The agent the user talks to when none is named: the first one by
/// [`primary_order`].
pub fn primary(agents: &PrimaryQuery) -> Option<Entity> {
    agents
        .iter()
        .min_by(|a, b| primary_order(a.2, a.1).cmp(&primary_order(b.2, b.1)))
        .map(|(entity, ..)| entity)
}

/// An agent's turn ended, however its turn entity went away. Triggered on
/// the agent once it is idle. Not triggered for a turn the app's exit
/// stops: the restart carries that one on.
#[derive(EntityEvent, Clone, Debug)]
pub struct TurnEnded {
    /// The agent.
    pub entity: Entity,
    /// How the turn ended.
    pub outcome: TurnOutcome,
}
