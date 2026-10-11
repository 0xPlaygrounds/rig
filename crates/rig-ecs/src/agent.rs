//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{Message, Usage};
use rig_core::effect::EffectId;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

use super::inbox::Origin;
use super::journal::ReflectSaved;
use super::model::Effort;
use super::stage::moved;
use super::turn::{NO_MODEL, Recovery, start_turn};

/// Marks an agent. Spawning it adds every per-agent component with its
/// default, including a fresh [`AgentId`]. An agent has no
/// [`ModelChoice`](super::model::ModelChoice) until one is picked.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(
    AgentId,
    Conversation,
    Effort,
    LastUsage,
    SystemPrompt,
    ToolAccess,
    TurnsStarted
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

    /// The agent among `agents` whose id this is, such as in a
    /// `Query<(Entity, &AgentId)>`: ids name agents across a restart, where
    /// entities do not, as in an [`Origin`]'s `from`.
    pub fn find_in<'a>(
        &self,
        agents: impl IntoIterator<Item = (Entity, &'a Self)>,
    ) -> Option<Entity> {
        let mut agents = agents.into_iter();
        agents.find_map(|(agent, id)| (id == self).then_some(agent))
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
#[derive(Component, Reflect, Clone, Copy, Debug, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
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
    /// was delivered by an agent or a plugin, or is the user's with a title
    /// or a request id; `None` for the user's plain text, the model's
    /// messages and tool results.
    pub fn origin(&self, message: usize, item: usize) -> Option<&Origin> {
        self.origins
            .iter()
            .find(|(at, index, _)| *at == message && *index == item)
            .map(|(.., origin)| origin)
    }

    /// Why the user's last message is left unanswered, if it was halted:
    /// stopped by the user, or kept for the next turn. Cleared when a
    /// message is added or taken out.
    pub fn halted(&self) -> Option<Halt> {
        self.halted
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

    /// The `seq` of the agent log's record that began message `message`,
    /// if it was logged: where a fork of the session cuts it
    /// ([`fork_session`](super::journal::fork_session)).
    pub fn seq(&self, message: usize) -> Option<u64> {
        self.seqs.get(message).copied()
    }

    /// The first message whose log record is `seq` or later; the length
    /// when none is.
    pub(crate) fn at_seq(&self, seq: u64) -> usize {
        self.seqs.partition_point(|&begun| begun < seq)
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

    /// Takes out every message from `len` on; whether any was.
    pub(crate) fn truncate(&mut self, len: usize) -> bool {
        if len >= self.messages.len() {
            return false;
        }
        self.messages.truncate(len);
        self.halted = None;
        self.origins.retain(|(at, ..)| *at < len);
        self.seqs.truncate(len);
        true
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
    /// The messages a request sends: the summary, then the messages after
    /// the first `upto`. The summary goes into the first of those when that
    /// is the user's, so user and assistant messages still alternate.
    pub fn request(&self, messages: &[Message]) -> Vec<Message> {
        let mut live = messages.get(self.upto..).unwrap_or_default().to_vec();
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
/// the project's instructions and the environment. Empty, the agent has
/// none of its own, and the app's plugins may give it one in `Prepare`,
/// as rig-basics gives rig's. Saved with the session, as `null` when empty,
/// so a restored agent without its own gets what the build that restores
/// it gives.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(from = "Option<String>", into = "Option<String>")]
pub struct SystemPrompt(pub String);

impl From<SystemPrompt> for Option<String> {
    fn from(prompt: SystemPrompt) -> Self {
        (!prompt.0.is_empty()).then_some(prompt.0)
    }
}

impl From<Option<String>> for SystemPrompt {
    fn from(prompt: Option<String>) -> Self {
        Self(prompt.unwrap_or_default())
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
/// that ends it. At most one per agent. A turn ends with a [`TurnOutcome`]
/// on it: the kernel proposes one in Commit when the model answered or
/// the turn failed, Review may replace it, hold it ([`HoldOf`]) or carry
/// the turn on, and a plugin may insert `Failed` or `Stopped` at any
/// time. Once nothing holds the outcome and every call of the turn
/// settled, Commit ends the turn: it removes this, so the agent is idle,
/// and marks the turn [`Settled`](super::stage::Settled) for the readers
/// of how it ended. The kernel makes a turn [`ForAgent`] its agent when it
/// is spawned, so it is the agent's work, and goes when the agent goes,
/// even after it ended. The turn counts its retries in a [`Recovery`].
/// Its `Name` is `turn` unless it is spawned with another.
///
/// [`HoldOf`]: super::stage::HoldOf
/// [`ForAgent`]: super::stage::ForAgent
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = ActiveTurn)]
#[require(Recovery, Turn, Name = Name::new("turn"))]
pub struct TurnOf(pub Entity);

/// Where a turn stands, on the turn. The kernel fills it in when the turn
/// is spawned and keeps it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[component(on_add = start_turn)]
#[reflect(Component, Default)]
pub struct Turn {
    /// How many turns its agent started before this one, over the whole
    /// session: the agent's count is saved with it.
    pub index: u64,
    /// Where the turn begins in its agent's [`Conversation`]: the index of
    /// the user's message it answers, or, when the conversation does not
    /// end in one, of the first message the turn adds.
    pub first_message: usize,
    /// The model call whose reply's tool calls run now, their effects'
    /// parent, until their results are committed. Other calls of the turn
    /// keep it open but are never committed as tool results.
    #[reflect(ignore)]
    pub reply: Option<EffectId>,
}

/// How many turns the agent started, the next turn's [`Turn::index`].
/// Saved with the session, as `null` before the first.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(from = "Option<u64>", into = "Option<u64>")]
pub(crate) struct TurnsStarted(pub(crate) u64);

impl From<TurnsStarted> for Option<u64> {
    fn from(started: TurnsStarted) -> Self {
        (started.0 > 0).then_some(started.0)
    }
}

impl From<Option<u64>> for TurnsStarted {
    fn from(started: Option<u64>) -> Self {
        Self(started.unwrap_or_default())
    }
}

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
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct Partial {
    /// Answer text.
    pub text: String,
    /// Reasoning text.
    pub reasoning: String,
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

/// Stop the agent's running turn at once: it ends
/// [`TurnOutcome::Stopped`] with `reason`, the text its reply streamed so
/// far is kept, every call of the turn that has not settled is answered as
/// interrupted, and whatever held them is despawned. It cannot be reviewed
/// or held.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Interrupt {
    /// The agent.
    pub entity: Entity,
    /// Why, when it is not the user who stops it.
    pub reason: Option<String>,
}

impl Interrupt {
    /// The user stopping `agent`'s turn.
    pub fn new(agent: Entity) -> Self {
        Self {
            entity: agent,
            reason: None,
        }
    }
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
    fn new(agent: Option<Entity>, text: String, level: NoticeLevel) -> Self {
        Self { agent, text, level }
    }

    /// Information about `agent`, or about the whole app with `None`.
    pub fn info(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self::new(agent.into(), text.into(), NoticeLevel::Info)
    }

    /// A failure or refusal concerning `agent`, or the whole app with
    /// `None`.
    pub fn error(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self::new(agent.into(), text.into(), NoticeLevel::Error)
    }

    /// The refusal of what would change `agent`, or the whole app with
    /// `None`, while a turn runs: the rest of the turn would go on with
    /// something it did not start with.
    pub fn turn_running(agent: impl Into<Option<Entity>>) -> Self {
        Self::info(agent, "A turn is running; stop it first.")
    }

    /// The refusal of what needs a model when `agent` has none.
    pub fn no_model(agent: Entity) -> Self {
        Self::info(agent, NO_MODEL)
    }
}

/// How a turn ends, on the turn. The kernel proposes `Answered` or `Failed`
/// in Commit; in the next pass Review may replace it (`insert`), hold it
/// ([`HoldOf`](super::stage::HoldOf) the turn), or carry the turn on by
/// sending it a `Queue` or `Steer` message, which takes an `Answered`
/// proposal away, or by spawning its next
/// [`TurnCall`](super::turn::TurnCall), which takes any away once
/// admitted; the holds on a proposal taken away are despawned with it. A
/// plugin may insert `Failed` or `Stopped` at any time: the
/// turn's calls that have not started then never start, the others are
/// answered as stopped, and the turn ends once they settled. A turn that
/// ended is [`Settled`](super::stage::Settled) with it, until the next
/// frame: `On<Add<Settled>>` or `Added<Settled>` readers of
/// `(&TurnOutcome, &ForAgent)` see each turn's last outcome once.
#[derive(Component, Clone, Debug)]
#[component(on_insert = moved::<TurnOutcome>, on_remove = moved::<TurnOutcome>)]
pub enum TurnOutcome {
    /// The model answered: its last message, which ended the turn.
    Answered(Message),
    /// The turn failed, for the reason given.
    Failed(String),
    /// The turn was stopped before an answer: by the user when no reason
    /// is given, or on exit.
    Stopped(Option<String>),
}

impl TurnOutcome {
    /// What the calls of a turn that ends this way are told, when it
    /// stops them: `Failed` and `Stopped` do, `Answered` does not.
    pub(crate) fn stops(&self) -> Option<String> {
        match self {
            Self::Answered(_) => None,
            Self::Failed(why) => Some(format!("stopped: the turn failed: {why}")),
            Self::Stopped(None) => Some("stopped".to_owned()),
            Self::Stopped(Some(why)) => Some(format!("stopped: {why}")),
        }
    }
}
