//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use std::ops::{Deref, DerefMut};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{Message, Usage};
use rig_core::conversation::{self, MessageId, short_id};
use rig_core::effect::EffectId;
use rig_memory::Anchor;
use serde::{Deserialize, Serialize};

use super::model::ReasoningSetting;
use super::session_log::{ReflectSaved, save_discarded};
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
    ReasoningSetting,
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

impl From<AgentId> for String {
    fn from(id: AgentId) -> Self {
        id.0
    }
}

impl AgentId {
    /// The first characters of the id, enough to tell agents apart.
    pub fn short(&self) -> &str {
        short_id(&self.0)
    }

    /// The agent among `agents` whose id this is, such as in a
    /// `Query<(Entity, &AgentId)>`: ids name agents across a restart, where
    /// entities do not, as in an [`Author`](conversation::Author)'s `from`.
    pub fn find_in<'a>(
        &self,
        agents: impl IntoIterator<Item = (Entity, &'a Self)>,
    ) -> Option<Entity> {
        let mut agents = agents.into_iter();
        agents.find_map(|(agent, id)| (id == self).then_some(agent))
    }
}

/// The key the agent's model requests share a provider's prompt cache
/// under (`prompt_cache_key`); empty for the agent's [`AgentId`]. An agent
/// forked from another keeps the other's ([`ForkAgent`](super::fork::ForkAgent)),
/// so the fork's requests read what the source's wrote. Saved with the
/// session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// The key the requests of the agent `id`, with `key`, are sent under.
    pub fn of<'a>(key: Option<&'a Self>, id: &'a AgentId) -> &'a str {
        match key {
            Some(Self(key)) if !key.is_empty() => key,
            _ => &id.0,
        }
    }
}

/// The agent that spawned this one, such as the agent whose tool call
/// started it. Despawning that agent despawns this one, and stopping its
/// turn ([`StopTurn`]) stops this one's too: work meant to outlive its
/// parent is spawned without it. The kernel gives it no other meaning: a
/// restore links the agents again by their logs' headers, and views list
/// agents by it.
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

/// The agent's conversation: rig-core's
/// [`Conversation`](conversation::Conversation), which it derefs to, so a
/// plugin reads and changes it with that type's methods. Requests send it as
/// it is.
///
/// Each change waits in the conversation's outbox until the kernel saves it
/// and announces it as a
/// [`ConversationChanged`](super::session_log::ConversationChanged). A value
/// inserted in place of the agent's own is saved as one reset, after the
/// changes the old one had not saved. Saved through serde.
///
/// ```
/// use rig_ecs::prelude::*;
///
/// // `/undo`: the user's last message and everything after it go.
/// fn undo(mut agents: Query<&mut Conversation, With<Agent>>) {
///     for mut conversation in &mut agents {
///         let last = conversation.entries().rev().find(|entry| entry.is_users_own());
///         if let Some(last) = last.map(|entry| entry.index) {
///             conversation.splice(last.., [], Author::plugin("undo"));
///         }
///     }
/// }
/// # let _ = undo;
/// ```
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[component(on_discard = save_discarded)]
#[reflect(opaque, Component, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Conversation(pub conversation::Conversation);

impl Deref for Conversation {
    type Target = conversation::Conversation;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Conversation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<conversation::Conversation> for Conversation {
    fn from(conversation: conversation::Conversation) -> Self {
        Self(conversation)
    }
}

/// The usage of the agent's last model reply that reported its tokens,
/// with the last entry of the conversation it counts (rig-memory's
/// [`Anchor`]), `None` before one did. Only the kernel writes it, from
/// what the provider reported. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LastUsage(pub Option<Anchor>);

impl LastUsage {
    /// What the provider reported for the last reply.
    pub fn usage(&self) -> Option<&Usage> {
        self.0.as_ref().map(|anchor| &anchor.usage)
    }

    /// The tokens the last reply read and wrote, which the next request
    /// sends again, when it reported them.
    pub fn context(&self) -> Option<u64> {
        self.usage()?.context_tokens()
    }
}

/// The agent's own part of its system prompt: who it is and how it works.
/// Each request's prompt adds the app's
/// [`PromptSection`](super::prompt::PromptSection)s and the agent's own.
/// Empty, the agent has
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

/// A running turn of the agent it names, from a user message to the reply
/// that ends it; at most one per agent. It ends as its [`TurnOutcome`] says:
/// Commit then removes this, so the agent is idle, and marks the turn
/// [`Settled`](super::stage::Settled). The turn stays [`ForAgent`] its agent,
/// and goes with it. Its retries are counted in a [`Recovery`]; its `Name` is
/// `turn` unless spawned with another.
///
/// [`ForAgent`]: super::stage::ForAgent
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = ActiveTurn)]
#[require(Recovery, Turn, Name = Name::new("turn"))]
pub struct TurnOf(pub Entity);

/// Where a turn stands, on the turn. The kernel fills it in when the turn
/// is spawned, from the turn's record in its agent's conversation
/// ([`TurnRecord`](conversation::TurnRecord)), and keeps it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[component(on_add = start_turn)]
#[require(super::tools::MissingTools)]
#[reflect(Component, Default)]
pub struct Turn {
    /// How many turns its agent started before this one, over the whole
    /// session.
    pub index: u64,
    /// Where the turn begins in its agent's [`Conversation`]: the id of
    /// the user's message it answers, or, when the conversation does not
    /// end in one, of the first message the turn adds. Its index now is
    /// [`conversation::Conversation::start_of`] it, as splices move messages.
    /// The entries it answers are
    /// [`conversation::Conversation::asked`], the message that asked for it
    /// [`conversation::Conversation::request`].
    /// Not reflected: rig-core's types are not `Reflect`.
    #[reflect(ignore)]
    pub start: MessageId,
    /// The model call whose reply's tool calls run now, their effects'
    /// parent, until their results are committed. Other calls of the turn
    /// keep it open but are never committed as tool results.
    #[reflect(ignore)]
    pub reply: Option<EffectId>,
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

/// Text streamed so far by an in-flight model call, for views. A plugin
/// that continues a reply in a successor call may start the successor's
/// `Partial` with the text so far: views show one growing reply, and a
/// stop saves what the user saw.
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

/// Stop the agent's running turn at once, and the running turns of every
/// agent below it in [`Spawned`], even when the agent itself is idle:
/// stopping follows ancestry, so work meant to outlive its parent is
/// spawned without [`SpawnedBy`]. Each turn ends [`TurnOutcome::Stopped`]
/// with `reason`, the text its reply streamed so far is kept, every call of
/// it that has not settled is answered as stopped (rig-core's
/// [`not_finished`](rig_core::tool::feedback::not_finished)), and whatever
/// held them is despawned. It cannot be reviewed or held.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct StopTurn {
    /// The agent.
    pub entity: Entity,
    /// Why, when it is not the user who stops it.
    pub reason: Option<String>,
}

impl StopTurn {
    /// The user stopping `agent`'s turn.
    pub fn new(agent: Entity) -> Self {
        Self {
            entity: agent,
            reason: None,
        }
    }
}

/// How a [`Notice`] is shown and logged.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NoticeLevel {
    /// Information.
    #[default]
    Info,
    /// Something went wrong that did not stop anything, such as a refused
    /// key with another one used instead: shown like an error, but never a
    /// failure.
    Warn,
    /// Something failed or was refused.
    Error,
}

/// A line for the user, shown by views. A system reads the notices sent
/// before it in the frame and, from the frame before, those sent after it.
/// So one sent in `Last`, such as the session log's write failure (in
/// [`WriteLogs`](crate::WriteLogs)), is read a frame late and, on the exit
/// frame, only by a system ordered after its sender.
#[derive(Message, Clone, Debug)]
pub struct Notice {
    /// The agent it is about, or `None` when it is about the whole app.
    pub agent: Option<Entity>,
    /// The text.
    pub text: String,
    /// Whether it informs, warns or reports a failure.
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

    /// A warning about `agent`, or about the whole app with `None`:
    /// something went wrong, but nothing failed.
    pub fn warn(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self::new(agent.into(), text.into(), NoticeLevel::Warn)
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

/// How a turn ends, on the turn.
///
/// - **Proposed:** the kernel proposes `Answered` or `Failed` in Commit.
/// - **Reviewed:** in the next pass Review may edit it in place
///   (`Query<&mut TurnOutcome, With<TurnOf>>`) or hold it
///   ([`HoldOf`](super::stage::HoldOf) the turn). A `Queue` or `Steer`
///   message the agent accepts takes an `Answered` proposal away, and its
///   next [`TurnCall`](super::turn::TurnCall), once admitted, takes any;
///   the holds on a proposal taken away are despawned with it.
/// - **Stopped:** a plugin may insert `Failed` or `Stopped` at any time with
///   a plain `insert`, so no pending proposal loses it. The turn's calls not
///   started never start, the others are answered as stopped, and the turn
///   ends once they settled.
///
/// A turn that ended is [`Settled`](super::stage::Settled) with it until the
/// next frame: `On<Add<Settled>>` or `Added<Settled>` readers of
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
    /// Why the calls of a turn that ends this way did not finish, when it
    /// stops them: `Failed` and `Stopped` do, `Answered` does not. A tool
    /// call reads it through rig-core's
    /// [`not_finished`](rig_core::tool::feedback::not_finished).
    pub(crate) fn stops(&self) -> Option<String> {
        match self {
            Self::Answered(_) => None,
            Self::Failed(why) => Some(why.clone()),
            Self::Stopped(None) => Some("the user stopped the turn".to_owned()),
            Self::Stopped(Some(why)) => Some(why.clone()),
        }
    }
}
