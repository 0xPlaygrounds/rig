//! The agent entity: its components, the calls it owns, and the requests a
//! view sends it.

use std::ops::{Deref, DerefMut};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{AssistantContent, Message, Usage};
use rig_core::conversation::{self, MessageId, StopCause, short_id};
use rig_core::effect::EffectId;
use rig_core::message::{CallId, LocalCallId, ToolCall, ToolFunction, ToolName};
use rig_core::streaming::{Part, PartKind, StreamEvent, parse_partial_arguments};
use rig_memory::Anchor;
use serde::{Deserialize, Serialize};

use super::model::ReasoningSetting;
use super::session_log::{ReflectSaved, Reset, save_discarded};
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
/// what the provider reported. Saved with the session; a fork starts
/// without one, as its next request is its own.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Saved(Reset),
    Default,
    Clone,
    Serialize,
    Deserialize
)]
#[serde(transparent)]
pub struct LastUsage(pub Option<Anchor>);

impl LastUsage {
    /// What the provider reported for the last reply.
    pub fn usage(&self) -> Option<&Usage> {
        self.0.as_ref().map(|anchor| &anchor.usage)
    }

    /// The tokens the last reply read and wrote, which the next request
    /// sends again, when it reported them. Check [`ContextStale`] before
    /// displaying this figure as the current context.
    pub fn context(&self) -> Option<u64> {
        self.usage()?.context_tokens()
    }
}

/// The agent's [`LastUsage`] no longer counts its conversation after a
/// compaction summarized, cleared or restored part of it, or on a fork,
/// whose conversation no reply of its own measured. The kernel removes
/// this marker when a reply reports a new context figure. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct ContextStale {}

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

/// Which registered tools the agent may call: those it keeps, saved with
/// the session, within this run's tools ([`RunTools`]), which the session
/// never saves.
/// Only [`within`](Self::within), [`narrow`](Self::narrow),
/// [`grant`](Self::grant) and the kernel build a narrower one, so an
/// access made from another never has a tool that one lacks; before it
/// admits anything the kernel keeps each agent's access within the run's
/// tools and the access of every agent above it ([`SpawnedBy`]), whatever
/// made or restored it.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(from = "Kept", into = "Kept")]
pub struct ToolAccess {
    kept: Kept,
    /// This run's tools, which only the kernel sets.
    run: Option<Granted>,
}

/// What the session saves of a [`ToolAccess`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum Kept {
    /// Every registered tool.
    #[default]
    All,
    /// Only the tools named.
    Only(Granted),
}

impl From<Kept> for ToolAccess {
    fn from(kept: Kept) -> Self {
        Self { kept, run: None }
    }
}

impl From<ToolAccess> for Kept {
    fn from(access: ToolAccess) -> Self {
        access.kept
    }
}

/// Tool names, each once, in the order given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct Granted(Vec<String>);

impl Granted {
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|it| it == name)
    }

    /// These names that `allows` allows.
    fn within(&self, allows: impl Fn(&str) -> bool) -> Self {
        Self(self.0.iter().filter(|name| allows(name)).cloned().collect())
    }
}

/// The tools this run leaves every agent, such as `--tools` names: the
/// kernel keeps each agent's [`ToolAccess`] within them, and the session
/// never saves them, so a later run has every tool the session kept.
/// Without it, a run leaves every tool.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct RunTools(Granted);

impl RunTools {
    /// The tools `names`, each once.
    pub fn new<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        Self(Granted(unique(names)))
    }
}

/// Why [`ToolAccess::grant`] refused: `name` is not one of the tools the
/// granting access keeps, `yours`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotYours {
    /// The tool asked for.
    pub name: String,
    /// The registered tools the granting access keeps, sorted.
    pub yours: Vec<String>,
}

impl std::fmt::Display for NotYours {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (name, yours) = (&self.name, self.yours.join(", "));
        write!(f, "`{name}` is not one of your tools ({yours})")
    }
}

impl std::error::Error for NotYours {}

impl ToolAccess {
    /// Every registered tool.
    pub fn all() -> Self {
        Self::default()
    }

    /// Whether the tool `name` may be called in this run. A child's tools
    /// are built with [`keeps`](Self::keeps), so they outlive the run.
    pub fn allows(&self, name: &str) -> bool {
        self.keeps(name) && self.run.as_ref().is_none_or(|run| run.has(name))
    }

    /// Whether the agent keeps the tool `name`, whatever this run's tools:
    /// what the session saves, and what a child it starts may be given.
    pub fn keeps(&self, name: &str) -> bool {
        match &self.kept {
            Kept::All => true,
            Kept::Only(granted) => granted.has(name),
        }
    }

    /// The names it keeps, in the order granted; `None` for every tool.
    pub fn names(&self) -> Option<&[String]> {
        match &self.kept {
            Kept::All => None,
            Kept::Only(granted) => Some(&granted.0),
        }
    }

    /// This access kept within `other`: the tools both allow, in this
    /// one's order, or `other`'s when this one allows every tool.
    pub fn within(&self, other: &Self) -> Self {
        let kept = match (&self.kept, &other.kept) {
            (_, Kept::All) => self.kept.clone(),
            (Kept::All, Kept::Only(_)) => other.kept.clone(),
            (Kept::Only(mine), Kept::Only(theirs)) => Kept::Only(mine.within(|it| theirs.has(it))),
        };
        let run = match (&self.run, &other.run) {
            (Some(mine), Some(theirs)) => Some(mine.within(|it| theirs.has(it))),
            (mine, theirs) => mine.as_ref().or(theirs.as_ref()).cloned(),
        };
        Self { kept, run }
    }

    /// This access kept to `names`, each once: never more than it was, such
    /// as for an agent profile.
    pub fn narrow<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Self {
        self.within(&Kept::Only(Granted(unique(names))).into())
    }

    /// A child's access: the tools `asked`, each once, every one a tool of
    /// `registered` this access keeps; or why not, for the first that is
    /// not. The child keeps them and, as every agent, uses them within this
    /// run's tools, so a later run gives it what it was given.
    pub fn grant<'a>(
        &self,
        asked: &[String],
        registered: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, NotYours> {
        let mut yours: Vec<&str> = registered
            .into_iter()
            .filter(|name| self.keeps(name))
            .collect();
        yours.sort_unstable();
        yours.dedup();
        match asked.iter().find(|name| !yours.contains(&name.as_str())) {
            Some(name) => Err(NotYours {
                name: name.clone(),
                yours: yours.into_iter().map(str::to_owned).collect(),
            }),
            None => Ok(Self {
                kept: Kept::Only(Granted(unique(asked.iter().map(String::as_str)))),
                run: self.run.clone(),
            }),
        }
    }

    /// This access within the run's tools `run`, the kernel's to set.
    pub(crate) fn for_run(&self, run: Option<&RunTools>) -> Self {
        let run = run.map(|run| run.0.clone());
        Self {
            kept: self.kept.clone(),
            run,
        }
    }
}

/// `names` in their order, each once.
fn unique<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut unique: Vec<String> = Vec::new();
    for name in names {
        if !unique.iter().any(|kept| kept == name) {
            unique.push(name.to_owned());
        }
    }
    unique
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

/// What an in-flight model call streamed so far: its text, reasoning and
/// tool calls, the one record of the reply in flight that views, the
/// status and a stop read. The kernel follows the stream into it
/// ([`Partial::follow`]). A plugin that continues a reply in a successor
/// call may start the successor's `Partial` with what came so far: views
/// show one growing reply, and a stop saves what the user saw.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct Partial {
    /// Answer text.
    pub text: String,
    /// Reasoning text.
    pub reasoning: String,
    /// The tool calls the reply opened, in the order they opened.
    #[reflect(ignore)]
    pub calls: Vec<PartialCall>,
}

impl Partial {
    /// Adds what `event` streamed: a text or reasoning fragment, a tool
    /// call opening under its tool's name, a fragment of its arguments, or
    /// its end with the finished call.
    pub fn follow(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Text { text, .. } => self.text.push_str(text),
            StreamEvent::Reasoning { text, .. } => self.reasoning.push_str(text),
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name: Some(name),
            } => self.calls.push(PartialCall::new(*part, name.clone())),
            StreamEvent::Arguments { part, json } => {
                let call = self.calls.iter_mut().find(|call| call.streams_in(*part));
                if let Some((_, arguments)) = call.and_then(|call| call.streaming.as_mut()) {
                    arguments.push_str(json);
                }
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(done),
            } => {
                if let Some(call) = self.calls.iter_mut().find(|call| call.streams_in(*part)) {
                    call.call = done.clone();
                    call.streaming = None;
                }
            }
            _ => {}
        }
    }

    /// What a stop keeps of the reply, as an aborted message's content: the
    /// reasoning and text unless blank, then each tool call as
    /// [`PartialCall::stopped`] keeps it. None of the calls ever runs.
    pub fn stopped(&self) -> Vec<AssistantContent> {
        let reasoning = (!self.reasoning.trim().is_empty())
            .then(|| AssistantContent::reasoning(self.reasoning.clone()));
        let text =
            (!self.text.trim().is_empty()).then(|| AssistantContent::text(self.text.clone()));
        let calls = self.calls.iter().map(PartialCall::stopped);
        reasoning
            .into_iter()
            .chain(text)
            .chain(calls.map(AssistantContent::ToolCall))
            .collect()
    }
}

/// A tool call of a reply in flight ([`Partial::calls`]). One a plugin
/// carries over from an earlier reply, such as a continuation's, is
/// `PartialCall::from` the finished call.
#[derive(Clone, Debug)]
pub struct PartialCall {
    /// While it streams, its tool's name, no arguments and an id rig
    /// issued, as the provider's comes with its end; once it ended, the
    /// finished call.
    pub call: ToolCall,
    /// While it streams, the part of the reply it streams in and its
    /// argument JSON so far, as the provider sent it; `None` once it ended.
    pub streaming: Option<(Part, String)>,
}

impl PartialCall {
    /// A call to `name` that opened in `part`, with no arguments yet.
    pub fn new(part: Part, name: ToolName) -> Self {
        let id = CallId::Local(LocalCallId::new());
        let function = ToolFunction::new(name, serde_json::Value::Null);
        Self {
            call: ToolCall::new(id, function),
            streaming: Some((part, String::new())),
        }
    }

    /// Whether it is still streaming in `part`.
    fn streams_in(&self, part: Part) -> bool {
        self.streaming.as_ref().is_some_and(|(at, _)| *at == part)
    }

    /// The call as a view shows it: while it streams, with the arguments
    /// that its JSON so far states. For display only.
    pub fn shown(&self) -> ToolCall {
        let mut call = self.call.clone();
        if let Some((_, arguments)) = &self.streaming {
            call.function.arguments = parse_partial_arguments(arguments);
        }
        call
    }

    /// The call as a stop keeps it, never run. While it streams, its
    /// arguments are what [`ToolFunction::parse`] reads from the JSON so
    /// far, and the text as it came goes in `invalid_arguments` unless it is
    /// a whole object, also when none came, so nothing is invented. Once it
    /// ended, it is as it ended.
    pub fn stopped(&self) -> ToolCall {
        let mut call = self.call.clone();
        if let Some((_, arguments)) = &self.streaming {
            call.function = ToolFunction::parse(call.function.name.clone(), arguments);
            if arguments.trim().is_empty() {
                call.function.invalid_arguments = Some(arguments.clone());
            }
        }
        call
    }
}

impl From<ToolCall> for PartialCall {
    fn from(call: ToolCall) -> Self {
        Self {
            call,
            streaming: None,
        }
    }
}

/// Stop the agent's running turn at once, and the running turns of every
/// agent below it in [`Spawned`], even when the agent itself is idle:
/// stopping follows ancestry, so work meant to outlive its parent is
/// spawned without [`SpawnedBy`]; each agent it reached then gets a
/// [`StopReached`]. [`plugin_alone`](Self::plugin_alone) stops the agent's
/// turn only and sends none. Each turn ends [`TurnOutcome::Stopped`] with
/// the stop's cause, the text its reply streamed so far is kept, every
/// call of it that has not settled is answered as stopped (rig-core's
/// [`not_finished`](rig_core::tool::feedback::not_finished)), and whatever
/// held them is despawned. It cannot be reviewed or held.
///
/// Its fields are private, so the cause is the user's or a plugin's:
///
/// ```compile_fail,E0451
/// use bevy_ecs::prelude::*;
/// use rig_ecs::agent::StopTurn;
/// use rig_ecs::prelude::StopCause;
///
/// let agent = World::new().spawn_empty().id();
/// let stop = StopTurn { entity: agent, cause: StopCause::User, below: true };
/// ```
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(opaque, Event, Clone, Debug)]
pub struct StopTurn {
    entity: Entity,
    pub(crate) cause: StopCause,
    /// Whether the stop reaches the agents below the agent.
    pub(crate) below: bool,
}

impl StopTurn {
    /// The user stopping `agent`'s turn, and those below it.
    pub fn new(agent: Entity) -> Self {
        Self {
            entity: agent,
            cause: StopCause::User,
            below: true,
        }
    }

    /// The plugin `by` stopping `agent`'s turn, and those below it, for
    /// `why`. The request the turn answers is halted
    /// [`Kept`](conversation::Halt::Kept), left for the next turn.
    pub fn plugin(agent: Entity, by: &str, why: &str) -> Self {
        Self {
            entity: agent,
            cause: StopCause::Plugin {
                by: by.to_owned(),
                why: why.to_owned(),
            },
            below: true,
        }
    }

    /// The plugin `by` ending only `agent`'s own turn, for `why`, as its
    /// decision about that turn: the agents below go on, and no
    /// [`StopReached`] is sent. The user's stop always reaches below:
    ///
    /// ```compile_fail,E0599
    /// use bevy_ecs::prelude::*;
    /// use rig_ecs::agent::StopTurn;
    ///
    /// let agent = World::new().spawn_empty().id();
    /// let stop = StopTurn::new(agent).alone();
    /// ```
    pub fn plugin_alone(agent: Entity, by: &str, why: &str) -> Self {
        Self {
            below: false,
            ..Self::plugin(agent, by, why)
        }
    }
}

/// On each agent a [`StopTurn`] reached, idle or not, once, after its turn
/// is gone: the stop's cause. Only the kernel makes it, so an observer
/// reads one per agent instead of walking [`Spawned`] itself. A
/// [`StopTurn::plugin_alone`] sends none, and neither do the stops of the
/// app's exit and of a restart.
///
/// ```compile_fail,E0451
/// use bevy_ecs::prelude::*;
/// use rig_ecs::agent::StopReached;
/// use rig_ecs::prelude::StopCause;
///
/// let agent = World::new().spawn_empty().id();
/// let reached = StopReached { entity: agent, cause: StopCause::User, idle: true };
/// ```
#[derive(EntityEvent, Clone, Debug)]
pub struct StopReached {
    pub(crate) entity: Entity,
    pub(crate) cause: StopCause,
    pub(crate) idle: bool,
}

impl StopReached {
    /// Who or what stopped it.
    pub fn cause(&self) -> &StopCause {
        &self.cause
    }

    /// Whether the agent had no running turn for the stop to end. The
    /// stopped turn is gone when this is sent, so [`ActiveTurn`] cannot say.
    pub fn idle(&self) -> bool {
        self.idle
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
/// - **Failed:** a plugin may insert `Failed` at any time with a plain
///   `insert`, so no pending proposal loses it. The turn's calls not
///   started never start, the others are answered as stopped, and the turn
///   ends once they settled.
/// - **Stopped:** only a [`StopTurn`], an exit or a restart stops a turn,
///   at once ([`Stop`]).
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
    /// The turn was stopped before an answer.
    Stopped(Stop),
}

/// How a turn was stopped. Only the kernel makes one, as it stops the
/// turn, so a plugin stops a turn with a [`StopTurn`]:
///
/// ```compile_fail,E0451
/// use rig_ecs::agent::{Stop, TurnOutcome};
/// use rig_ecs::prelude::StopCause;
///
/// let stopped = TurnOutcome::Stopped(Stop { cause: StopCause::User, alone: false });
/// ```
#[derive(Clone, Debug)]
pub struct Stop {
    pub(crate) cause: StopCause,
    pub(crate) alone: bool,
}

impl Stop {
    /// Who or what stopped the turn.
    pub fn cause(&self) -> &StopCause {
        &self.cause
    }

    /// Whether a plugin ended only this turn, as its decision about it
    /// ([`StopTurn::plugin_alone`]): no stop reached the agent. Never so
    /// for the user's stop, the app's exit or a restart.
    pub fn alone(&self) -> bool {
        self.alone
    }
}

impl TurnOutcome {
    /// Why the calls of a turn that ends this way did not finish, when it
    /// stops them: `Failed` and `Stopped` do, `Answered` does not. A tool
    /// call reads it through rig-core's
    /// [`not_finished`](rig_core::tool::feedback::not_finished).
    pub(crate) fn stops(&self) -> Option<String> {
        match self {
            Self::Answered(_) => None,
            Self::Failed(why) => Some(format!("the turn failed: {why}")),
            Self::Stopped(stop) => Some(stop.cause.to_string()),
        }
    }
}

#[cfg(test)]
mod tests;
