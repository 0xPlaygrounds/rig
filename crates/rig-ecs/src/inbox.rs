//! Messages to agents. A message, whoever sends it (the user typing, an
//! agent reporting, a plugin), is an effect entity: an [`Incoming`]
//! [`ForAgent`] the agent, which moves through the stages like any other
//! effect. [`Deliver`] spawns one. Plugins edit it in Prepare
//! (`Added<Admitted>`), such as adding the files its text names as
//! [`Attachment`]s, and take it over in Gate by answering it with an
//! [`Outcome`]: it never reaches the agent then. At Start the kernel
//! accepts what is left and marks it [`Started`]:
//! - a user's text that starts with `/` runs as a slash command
//!   ([`Incoming::command`]) instead of being sent;
//! - an idle agent takes it into its conversation at once and starts a
//!   turn with it, and every message accepted with it goes to the model
//!   in that turn's first call; a [`DeliveryMode::Note`] starts no turn
//!   and is kept in the conversation, logged as halted, for the next one;
//! - a busy agent keeps it: a [`DeliveryMode::Steer`] message goes to the
//!   model with the turn's next call, after the tool results it waits
//!   for; the [`DeliveryMode::Queue`] ones carry the turn on together, as
//!   one step, once it would end; a note goes with whatever call comes
//!   next and never starts or carries on a turn by itself. A Steer or
//!   Queue message accepted while the turn's proposed outcome is
//!   [`TurnOutcome::Answered`] carries the turn on at once, and one still
//!   on its way keeps such a turn from ending until it is accepted or
//!   taken over.
//!
//! An agent's inbox is the messages it accepted that are not answered
//! yet: `Query<(&Incoming, &ForAgent), (With<Started>, Without<Outcome>)>`.
//! The kernel answers a message once it delivered it, ran it as a
//! command, or handed it back, and it settles. A turn that ends some other
//! way (stopped, failed) hands what the user typed and was not sent back
//! to the views as [`Recalled`], so nothing typed is lost or sent unasked,
//! and puts what agents and plugins sent in the conversation for the next
//! turn.
//!
//! Each message carries its [`Origin`], kept with the conversation and its
//! log. Text that is not the user's own goes to the model as a user
//! message headed by a line naming where it came from. A message may carry
//! [`Attachment`]s its sender read, such as images, which go before its
//! text when the agent's model reads them.

use std::collections::HashSet;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::effect::{EffectKind, HandlerKey};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};
use web_time::SystemTime;

use super::agent::{
    ActiveTurn, Agent, AgentId, Conversation, EffectParent, Halt, Notice, TurnOf, TurnOutcome,
};
use super::calls::Wake;
use super::commands::RunCommand;
use super::journal::Commit;
use super::model::Connection;
use super::stage::{
    AgentWork, Effect, ForAgent, Holds, Outcome, Settled, Stage, Started, Undecided, failed_in,
};
use super::turn::{Exiting, spawn_turn_call, take_outcome};

/// The id of a request to an agent, which the reply to it names, such as
/// the tool call that asked. Unique per sender, so it doubles as an
/// idempotency key.
#[derive(Reflect, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub String);

/// Who sent a message.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginKind {
    /// The user, typing.
    #[default]
    User,
    /// An agent, such as a subagent answering or an agent handing out a
    /// task.
    Agent,
    /// The plugin named.
    Plugin(String),
}

/// Where a message came from: who sent it, the agent it came from, and the
/// request it answers or makes.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// Who sent it.
    #[serde(default)]
    pub kind: OriginKind,
    /// The agent it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<AgentId>,
    /// The request it answers or makes. Kept for correlation, such as a
    /// report to the call that asked; not shown in its header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestId>,
    /// What it is about, such as a subagent's task title, shown in its
    /// header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Origin {
    /// The user's own text.
    pub fn user() -> Self {
        Self::default()
    }

    /// Output of the agent `from`.
    pub fn agent(from: AgentId, request: Option<RequestId>) -> Self {
        Self {
            kind: OriginKind::Agent,
            from: Some(from),
            request,
            title: None,
        }
    }

    /// Output of the plugin `name`.
    pub fn plugin(name: impl Into<String>) -> Self {
        Self {
            kind: OriginKind::Plugin(name.into()),
            ..Self::default()
        }
    }

    /// This origin about `title`, such as a task's.
    pub fn titled(self, title: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            ..self
        }
    }

    /// Who sent it, for views: such as `agent 1a2b3c4d "Fix the parser"`.
    /// The request id is left out; it stays in [`Origin::request`].
    pub fn label(&self) -> String {
        let who = match &self.kind {
            OriginKind::User => "you".to_owned(),
            OriginKind::Agent => {
                format!("agent {}", self.from.as_ref().map_or("?", AgentId::short))
            }
            OriginKind::Plugin(name) => format!("plugin {name}"),
        };
        match &self.title {
            Some(title) => format!("{who} \"{title}\""),
            None => who,
        }
    }

    /// The line that heads text of this origin for the model, or `None`
    /// for the user's own: it names the sender and says the text is not
    /// the user's words.
    pub fn header(&self) -> Option<String> {
        match self.kind {
            OriginKind::User => None,
            _ => Some(format!(
                "[Output of {}, not the user's words]",
                self.label()
            )),
        }
    }
}

/// When a message goes to the agent's model, and whether it asks for an
/// answer. An idle agent starts a turn with a [`Steer`](Self::Steer) or
/// [`Queue`](Self::Queue) message either way.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    /// With the running turn's next model call.
    #[default]
    Steer,
    /// Once the running turn would end, as its next step: everything
    /// queued meanwhile goes in that one step.
    Queue,
    /// Needs no answer, such as a report that only points at another: it
    /// goes with the agent's next model call, whenever one is made, and
    /// never makes one. An idle agent keeps it in its conversation for
    /// the next turn.
    Note,
}

/// Content read by a message's sender that goes before its text, such as
/// an image file the user named. An image goes only to a model that reads
/// images; another model gets the text alone, and the user is told.
#[derive(Clone, Debug)]
pub struct Attachment {
    /// How the user knows it, such as its path.
    pub label: String,
    /// The content.
    pub content: UserContent,
}

/// A message to the agent it is [`ForAgent`]: `text`, from `origin`, that
/// goes to the agent's model as `mode` says, after its `attachments`. An
/// effect that moves through the stages (see the [module](self)): edit it
/// in Prepare, take it over in Gate by answering it with an [`Outcome`];
/// once accepted, it is [`Started`] until the kernel answers it. Empty
/// text is ignored.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
#[require(Effect, Name = Name::new("message"))]
pub struct Incoming {
    /// The text.
    pub text: String,
    /// Where it came from.
    pub origin: Origin,
    /// When it goes to a busy agent's model.
    pub mode: DeliveryMode,
    /// What goes before the text.
    #[reflect(ignore)]
    pub attachments: Vec<Attachment>,
    /// Whether the user's text goes to the model as it is even when it
    /// starts with `/`, such as a refused command sent again.
    pub literal: bool,
}

impl Incoming {
    /// `text`, as `mode` says, as the user's own text unless
    /// [`with_origin`](Self::with_origin) names who sent it.
    pub fn new(text: impl Into<String>, mode: DeliveryMode) -> Self {
        Self {
            text: text.into(),
            origin: Origin::user(),
            mode,
            attachments: Vec::new(),
            literal: false,
        }
    }

    /// This message, from `origin`.
    #[must_use]
    pub fn with_origin(self, origin: Origin) -> Self {
        Self { origin, ..self }
    }

    /// This message sent [`literal`](Self::literal)ly.
    #[must_use]
    pub fn literal(self) -> Self {
        Self {
            literal: true,
            ..self
        }
    }

    /// The slash command line, without its `/`, that this message runs
    /// instead of being sent: the user's text that starts with `/`, unless
    /// it is [`literal`](Self::literal).
    pub fn command(&self) -> Option<&str> {
        let user = self.origin.kind == OriginKind::User && !self.literal;
        user.then(|| self.text.trim_start().strip_prefix('/'))
            .flatten()
    }
}

/// Send `message` to the agent `entity`: the kernel spawns it as an
/// [`Incoming`] effect for that agent, under the effect the agent's work
/// is recorded under ([`EffectParent`]), and the stages take it from there
/// (see the [module](self)). A plugin changes or takes over a message
/// through its `Incoming`, not this.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Deliver {
    /// The agent.
    pub entity: Entity,
    /// The message.
    pub message: Incoming,
}

impl Deliver {
    /// `text` for `agent`, as `mode` says, as the user's own text unless
    /// [`with_origin`](Self::with_origin) names who sent it.
    pub fn new(agent: Entity, text: impl Into<String>, mode: DeliveryMode) -> Self {
        Self {
            entity: agent,
            message: Incoming::new(text, mode),
        }
    }

    /// This message, from `origin`.
    #[must_use]
    pub fn with_origin(self, origin: Origin) -> Self {
        Self {
            message: self.message.with_origin(origin),
            ..self
        }
    }

    /// This message sent [`literal`](Incoming::literal)ly.
    #[must_use]
    pub fn literal(self) -> Self {
        Self {
            message: self.message.literal(),
            ..self
        }
    }
}

/// Text the user typed that was not sent: the messages the agent's turn
/// ended without sending, joined in the order they were typed, or a
/// refused slash command, with `why`. A view puts it back in its input.
#[derive(Message, Clone, Debug)]
pub struct Recalled {
    /// The agent.
    pub agent: Entity,
    /// The text.
    pub text: String,
    /// Why a command was refused.
    pub why: Option<String>,
}

/// Spawns a [`Deliver`]'s message for its agent, and wakes the loop: the
/// frame that sent it may have run its stages already.
pub(crate) fn on_deliver(
    deliver: On<Deliver>,
    agents: Query<Option<&EffectParent>, With<Agent>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let agent = deliver.entity;
    let Ok(parent) = agents.get(agent) else {
        return;
    };
    commands.spawn((
        deliver.message.clone(),
        ForAgent(agent),
        Effect::child_of(parent.map(|parent| parent.0)),
    ));
    wake.wake();
}

/// How the kernel answered a message it took: `delivered`, `command`,
/// `recalled` or `empty`.
fn handled(how: &'static str) -> Outcome {
    Outcome::custom(how)
}

/// Accepts each message about to start, in the order they were sent,
/// unless a Gate system failed this pass, in [`Stage::Start`]: runs a
/// user's slash command; gives an idle agent the message at once and
/// starts a turn with it, with every message accepted with it, unless it
/// is a note, which is kept for the next turn; and keeps it, [`Started`],
/// for a busy agent.
pub(crate) fn accept(
    ready: Query<(Entity, &Effect, &Incoming, Option<&ForAgent>), Undecided>,
    mut agents: Query<(&mut Conversation, Option<&Connection>, Has<ActiveTurn>), With<Agent>>,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if failed_in(Stage::Gate).is_some() {
        return;
    }
    let mut ready: Vec<_> = ready.iter().collect();
    ready.sort_unstable_by_key(|(_, effect, ..)| effect.id());
    let at = SystemTime::now();
    // The agents whose turn starts with this pass's messages.
    let mut starting = HashSet::new();
    for (entity, _, message, agent) in ready {
        let agent = agent.map(|agent| agent.0);
        let Some((agent, (mut conversation, connection, busy))) =
            agent.and_then(|agent| Some((agent, agents.get_mut(agent).ok()?)))
        else {
            let why = ErrorReport::new(ErrorKind::Internal, "a message needs an agent");
            commands.entity(entity).try_insert_if_new(Outcome(Err(why)));
            continue;
        };
        if let Some(line) = message.command() {
            let line = line.to_owned();
            commands.trigger(RunCommand {
                entity: agent,
                line,
            });
            commands
                .entity(entity)
                .try_insert_if_new(handled("command"));
            continue;
        }
        if message.text.trim().is_empty() {
            commands.entity(entity).try_insert_if_new(handled("empty"));
            continue;
        }
        let first_call = starting.contains(&agent);
        if busy && !first_call {
            commands.entity(entity).try_insert(Started { at });
            continue;
        }
        let spec = connection.map(|connection| &*connection.spec);
        // After a failure that kept the user's message, the new text joins it.
        let conversation = &mut *conversation;
        commit.pending(agent, spec, [message], conversation, &mut notices);
        commands
            .entity(entity)
            .try_insert((Started { at }, handled("delivered")));
        if first_call {
            continue;
        }
        if message.mode == DeliveryMode::Note {
            commit.halt(agent, conversation, Halt::Kept);
            continue;
        }
        starting.insert(agent);
        let turn = commands.spawn(TurnOf(agent)).id();
        spawn_turn_call(&mut commands, turn, agent);
    }
}

/// Carries on each turn proposed [`TurnOutcome::Answered`] whose agent
/// accepted a Steer or Queue message this pass, in [`Stage::Start`]: the
/// outcome is taken off, what carries the turn on goes into the
/// conversation, and the model is called again.
pub(crate) fn carry_on_accepted(
    accepted: Query<(&Incoming, &ForAgent), (Added<Started>, Without<Outcome>)>,
    mut agents: Query<(&mut Conversation, Option<&Connection>, &ActiveTurn)>,
    turns: Query<(&TurnOutcome, Option<&Holds>)>,
    inboxes: Inboxes,
    mut commit: Commit,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut carried = HashSet::new();
    for (message, &ForAgent(agent)) in &accepted {
        if message.mode == DeliveryMode::Note || !carried.insert(agent) {
            continue;
        }
        let Ok((mut conversation, connection, active)) = agents.get_mut(agent) else {
            continue;
        };
        let turn = active.turn();
        let Ok((TurnOutcome::Answered(_), holds)) = turns.get(turn) else {
            continue;
        };
        let Some(next) = carry_on(inboxes.of(agent)) else {
            continue;
        };
        let spec = connection.map(|connection| &*connection.spec);
        let conversation = &mut *conversation;
        commit.deliver(agent, spec, next, conversation, &mut notices, &mut commands);
        take_outcome(&mut commands, turn, holds);
        spawn_turn_call(&mut commands, turn, agent);
    }
}

/// The messages agents accepted and were not given yet, as the kernel
/// reads them.
#[derive(SystemParam)]
pub(crate) struct Inboxes<'w, 's> {
    work: Query<'w, 's, &'static AgentWork>,
    accepted: Query<
        'w,
        's,
        (Entity, &'static Effect, &'static Incoming),
        (With<Started>, Without<Outcome>, Without<Settled>),
    >,
    on_the_way: Query<'w, 's, &'static Incoming, (Without<Started>, Without<Outcome>)>,
}

impl Inboxes<'_, '_> {
    /// What `agent` accepted and was not given yet, oldest first.
    pub(crate) fn of(&self, agent: Entity) -> Vec<(Entity, &Incoming)> {
        let work = self.work.get(agent).into_iter().flat_map(AgentWork::iter);
        let mut waiting: Vec<_> = self.accepted.iter_many(work).flatten().collect();
        waiting.sort_unstable_by_key(|(_, effect, _)| effect.id());
        waiting
            .into_iter()
            .map(|(entity, _, message)| (entity, message))
            .collect()
    }

    /// Whether a Steer or Queue message to `agent` is on its way: spawned,
    /// not accepted, not answered.
    pub(crate) fn coming(&self, agent: Entity) -> bool {
        let work = self.work.get(agent).into_iter().flat_map(AgentWork::iter);
        self.on_the_way
            .iter_many(work)
            .flatten()
            .any(|message| message.mode != DeliveryMode::Note)
    }
}

/// What of `waiting` carries a turn on that would end: the notes with the
/// steering messages, else with everything queued, in the order they came;
/// none when neither waits.
pub(crate) fn carry_on(waiting: Vec<(Entity, &Incoming)>) -> Option<Vec<(Entity, &Incoming)>> {
    let steering = waiting
        .iter()
        .any(|(_, message)| message.mode == DeliveryMode::Steer);
    let carries = match steering {
        true => DeliveryMode::Steer,
        false => DeliveryMode::Queue,
    };
    let next: Vec<_> = waiting
        .into_iter()
        .filter(|(_, message)| {
            matches!(message.mode, DeliveryMode::Note) || message.mode == carries
        })
        .collect();
    next.iter()
        .any(|(_, message)| message.mode == carries)
        .then_some(next)
}

/// Hands back what the user typed that a turn that just ended did not
/// send, and puts what agents and plugins sent in the conversation, where
/// the next turn reads it. A conversation left ending in the user's
/// message is logged as halted, so a restore does not answer it, unless
/// the app is [`Exiting`].
pub(crate) fn recall_on_turn_end(
    end: On<Remove<ActiveTurn>>,
    mut agents: Query<(&mut Conversation, Option<&Connection>)>,
    inboxes: Inboxes,
    mut commit: Commit,
    exiting: Option<Res<Exiting>>,
    mut commands: Commands,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = end.entity;
    let Ok((mut conversation, connection)) = agents.get_mut(agent) else {
        return;
    };
    let (typed, sent): (Vec<_>, Vec<_>) = inboxes
        .of(agent)
        .into_iter()
        .partition(|(_, message)| message.origin.kind == OriginKind::User);
    let spec = connection.map(|connection| &*connection.spec);
    let conversation = &mut *conversation;
    commit.deliver(agent, spec, sent, conversation, &mut notices, &mut commands);
    if exiting.is_none() {
        commit.halt(agent, conversation, Halt::Kept);
    }
    if typed.is_empty() {
        return;
    }
    for &(entity, _) in &typed {
        commands
            .entity(entity)
            .try_insert_if_new(handled("recalled"));
    }
    let typed: Vec<&str> = typed
        .iter()
        .map(|(_, message)| message.text.trim())
        .collect();
    recalled.write(Recalled {
        agent,
        text: typed.join("\n\n"),
        why: None,
    });
}

/// The record's handler key and kind of a message: its text, origin and
/// mode, without its attachments.
pub(crate) fn message_kind(message: &Incoming) -> (HandlerKey, EffectKind) {
    let mut payload = serde_json::Map::new();
    payload.insert("text".to_owned(), message.text.clone().into());
    let origin = serde_json::to_value(&message.origin).unwrap_or_default();
    payload.insert("origin".to_owned(), origin);
    let mode = serde_json::to_value(message.mode).unwrap_or_default();
    payload.insert("mode".to_owned(), mode);
    let kind = EffectKind::Custom {
        kind: Arc::from("message"),
        payload: payload.into(),
    };
    (HandlerKey::new("message"), kind)
}

impl Commit<'_, '_> {
    /// Commits each of `messages`, accepted by `agent`, to its
    /// `conversation` as [`pending`](Self::pending) does, and answers it
    /// as delivered.
    pub(crate) fn deliver<'a>(
        &mut self,
        agent: Entity,
        spec: Option<&ModelSpec>,
        messages: impl IntoIterator<Item = (Entity, &'a Incoming)>,
        conversation: &mut Conversation,
        notices: &mut MessageWriter<Notice>,
        commands: &mut Commands,
    ) {
        for (entity, message) in messages {
            self.pending(agent, spec, [message], conversation, notices);
            commands
                .entity(entity)
                .try_insert_if_new(handled("delivered"));
        }
    }

    /// Commits each of `messages` to `conversation`, that of `agent`, whose
    /// model is `spec`, as a user message with its origin: its attachments
    /// the model takes, then its text, trimmed, headed by its origin's line
    /// when it is not the user's own. Each goes into the last message when
    /// that is the user's, such as the tool results the model waits for, so
    /// user and model keep taking turns.
    pub(crate) fn pending<'a>(
        &mut self,
        agent: Entity,
        spec: Option<&ModelSpec>,
        messages: impl IntoIterator<Item = &'a Incoming>,
        conversation: &mut Conversation,
        notices: &mut MessageWriter<Notice>,
    ) {
        for message in messages {
            let mut content = Vec::with_capacity(message.attachments.len() + 1);
            for Attachment {
                label,
                content: item,
            } in &message.attachments
            {
                let refused = match (item, spec) {
                    (UserContent::Image(_), Some(spec)) if !spec.input.image => Some(format!(
                        "{} does not read images, so {label} is sent as its name only.",
                        spec.display_name
                    )),
                    (UserContent::Image(_), None) => Some(format!(
                        "No model is connected, so {label} is sent as its name only."
                    )),
                    _ => None,
                };
                match refused {
                    Some(note) => {
                        notices.write(Notice::info(agent, note));
                    }
                    None => content.push(item.clone()),
                }
            }
            let text = message.text.trim();
            let (text, origin) = match message.origin.header() {
                None => (text.to_owned(), None),
                Some(header) => (format!("{header}\n{text}"), Some(message.origin.clone())),
            };
            content.push(UserContent::text(text));
            self.message_from(agent, conversation, Message::User { content }, origin);
        }
    }
}
