//! Messages to agents. A message, whoever sends it (the user typing, an
//! agent reporting, a plugin), is an effect entity: an [`Incoming`]
//! [`ForAgent`] the agent, spawned by [`Deliver`] or by hand, which plugins
//! edit in Prepare and take over in Gate by answering it. Each carries its
//! [`Author`], kept with the conversation and its log; text that is not
//! the user's own goes to the model headed by a line naming its sender.
//! Each message is its own entry: nothing merges into the message before.
//!
//! At Start the kernel accepts what is left, in the order it was sent: an
//! idle agent takes a message into its conversation and starts a turn with
//! it, unless it is a [`DeliveryMode::Inform`]; a busy agent keeps it until
//! the turn's next call ([`DeliveryMode::Steer`]) or its end
//! ([`DeliveryMode::Queue`]). A user's `/` line ([`Incoming::command`]) is
//! text to the kernel: an app that has slash commands, such as
//! rig-harness, takes it over in Gate and runs it.
//! A message delivered into a turn gets [`DeliveredIn`] that turn. An
//! agent's inbox is the messages it accepted that are not answered yet.
//!
//! What a turn ends without sending is [`Unsent`]: a plugin keeps it or
//! takes it over, else the kernel hands what the user typed back to the
//! views as [`Recalled`] and keeps what agents and plugins sent for the
//! next turn. The inbox is saved with the conversation and restored after
//! a restart. The plugin guide (rig-harness's `PLUGINS.md`, "What the user
//! types") has the full rules.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::conversation::{Aside, AsideId, Author, AuthorKind, Entry, Halt};
use rig_core::effect::{EffectKind, HandlerKey};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};
use web_time::SystemTime;

use super::agent::{
    ActiveTurn, Agent, CallOf, Conversation, EffectParent, Notice, TurnOf, TurnOutcome,
};
use super::model::Connection;
#[cfg(doc)]
use super::stage::Stage;
use super::stage::{
    AgentWork, Decision, Effect, ForAgent, Holds, Outcome, ReadyToStart, Settled, Started,
    SupersededBy, moved,
};
use super::tasks::Wake;
use super::turn::{Exiting, TurnCall, spawn_turn_call, take_outcome};

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
    #[serde(rename = "note")]
    Inform,
}

/// Content read by a message's sender, such as an image file the user
/// named. An image goes only to a model that reads images; another model
/// gets the rest of the message, and the user is told.
#[derive(Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(opaque, Clone, Debug)]
pub struct Attachment {
    /// How the user knows it, such as its path.
    pub label: String,
    /// The content.
    pub content: UserContent,
}

/// One piece of a message ([`Incoming::parts`]).
#[derive(Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum MessagePart {
    /// Text.
    Text(String),
    /// Content the sender read, such as an image.
    Attachment(Attachment),
}

/// A message to the agent it is [`ForAgent`]: its `parts`, text and
/// attachments in the order its author put them, by `author`, that go to
/// the agent's model as `mode` says. An effect that moves through the
/// stages (see the [module](self)): edit it in Prepare, take it over in
/// Gate by deciding it ([`Decision`]) with an
/// [`Outcome`]; once accepted, it is [`Started`] until the kernel answers
/// it. A message with no text and no attachment is ignored.
///
/// ```
/// use rig_ecs::prelude::*;
/// use rig_ecs::prelude::message::{ImageMediaType, UserContent};
///
/// let mut message = Incoming::new("What is in this picture?", DeliveryMode::Steer);
/// let image = UserContent::image_base64("iVBORw0KGgo=", Some(ImageMediaType::PNG), None);
/// let picture = Attachment { label: "cat.png".into(), content: image };
/// message.parts.push(MessagePart::Attachment(picture)); // after the text
/// message.text_mut().push_str(" Be brief.");
/// assert_eq!(message.text(), "What is in this picture? Be brief.");
/// ```
///
/// Spawning `(Incoming, ForAgent(agent))`, with any components of the
/// sender's own, sends it: it is recorded under the effect the agent's
/// work is recorded under ([`EffectParent`]), unless it names a parent,
/// and the loop is woken. [`Deliver`] does the same for a sender that
/// needs nothing else. Its serde form is what its agent's conversation
/// saves of it while it is in the inbox.
#[derive(Component, Reflect, Clone, Debug, Serialize, Deserialize)]
#[component(on_add = sent)]
#[reflect(Component, Clone, Debug)]
#[require(Effect, Name = Name::new("message"))]
pub struct Incoming {
    /// What it says, in its author's order.
    pub parts: Vec<MessagePart>,
    /// Who wrote it.
    #[reflect(ignore)]
    pub author: Author,
    /// When it goes to a busy agent's model.
    pub mode: DeliveryMode,
    /// Whether the user's text goes to the model as it is even when it
    /// starts with `/`, such as a refused command sent again.
    pub literal: bool,
}

impl Incoming {
    /// `text`, as `mode` says, as the user's own text unless
    /// [`with_author`](Self::with_author) names who wrote it.
    pub fn new(text: impl Into<String>, mode: DeliveryMode) -> Self {
        Self {
            parts: vec![MessagePart::Text(text.into())],
            author: Author::user(),
            mode,
            literal: false,
        }
    }

    /// Its first text part, empty when it has none.
    pub fn text(&self) -> &str {
        texts(&self.parts).next().unwrap_or_default()
    }

    /// Its first text part, added after the other parts when it has none.
    pub fn text_mut(&mut self) -> &mut String {
        let first = self
            .parts
            .iter()
            .position(|part| matches!(part, MessagePart::Text(_)));
        let (at, part) = match first {
            Some(at) => (at, self.parts.remove(at)),
            None => (self.parts.len(), MessagePart::Text(String::new())),
        };
        match self.parts.insert_mut(at, part) {
            MessagePart::Text(text) => text,
            // Never: the part put back is a text.
            MessagePart::Attachment(attachment) => &mut attachment.label,
        }
    }

    /// Whether it has no text and no attachment, which the kernel ignores.
    fn is_empty(&self) -> bool {
        self.parts.iter().all(|part| match part {
            MessagePart::Text(text) => text.trim().is_empty(),
            MessagePart::Attachment(_) => false,
        })
    }

    /// This message, by `author`.
    #[must_use]
    pub fn with_author(self, author: Author) -> Self {
        Self { author, ..self }
    }

    /// This message sent [`literal`](Self::literal)ly.
    #[must_use]
    pub fn literal(self) -> Self {
        Self {
            literal: true,
            ..self
        }
    }

    /// The slash command line, without its `/`, that an app with slash
    /// commands runs instead of sending this message: the user's text that
    /// starts with `/`, unless it is [`literal`](Self::literal).
    pub fn command(&self) -> Option<&str> {
        let user = self.author.kind == AuthorKind::User && !self.literal;
        user.then(|| self.text().trim_start().strip_prefix('/'))
            .flatten()
    }
}

/// Send `message` to the agent `entity`: the kernel spawns it as an
/// [`Incoming`] effect [`ForAgent`] that agent, and the stages take it
/// from there (see the [module](self)). A plugin changes or takes over a
/// message through its `Incoming`, not this.
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
    /// [`with_author`](Self::with_author) names who wrote it.
    pub fn new(agent: Entity, text: impl Into<String>, mode: DeliveryMode) -> Self {
        Self {
            entity: agent,
            message: Incoming::new(text, mode),
        }
    }

    /// This message, by `author`.
    #[must_use]
    pub fn with_author(self, author: Author) -> Self {
        Self {
            message: self.message.with_author(author),
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

/// Text the user typed that was not sent or answered: the messages a failed
/// turn took out of the conversation, in order, then those the agent's
/// turn ended without sending, in the order they were sent, joined, or a
/// refused slash command, with `why`. A view puts it back in its input. A
/// failed turn's message comes back as the conversation holds it, after
/// Prepare, its text only; nothing else keeps it, so it does not outlive
/// a restart unless sent again.
#[derive(Message, Clone, Debug)]
pub struct Recalled {
    /// The agent.
    pub agent: Entity,
    /// The text.
    pub text: String,
    /// Why a command was refused.
    pub why: Option<String>,
}

/// On a message ([`Incoming`]) the kernel delivered into a turn: the turn whose
/// model call it went with, the turn it started or the running turn it steered
/// or carried on. Inserted when the message goes into the conversation, with
/// its answer. It is no hold: a turn's end does not wait for it. An
/// `Inform` message to an idle agent, and what a turn that ended some other way
/// puts in the conversation, go with no turn and have none.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct DeliveredIn(pub Entity);

/// On an agent whose failed turn took the user's unanswered messages out
/// of its conversation: their texts, in order, which go back to the views
/// before the messages the turn ended without sending ([`settle_unsent`]).
#[derive(Component, Debug)]
pub(crate) struct Unanswered(pub(crate) Vec<String>);

/// On a message its agent accepted and had not sent when the turn it
/// waited for ended, however it ended, or that a restart brought back to
/// an idle agent. In Gate a plugin keeps it in the inbox by removing this,
/// so it goes with the agent's next model call as its mode says, or takes
/// it over by deciding it. Otherwise, once every plugin had its say
/// ([`GateSystems::Fallback`](super::stage::GateSystems::Fallback)), the
/// kernel hands the user's back to the views as one [`Recalled`] and puts
/// the others in the conversation for the next turn. Not on exit: the
/// inbox stays saved for the restart.
#[derive(Component, Clone, Debug)]
#[component(on_insert = moved::<Unsent>)]
pub struct Unsent {
    /// How the turn ended; `None` after a restart.
    pub ended: Option<TurnOutcome>,
}

/// The kind of the aside that keeps a message in its agent's inbox, its
/// value the message, saved with the conversation until it is answered.
const INBOX_MARK: &str = "rig.inbox";

/// On a message in its agent's inbox: its `rig.inbox` aside.
#[derive(Component, Clone, Copy, Debug)]
pub(crate) struct InboxMark(AsideId);

/// Marks `message` kept in the inbox of the agent whose `conversation` it
/// is, after the last entry.
fn mark(conversation: &mut Conversation, message: &Incoming) -> InboxMark {
    let after = conversation.entries().next_back().map(|entry| entry.id);
    let value = serde_json::to_value(message).unwrap_or_default();
    let aside = Aside::new(INBOX_MARK, Author::rig(), value).after(after);
    InboxMark(conversation.add_aside(aside))
}

/// The messages kept in the inbox of `conversation`, with their marks.
pub(crate) fn kept(conversation: &Conversation) -> Vec<(InboxMark, Incoming)> {
    let marks = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == INBOX_MARK);
    let read = |aside: &Aside| Incoming::deserialize(&aside.value).ok();
    marks
        .filter_map(|(id, aside)| Some((InboxMark(id), read(aside)?)))
        .collect()
}

/// Takes its mark off a message in the inbox once it is answered.
pub(crate) fn unmark(
    answered: On<Insert<Outcome>>,
    marked: Query<(&InboxMark, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
) {
    if let Ok((&InboxMark(id), &ForAgent(agent))) = marked.get(answered.entity)
        && let Ok(mut conversation) = agents.get_mut(agent)
    {
        conversation.remove_aside(id);
    }
}

/// Spawns a [`Deliver`]'s message for its agent.
pub(crate) fn on_deliver(
    deliver: On<Deliver>,
    agents: Query<(), With<Agent>>,
    mut commands: Commands,
) {
    if agents.contains(deliver.entity) {
        commands.spawn((deliver.message.clone(), ForAgent(deliver.entity)));
    }
}

/// Puts a new message under the effect its agent's work is recorded
/// under, unless it names a parent, and wakes the loop: the frame that
/// sent it may have run its stages already.
fn sent(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    let parent = world
        .get::<ForAgent>(entity)
        .and_then(|agent| world.get::<EffectParent>(agent.0))
        .map(|parent| parent.0);
    if let Some(parent) = parent
        && let Some(mut effect) = world.get_mut::<Effect>(entity)
        && effect.parent.is_none()
    {
        effect.parent = Some(parent);
    }
    if let Some(wake) = world.get_resource::<Wake>() {
        wake.wake();
    }
}

/// Accepts each message about to start, in the order they were sent, unless a
/// Gate system failed this pass, in [`Stage::Start`]: gives an idle agent the
/// message at once and starts a turn with it, with every message accepted with
/// it, unless it is an `Inform` message, which is kept for the next turn; and
/// keeps it, [`Started`], for a busy agent.
pub(crate) fn accept(
    ready: Query<(Entity, &Effect, &Incoming, Option<&ForAgent>), ReadyToStart>,
    mut agents: Query<(&mut Conversation, Option<&Connection>, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut ready: Vec<_> = ready.iter().collect();
    ready.sort_unstable_by_key(|(_, effect, ..)| effect.id());
    let at = SystemTime::now();
    // The agents whose turn starts with this pass's messages, and the turn.
    let mut starting = HashMap::new();
    for (entity, effect, message, agent) in ready {
        let agent = agent.map(|agent| agent.0);
        let Some((agent, (mut conversation, connection, busy))) =
            agent.and_then(|agent| Some((agent, agents.get_mut(agent).ok()?)))
        else {
            let why = ErrorReport::new(ErrorKind::Internal, "a message needs an agent");
            commands.entity(entity).try_insert_if_new(Outcome(Err(why)));
            continue;
        };
        if message.is_empty() {
            commands
                .entity(entity)
                .try_insert_if_new(Outcome::custom("empty"));
            continue;
        }
        let joins = starting.get(&agent).copied();
        if busy && joins.is_none() {
            let mark = mark(&mut conversation, message);
            commands.entity(entity).try_insert((Started { at }, mark));
            continue;
        }
        let spec = connection.map(|connection| &*connection.spec);
        // An `Inform` message to an idle agent waits for the next turn.
        let turn = match joins {
            None if message.mode == DeliveryMode::Inform => None,
            None => Some(commands.spawn(TurnOf(agent)).id()),
            joins => joins,
        };
        commands.entity(entity).try_insert(Started { at });
        let conversation = &mut *conversation;
        let into = (turn, &mut commands);
        deliver(
            agent,
            spec,
            [(entity, message)],
            conversation,
            &mut notices,
            into,
        );
        match (turn, joins) {
            (None, _) => {
                conversation.halt(Halt::Kept);
            }
            (Some(turn), None) => {
                starting.insert(agent, turn);
                // The turn's first call is the message's work.
                commands.spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::with_parent_id(Some(effect.id())),
                ));
            }
            (Some(_), Some(_)) => {}
        }
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
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut carried = HashSet::new();
    for (message, &ForAgent(agent)) in &accepted {
        if message.mode == DeliveryMode::Inform || !carried.insert(agent) {
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
        let into = (Some(turn), &mut commands);
        deliver(agent, spec, next, conversation, &mut notices, into);
        take_outcome(&mut commands, turn, holds);
        spawn_turn_call(&mut commands, turn, agent);
    }
}

/// The messages agents accepted and were not given yet, as the kernel
/// reads them. A message decided is out the moment it is, though its
/// decision becomes its [`Outcome`] only at Start.
#[derive(SystemParam)]
pub(crate) struct Inboxes<'w, 's> {
    work: Query<'w, 's, &'static AgentWork>,
    accepted: Query<
        'w,
        's,
        (
            Entity,
            &'static Effect,
            &'static Incoming,
            &'static Decision,
        ),
        (With<Started>, Without<Outcome>, Without<Settled>),
    >,
    /// Not a message that settled, or that another superseded: its
    /// successor is on its way in its place.
    on_the_way: Query<
        'w,
        's,
        (&'static Incoming, &'static Decision),
        (
            Without<Started>,
            Without<Outcome>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
}

impl Inboxes<'_, '_> {
    /// What `agent` accepted and was not given yet, oldest first.
    pub(crate) fn of(&self, agent: Entity) -> Vec<(Entity, &Incoming)> {
        let work = self.work.get(agent).into_iter().flat_map(AgentWork::iter);
        let accepted = self.accepted.iter_many(work).flatten();
        let mut waiting: Vec<_> = accepted
            .filter(|(.., decision)| !decision.is_decided())
            .collect();
        waiting.sort_unstable_by_key(|(_, effect, ..)| effect.id());
        waiting
            .into_iter()
            .map(|(entity, _, message, _)| (entity, message))
            .collect()
    }

    /// Whether a Steer or Queue message to `agent` is on its way: spawned,
    /// not accepted, not decided.
    pub(crate) fn coming(&self, agent: Entity) -> bool {
        let work = self.work.get(agent).into_iter().flat_map(AgentWork::iter);
        self.on_the_way
            .iter_many(work)
            .flatten()
            .any(|(message, decision)| {
                message.mode != DeliveryMode::Inform && !decision.is_decided()
            })
    }
}

/// What of `waiting` carries a turn on that would end: the Inform messages
/// with the steering messages, else with everything queued, in the order they came;
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
            matches!(message.mode, DeliveryMode::Inform) || message.mode == carries
        })
        .collect();
    next.iter()
        .any(|(_, message)| message.mode == carries)
        .then_some(next)
}

/// Marks what the agent of a turn that just ended accepted and did not
/// send [`Unsent`], unless the app is [`Exiting`].
pub(crate) fn unsent_on_turn_end(
    end: On<Remove<TurnOf>>,
    turns: Query<(&TurnOf, Option<&TurnOutcome>)>,
    inboxes: Inboxes,
    exiting: Option<Res<Exiting>>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), ended)) = turns.get(end.entity) else {
        return;
    };
    if exiting.is_some() {
        return;
    }
    for (message, _) in inboxes.of(agent) {
        let ended = ended.cloned();
        commands.entity(message).try_insert(Unsent { ended });
    }
}

/// Settles each [`Unsent`] message nothing kept, holds or decided, in
/// [`GateSystems::Fallback`](super::stage::GateSystems::Fallback): an
/// idle agent hands the user's back to the views, joined in one
/// [`Recalled`] in the order they were sent after any [`Unanswered`] ones,
/// and puts the others in its conversation, left halted for the next turn;
/// a busy agent keeps them for its running turn.
pub(crate) fn settle_unsent(
    unsent: Query<
        (Entity, &Effect, &Incoming, &ForAgent, &Decision),
        (With<Unsent>, Without<Outcome>, Without<Holds>),
    >,
    mut agents: Query<(&mut Conversation, Option<&Connection>, Has<ActiveTurn>)>,
    unanswered: Query<(Entity, &Unanswered)>,
    mut commands: Commands,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let mut unsent: Vec<_> = unsent.iter().collect();
    unsent.sort_unstable_by_key(|(_, effect, ..)| effect.id());
    let mut typed: HashMap<Entity, Vec<&str>> = HashMap::new();
    for (agent, Unanswered(texts)) in &unanswered {
        commands.entity(agent).try_remove::<Unanswered>();
        let texts = texts.iter().map(|text| text.trim());
        typed.entry(agent).or_default().extend(texts);
    }
    for (entity, _, message, &ForAgent(agent), decision) in unsent {
        let Ok((mut conversation, connection, busy)) = agents.get_mut(agent) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        if busy {
            commands.entity(entity).try_remove::<Unsent>();
        } else if message.author == Author::user() {
            commands
                .entity(entity)
                .try_insert_if_new(Outcome::custom("recalled"));
            typed.entry(agent).or_default().push(message.text().trim());
        } else {
            let spec = connection.map(|connection| &*connection.spec);
            let conversation = &mut *conversation;
            let into = (None, &mut commands);
            deliver(
                agent,
                spec,
                [(entity, message)],
                conversation,
                &mut notices,
                into,
            );
            conversation.halt(Halt::Kept);
        }
    }
    for (agent, typed) in typed {
        recalled.write(Recalled {
            agent,
            text: typed.join("\n\n"),
            why: None,
        });
    }
}

/// The payload of a message's effect record (custom kind `message`): its
/// joined text, author and mode, without its attachments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageRecord {
    /// The text.
    pub text: String,
    /// Who wrote it.
    pub author: Author,
    /// When it went to a busy agent's model.
    pub mode: DeliveryMode,
}

impl MessageRecord {
    /// The kind of a message's record.
    pub const KIND: &str = "message";

    /// The message a record of `kind` is about, if it is a message's.
    pub fn read(kind: &EffectKind) -> Option<Self> {
        match kind {
            EffectKind::Custom { kind, payload } if &**kind == Self::KIND => {
                serde_json::from_value(payload.clone()).ok()
            }
            _ => None,
        }
    }
}

/// The record's handler key and kind of a message.
pub(crate) fn message_kind(message: &Incoming) -> (HandlerKey, EffectKind) {
    let record = MessageRecord {
        text: texts(&message.parts).collect::<Vec<_>>().join("\n"),
        author: message.author.clone(),
        mode: message.mode,
    };
    let kind = EffectKind::Custom {
        kind: Arc::from(MessageRecord::KIND),
        payload: serde_json::to_value(record).unwrap_or_default(),
    };
    (HandlerKey::new(MessageRecord::KIND), kind)
}

/// Puts each of `messages`, accepted by `agent`, in its `conversation`
/// as [`pending`] does, and answers it as delivered, [`DeliveredIn`] the
/// turn `into` names, if any.
pub(crate) fn deliver<'a>(
    agent: Entity,
    spec: Option<&ModelSpec>,
    messages: impl IntoIterator<Item = (Entity, &'a Incoming)>,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
    (into, commands): (Option<Entity>, &mut Commands),
) {
    for (entity, message) in messages {
        push_messages(agent, spec, [message], conversation, notices);
        let mut delivered = commands.entity(entity);
        delivered.try_insert_if_new(Outcome::custom("delivered"));
        if let Some(turn) = into {
            delivered.try_insert(DeliveredIn(turn));
        }
    }
}

/// The text parts of `parts`, in order.
fn texts(parts: &[MessagePart]) -> impl Iterator<Item = &str> {
    parts.iter().filter_map(|part| match part {
        MessagePart::Text(text) => Some(text.as_str()),
        MessagePart::Attachment(_) => None,
    })
}

/// Pushes each of `messages` to `conversation`, that of `agent`, whose
/// model is `spec`, as a user message of its own by its author: its parts
/// in order, each attachment the model takes and each text trimmed, the
/// first text headed by its author's line when it is not the user's own.
/// The author is kept, and logged, unless it is the plain user, so a
/// user's message with a title keeps it.
pub(crate) fn push_messages<'a>(
    agent: Entity,
    spec: Option<&ModelSpec>,
    messages: impl IntoIterator<Item = &'a Incoming>,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) {
    for message in messages {
        let mut header = message.author.header();
        let mut content = Vec::with_capacity(message.parts.len());
        let mut refused = Vec::new();
        for part in &message.parts {
            let attachment = match part {
                MessagePart::Text(text) => {
                    let text = text.trim();
                    let text = match header.take() {
                        Some(header) if text.is_empty() => header,
                        Some(header) => format!("{header}\n{text}"),
                        None => text.to_owned(),
                    };
                    if !text.is_empty() {
                        content.push(UserContent::text(text));
                    }
                    continue;
                }
                MessagePart::Attachment(attachment) => attachment,
            };
            let Attachment {
                label,
                content: item,
            } = attachment;
            let why = match (item, spec) {
                (UserContent::Image(_), Some(spec)) if !spec.input.image => Some(format!(
                    "{} does not read images, so {label} is sent as its name only.",
                    spec.display_name
                )),
                (UserContent::Image(_), None) => Some(format!(
                    "No model is connected, so {label} is sent as its name only."
                )),
                _ => None,
            };
            match why {
                Some(note) => {
                    notices.write(Notice::info(agent, note));
                    refused.push(label.as_str());
                }
                None => content.push(item.clone()),
            }
        }
        // A message of attachments alone keeps its header first, and the
        // names of those the model cannot read when nothing else is left.
        if content.is_empty() {
            content.extend(refused.into_iter().map(UserContent::text));
        }
        if let Some(header) = header {
            content.insert(0, UserContent::text(header));
        }
        let entry = Entry::new(Message::User { content });
        let entry = match message.author == Author::user() {
            true => entry,
            false => entry.by(message.author.clone()),
        };
        conversation.push(entry);
    }
}

#[cfg(test)]
mod tests;
