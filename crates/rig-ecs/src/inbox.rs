//! Messages to agents. A message, whoever sends it (the user typing, an
//! agent reporting, a plugin), is an effect entity: an [`Incoming`]
//! [`ForAgent`] the agent, spawned by [`Deliver`] or by hand, which plugins
//! edit in Prepare and take over in Gate by answering it. Each carries its
//! [`Origin`], kept with the conversation and its log; text that is not
//! the user's own goes to the model headed by a line naming its sender.
//!
//! At Start the kernel accepts what is left, in the order it was sent: an
//! idle agent takes a message into its conversation and starts a turn with
//! it, unless it is a [`DeliveryMode::Note`]; a busy agent keeps it until
//! the turn's next call ([`DeliveryMode::Steer`]) or its end
//! ([`DeliveryMode::Queue`]). A user's `/` line ([`Incoming::command`]) is
//! text to the kernel: an app that has slash commands, such as
//! rig-harness, takes it over in Gate and runs it.
//! A message delivered into a turn gets [`DeliveredIn`] that turn. An
//! agent's inbox is the messages it accepted that are not answered yet.
//!
//! A turn that ends some other way hands what the user typed and was not
//! sent back to the views as [`Recalled`], and keeps what agents and
//! plugins sent for the next turn. The plugin guide (rig-harness's
//! `PLUGINS.md`, "What the user types") has the full rules.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_reflect::prelude::*;
use rig_cassette::journal::{Halt, Origin, OriginKind};
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::effect::{EffectKind, HandlerKey};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};
use web_time::SystemTime;

use super::agent::{
    ActiveTurn, Agent, CallOf, Conversation, EffectParent, Notice, TurnOf, TurnOutcome,
};
use super::calls::Wake;
use super::journal::Commit;
use super::model::Connection;
use super::stage::{
    AgentWork, Effect, ForAgent, Holds, Outcome, Settled, Stage, Started, Undecided, failed_in,
};
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
/// in Prepare, take it over in Gate by deciding it
/// ([`Decision`](super::stage::Decision)) with an [`Outcome`];
/// once accepted, it is [`Started`] until the kernel answers it. Empty
/// text is ignored.
///
/// Spawning `(Incoming, ForAgent(agent))`, with any components of the
/// sender's own, sends it: it is recorded under the effect the agent's
/// work is recorded under ([`EffectParent`]), unless it names a parent,
/// and the loop is woken. [`Deliver`] does the same for a sender that
/// needs nothing else.
#[derive(Component, Reflect, Clone, Debug)]
#[component(on_add = sent)]
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

    /// The slash command line, without its `/`, that an app with slash
    /// commands runs instead of sending this message: the user's text that
    /// starts with `/`, unless it is [`literal`](Self::literal).
    pub fn command(&self) -> Option<&str> {
        let user = self.origin.kind == OriginKind::User && !self.literal;
        user.then(|| self.text.trim_start().strip_prefix('/'))
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

/// On a message ([`Incoming`]) the kernel delivered into a turn: the turn
/// whose model call it went with, the turn it started or the running turn
/// it steered or carried on. Inserted when the message goes into the
/// conversation, with its answer; read-only, so a turn's end does not wait
/// for its messages. A note to an idle agent, and what a turn that ended
/// some other way puts in the conversation, go with no turn and have none.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct DeliveredIn(pub Entity);

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

/// How the kernel answered a message it took: `delivered`, `recalled` or
/// `empty`.
fn handled(how: &'static str) -> Outcome {
    Outcome::custom(how)
}

/// Accepts each message about to start, in the order they were sent,
/// unless a Gate system failed this pass, in [`Stage::Start`]: gives an
/// idle agent the message at once and starts a turn with it, with every message accepted with it, unless it
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
        if message.text.trim().is_empty() {
            commands.entity(entity).try_insert_if_new(handled("empty"));
            continue;
        }
        let joins = starting.get(&agent).copied();
        if busy && joins.is_none() {
            commands.entity(entity).try_insert(Started { at });
            continue;
        }
        let spec = connection.map(|connection| &*connection.spec);
        // A note to an idle agent waits for the next turn.
        let turn = match joins {
            None if message.mode == DeliveryMode::Note => None,
            None => Some(commands.spawn(TurnOf(agent)).id()),
            joins => joins,
        };
        commands.entity(entity).try_insert(Started { at });
        // After a failure that kept the user's message, the new text joins it.
        let conversation = &mut *conversation;
        let into = (turn, &mut commands);
        commit.deliver(
            agent,
            spec,
            [(entity, message)],
            conversation,
            &mut notices,
            into,
        );
        match (turn, joins) {
            (None, _) => commit.halt(agent, conversation, Halt::Kept),
            (Some(turn), None) => {
                starting.insert(agent, turn);
                // The turn's first call is the message's work.
                commands.spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::child_of(Some(effect.id())),
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
        let into = (Some(turn), &mut commands);
        commit.deliver(agent, spec, next, conversation, &mut notices, into);
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
    let into = (None, &mut commands);
    commit.deliver(agent, spec, sent, conversation, &mut notices, into);
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

/// What a message's record says of it, its kind's payload: its text,
/// origin and mode, without its attachments. The record's kind is the
/// custom kind `message`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageRecord {
    /// The text.
    pub text: String,
    /// Where it came from.
    pub origin: Origin,
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
        text: message.text.clone(),
        origin: message.origin.clone(),
        mode: message.mode,
    };
    let kind = EffectKind::Custom {
        kind: Arc::from(MessageRecord::KIND),
        payload: serde_json::to_value(record).unwrap_or_default(),
    };
    (HandlerKey::new(MessageRecord::KIND), kind)
}

impl Commit<'_, '_> {
    /// Commits each of `messages`, accepted by `agent`, to its
    /// `conversation` as [`pending`](Self::pending) does, and answers it
    /// as delivered, [`DeliveredIn`] the turn `into` names, if any.
    pub(crate) fn deliver<'a>(
        &mut self,
        agent: Entity,
        spec: Option<&ModelSpec>,
        messages: impl IntoIterator<Item = (Entity, &'a Incoming)>,
        conversation: &mut Conversation,
        notices: &mut MessageWriter<Notice>,
        (into, commands): (Option<Entity>, &mut Commands),
    ) {
        for (entity, message) in messages {
            self.pending(agent, spec, [message], conversation, notices);
            let mut delivered = commands.entity(entity);
            delivered.try_insert_if_new(handled("delivered"));
            if let Some(turn) = into {
                delivered.try_insert(DeliveredIn(turn));
            }
        }
    }

    /// Commits each of `messages` to `conversation`, that of `agent`, whose
    /// model is `spec`, as a user message with its origin: its attachments
    /// the model takes, then its text, trimmed, headed by its origin's line
    /// when it is not the user's own. The origin is kept, and logged, unless
    /// it is the plain user's, so a user's message with a title keeps it.
    /// Each goes into the last message when that is the user's, such as
    /// the tool results the model waits for, so user and model keep taking
    /// turns.
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
            let text = match message.origin.header() {
                None => text.to_owned(),
                Some(header) => format!("{header}\n{text}"),
            };
            let origin = (message.origin != Origin::user()).then(|| message.origin.clone());
            content.push(UserContent::text(text));
            self.message_from(agent, conversation, Message::User { content }, origin);
        }
    }
}
