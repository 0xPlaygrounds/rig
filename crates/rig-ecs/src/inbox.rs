//! Delivering messages to agents. [`Deliver`] is the one way a message
//! reaches an agent's conversation, whoever sends it: the user typing, an
//! agent reporting, a plugin. An idle agent starts a turn with it, at the
//! end of the frame, so every message that reaches it in the same frame
//! goes to the model in that turn's first call. A busy one keeps it in its
//! [`Inbox`]: a [`DeliveryMode::Steer`] message goes to the model with the
//! turn's next call, after the tool results it waits for; the
//! [`DeliveryMode::Queue`] ones carry the turn on together, as one step,
//! once it would end. A [`DeliveryMode::Note`] needs no answer: it goes to
//! the model with whatever call comes next and never starts or carries on
//! a turn by itself. A turn that ends some other way (stopped, failed)
//! hands what the user typed and was not sent back to the views as
//! [`Recalled`], so nothing typed is lost or sent unasked, and puts what
//! agents and plugins sent in the conversation for the next turn.
//!
//! Each message carries its [`Origin`], kept with the conversation and its
//! log. Text that is not the user's own goes to the model as a user
//! message headed by a line naming where it came from. A message may carry
//! [`Attachment`]s its sender read, such as images, which go before its
//! text when the agent's model reads them.

use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

use super::agent::{
    ActiveTurn, Agent, AgentId, Connection, Conversation, Notice, TurnOf, TurnRequest,
};
use super::calls::Wake;
use super::journal::SessionLog;
use super::turn::{CallModel, Exiting};

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
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// Put `text` in the agent's conversation, from `origin`, as `mode` says,
/// after its `attachments`. Empty text is ignored.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Deliver {
    /// The agent.
    pub entity: Entity,
    /// The text.
    pub text: String,
    /// Where it came from.
    pub origin: Origin,
    /// When it goes to a busy agent's model.
    pub mode: DeliveryMode,
    /// What goes before the text.
    #[reflect(ignore)]
    pub attachments: Vec<Attachment>,
}

impl Deliver {
    /// The user's `text` for `agent`, with what it attaches.
    pub fn user(
        agent: Entity,
        text: impl Into<String>,
        mode: DeliveryMode,
        attachments: Vec<Attachment>,
    ) -> Self {
        Self {
            entity: agent,
            text: text.into(),
            origin: Origin::user(),
            mode,
            attachments,
        }
    }
}

/// A message waiting in an [`Inbox`].
#[derive(Clone, Debug)]
pub struct Pending {
    /// The text.
    pub text: String,
    /// Where it came from.
    pub origin: Origin,
    /// What goes before the text.
    pub attachments: Vec<Attachment>,
}

/// What was sent to the agent while it worked, not yet delivered.
#[derive(Component, Clone, Debug, Default)]
pub struct Inbox {
    /// Messages for the running turn, sent with its next model call.
    pub steering: Vec<Pending>,
    /// Messages for after the turn, sent together when it would end.
    pub queued: VecDeque<Pending>,
    /// Messages that need no answer, sent with the next model call that
    /// something else makes.
    pub notes: Vec<Pending>,
}

impl Inbox {
    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.steering.is_empty() && self.queued.is_empty() && self.notes.is_empty()
    }
}

/// On a turn that has not called its model yet: it does at the end of the
/// frame ([`start_turns`]), and what reaches its agent until then goes
/// with that first call.
#[derive(Component, Debug)]
pub(crate) struct Starting;

/// Messages the user typed that the agent's turn ended without sending,
/// joined in the order they were typed. A view puts them back in its input.
#[derive(Message, Clone, Debug)]
pub struct Recalled {
    /// The agent.
    pub agent: Entity,
    /// The messages.
    pub text: String,
}

/// The agent a message goes to, and what adding it needs.
pub(crate) struct Delivery<'a> {
    /// The agent.
    pub(crate) agent: Entity,
    /// Its id, which names its log.
    pub(crate) id: &'a AgentId,
    /// Its model, which decides whether images go with the message.
    pub(crate) spec: Option<&'a ModelSpec>,
    /// The log every message goes through.
    pub(crate) log: &'a SessionLog,
}

/// Starts a turn of an idle agent with the message, or keeps it in a busy
/// one's inbox. A note to an idle agent goes in its conversation, logged
/// as halted, for its next turn.
pub(crate) fn on_deliver(
    deliver: On<Deliver>,
    mut agents: Query<
        (
            &AgentId,
            &mut Inbox,
            &mut Conversation,
            Option<&Connection>,
            Option<&ActiveTurn>,
        ),
        With<Agent>,
    >,
    starting: Query<(), With<Starting>>,
    log: Res<SessionLog>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = deliver.entity;
    let Ok((id, mut inbox, mut conversation, connection, active)) = agents.get_mut(agent) else {
        return;
    };
    let text = deliver.text.trim();
    if text.is_empty() {
        return;
    }
    let pending = Pending {
        text: text.to_owned(),
        origin: deliver.origin.clone(),
        attachments: deliver.attachments.clone(),
    };
    if let Some(active) = active {
        // A turn that has not called its model yet takes every message
        // with its first call.
        if starting.contains(active.turn()) {
            inbox.steering.push(pending);
            return;
        }
        match deliver.mode {
            DeliveryMode::Steer => inbox.steering.push(pending),
            DeliveryMode::Queue => inbox.queued.push_back(pending),
            DeliveryMode::Note => inbox.notes.push(pending),
        }
        return;
    }
    let to = Delivery {
        agent,
        id,
        spec: connection.map(|connection| &*connection.spec),
        log: &log,
    };
    let mut request = None;
    // After a failure that kept the user's message, the new text joins it.
    commit(&to, pending, &mut conversation, &mut request, &mut notices);
    if deliver.mode == DeliveryMode::Note {
        log.halt(id, &conversation);
        return;
    }
    commands.spawn((
        Name::new("turn"),
        TurnOf(agent),
        TurnRequest(request),
        Starting,
    ));
    // The frame that starts it may have run its last systems already.
    wake.wake();
}

/// Calls the model for each turn [`Starting`] this frame, once every
/// message of the frame reached its agent.
pub(crate) fn start_turns(turns: Query<Entity, With<Starting>>, mut commands: Commands) {
    for turn in &turns {
        commands.entity(turn).remove::<Starting>();
        commands.trigger(CallModel { entity: turn });
    }
}

/// Hands back what the user typed that a turn that just ended did not
/// send, and puts what agents and plugins sent in the conversation, where
/// the next turn reads it. A conversation left ending in the user's
/// message is logged as halted, so a restore does not answer it, unless
/// the app is [`Exiting`].
pub(crate) fn recall_on_turn_end(
    end: On<Remove<ActiveTurn>>,
    mut agents: Query<(&AgentId, &mut Inbox, &mut Conversation, Option<&Connection>)>,
    log: Res<SessionLog>,
    exiting: Option<Res<Exiting>>,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = end.entity;
    let Ok((id, mut inbox, mut conversation, connection)) = agents.get_mut(agent) else {
        return;
    };
    let to = Delivery {
        agent,
        id,
        spec: connection.map(|connection| &*connection.spec),
        log: &log,
    };
    let mut typed = Vec::new();
    let inbox = &mut *inbox;
    let waiting: Vec<Pending> = inbox
        .notes
        .drain(..)
        .chain(inbox.steering.drain(..))
        .chain(inbox.queued.drain(..))
        .collect();
    for pending in waiting {
        if pending.origin.kind == OriginKind::User {
            typed.push(pending.text);
        } else {
            commit(&to, pending, &mut conversation, &mut None, &mut notices);
        }
    }
    if exiting.is_none() {
        log.halt(id, &conversation);
    }
    if !typed.is_empty() {
        recalled.write(Recalled {
            agent,
            text: typed.join("\n\n"),
        });
    }
}

/// Moves the notes into the conversation, for a model call about to be
/// made: into its last message when that is the user's.
pub(crate) fn deliver_notes(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) {
    for pending in inbox.notes.drain(..) {
        commit(to, pending, conversation, &mut None, notices);
    }
}

/// Moves the notes, then the steering messages into the conversation:
/// into its last message when that is the user's, such as the tool
/// results the model waits for, so user and model keep taking turns.
/// Whether there were steering messages; without them nothing moves.
/// `request` becomes the latest request they carry.
pub(crate) fn deliver_steering(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    request: &mut Option<RequestId>,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    if inbox.steering.is_empty() {
        return false;
    }
    deliver_notes(to, inbox, conversation, notices);
    for pending in inbox.steering.drain(..) {
        commit(to, pending, conversation, request, notices);
    }
    true
}

/// Moves the notes, then every queued message into the conversation, as
/// one step. Whether there were queued messages; without them nothing
/// moves. `request` becomes the latest request they carry.
pub(crate) fn deliver_queued(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    request: &mut Option<RequestId>,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    if inbox.queued.is_empty() {
        return false;
    }
    deliver_notes(to, inbox, conversation, notices);
    for pending in inbox.queued.drain(..) {
        commit(to, pending, conversation, request, notices);
    }
    true
}

/// Commits `pending` as a user message with its origin: its attachments
/// the model takes, then its text, headed by its origin's line when it is
/// not the user's own. It goes into the last message when that is the
/// user's. Its request, if any, replaces `request`.
fn commit(
    to: &Delivery<'_>,
    pending: Pending,
    conversation: &mut Conversation,
    request: &mut Option<RequestId>,
    notices: &mut MessageWriter<Notice>,
) {
    let Pending {
        text,
        origin,
        attachments,
    } = pending;
    if let Some(carried) = &origin.request {
        *request = Some(carried.clone());
    }
    let mut content = Vec::with_capacity(attachments.len() + 1);
    for Attachment {
        label,
        content: item,
    } in attachments
    {
        let refused = match (&item, to.spec) {
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
                notices.write(Notice::info(to.agent, note));
            }
            None => content.push(item),
        }
    }
    let (text, origin) = match origin.header() {
        None => (text, None),
        Some(header) => (format!("{header}\n{text}"), Some(origin)),
    };
    content.push(UserContent::text(text));
    to.log
        .commit(to.id, conversation, Message::User { content }, origin);
}
