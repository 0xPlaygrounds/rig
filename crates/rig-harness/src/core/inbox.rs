//! Delivering messages to agents. [`Deliver`] is the one way a message
//! reaches an agent's conversation, whoever sends it: the user typing, an
//! agent reporting, a plugin. An idle agent starts a turn with it. A busy
//! one keeps it in its [`Inbox`]: a [`DeliveryMode::Steer`] message goes to
//! the model with the turn's next call, after the tool results it waits
//! for; a [`DeliveryMode::Queue`] one carries the turn on once it would
//! end. A turn that ends some other way (stopped, failed) hands what the
//! user typed and was not sent back to the views as [`Recalled`], so
//! nothing typed is lost or sent unasked, and puts what agents and plugins
//! sent in the conversation for the next turn.
//!
//! Each message carries its [`Origin`], kept with the conversation and its
//! log. Text that is not the user's own goes to the model as a user
//! message headed by a line naming where it came from.

use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use serde::{Deserialize, Serialize};

use super::agent::{
    ActiveTurn, Agent, AgentId, Connection, Conversation, Notice, TurnOf, TurnRequest,
};
use super::attach;
use super::journal::SessionLog;
use super::turn::CallModel;

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
    /// The request it answers or makes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestId>,
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
        }
    }

    /// Who sent it, for views: such as `agent 1a2b3c4d (request …)`.
    pub fn label(&self) -> String {
        let who = match &self.kind {
            OriginKind::User => "you".to_owned(),
            OriginKind::Agent => {
                format!("agent {}", self.from.as_ref().map_or("?", AgentId::short))
            }
            OriginKind::Plugin(name) => format!("plugin {name}"),
        };
        match &self.request {
            Some(request) => format!("{who} (request {})", request.0),
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

/// When a message to a busy agent goes to its model. An idle agent starts
/// a turn with it either way.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeliveryMode {
    /// With the running turn's next model call.
    #[default]
    Steer,
    /// Once the running turn would end, as its next step.
    Queue,
}

/// Put `text` in the agent's conversation, from `origin`, as `mode` says.
/// An image the user names as `@path` goes with the user's own text when
/// the model reads images. Empty text is ignored.
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
}

impl Deliver {
    /// The user's `text` for `agent`.
    pub fn user(agent: Entity, text: impl Into<String>, mode: DeliveryMode) -> Self {
        Self {
            entity: agent,
            text: text.into(),
            origin: Origin::user(),
            mode,
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
}

/// What was sent to the agent while it worked, not yet delivered.
#[derive(Component, Clone, Debug, Default)]
pub struct Inbox {
    /// Messages for the running turn, sent with its next model call.
    pub steering: Vec<Pending>,
    /// Messages for after the turn, sent one at a time when it would end.
    pub queued: VecDeque<Pending>,
}

impl Inbox {
    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.steering.is_empty() && self.queued.is_empty()
    }
}

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
    pub(crate) spec: Option<&'static ModelSpec>,
    /// The log every message goes through.
    pub(crate) log: &'a SessionLog,
}

/// Starts a turn of an idle agent with the message, or keeps it in a busy
/// one's inbox.
pub(crate) fn on_deliver(
    deliver: On<Deliver>,
    mut agents: Query<
        (
            &AgentId,
            &mut Inbox,
            &mut Conversation,
            Option<&Connection>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    log: Res<SessionLog>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = deliver.entity;
    let Ok((id, mut inbox, mut conversation, connection, busy)) = agents.get_mut(agent) else {
        return;
    };
    let text = deliver.text.trim();
    if text.is_empty() {
        return;
    }
    let pending = Pending {
        text: text.to_owned(),
        origin: deliver.origin.clone(),
    };
    if busy {
        match deliver.mode {
            DeliveryMode::Steer => inbox.steering.push(pending),
            DeliveryMode::Queue => inbox.queued.push_back(pending),
        }
        return;
    }
    let to = Delivery {
        agent,
        id,
        spec: connection.map(|connection| connection.spec),
        log: &log,
    };
    let mut request = None;
    // After a failure that kept the user's message, the new text joins it.
    commit(&to, pending, &mut conversation, &mut request, &mut notices);
    let turn = commands
        .spawn((Name::new("turn"), TurnOf(agent), TurnRequest(request)))
        .id();
    commands.trigger(CallModel { entity: turn });
}

/// Hands back what the user typed that a turn that just ended did not
/// send, and puts what agents and plugins sent in the conversation, where
/// the next turn reads it. A conversation left ending in the user's
/// message is logged as halted, so a restore does not answer it.
pub(crate) fn recall_on_turn_end(
    end: On<Remove<ActiveTurn>>,
    mut agents: Query<(&AgentId, &mut Inbox, &mut Conversation, Option<&Connection>)>,
    log: Res<SessionLog>,
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
        spec: connection.map(|connection| connection.spec),
        log: &log,
    };
    let mut typed = Vec::new();
    let waiting: Vec<Pending> = inbox
        .steering
        .drain(..)
        .chain(inbox.queued.drain(..))
        .collect();
    for pending in waiting {
        if pending.origin.kind == OriginKind::User {
            typed.push(pending.text);
        } else {
            commit(&to, pending, &mut conversation, &mut None, &mut notices);
        }
    }
    log.halt(id, &conversation);
    if !typed.is_empty() {
        recalled.write(Recalled {
            agent,
            text: typed.join("\n\n"),
        });
    }
}

/// Moves the steering messages into the conversation: into its last
/// message when that is the user's, such as the tool results the model
/// waits for, so user and model keep taking turns. Whether there were any.
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
    for pending in inbox.steering.drain(..) {
        commit(to, pending, conversation, request, notices);
    }
    true
}

/// Moves the oldest queued message into the conversation. Whether there
/// was one. `request` becomes its request, if it carries one.
pub(crate) fn deliver_queued(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    request: &mut Option<RequestId>,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let Some(pending) = inbox.queued.pop_front() else {
        return false;
    };
    commit(to, pending, conversation, request, notices);
    true
}

/// Commits `pending` as a user message with its origin: the user's own
/// text with the images it names, any other headed by its origin's line.
/// It goes into the last message when that is the user's. Its request, if
/// any, replaces `request`.
fn commit(
    to: &Delivery<'_>,
    pending: Pending,
    conversation: &mut Conversation,
    request: &mut Option<RequestId>,
    notices: &mut MessageWriter<Notice>,
) {
    let Pending { text, origin } = pending;
    if let Some(carried) = &origin.request {
        *request = Some(carried.clone());
    }
    let (message, origin) = match origin.header() {
        None => {
            let (message, notes) = attach::user_message(&text, to.spec);
            for note in notes {
                notices.write(Notice::info(to.agent, note));
            }
            (message, None)
        }
        Some(header) => (Message::user(format!("{header}\n{text}")), Some(origin)),
    };
    to.log.commit(to.id, conversation, message, origin);
}
