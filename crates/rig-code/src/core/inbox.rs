//! Typing while the agent works. A message sent to a busy agent waits in
//! its [`Inbox`]: one sent with [`Submit`] steers the
//! running turn and goes to the model with its next call, after the tool
//! results it waits for; one sent with [`FollowUp`] goes once the turn
//! would end, as the next step of the same turn. A turn that ends some
//! other way (stopped, failed) hands what was not delivered back to the
//! views as [`Recalled`], so nothing typed is lost or sent unasked.

use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;

use super::agent::{ActiveTurn, Agent, Notice, Submit};
use super::attach;

/// What was typed to the agent while it worked, not yet sent.
#[derive(Component, Clone, Debug, Default)]
pub struct Inbox {
    /// Messages for the running turn, sent with its next model call.
    pub steering: Vec<String>,
    /// Messages for after the turn, sent one at a time when it would end.
    pub follow_ups: VecDeque<String>,
}

impl Inbox {
    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.steering.is_empty() && self.follow_ups.is_empty()
    }

    /// Takes every waiting message, steering first, joined by blank lines.
    pub fn take_all(&mut self) -> String {
        let all: Vec<String> = self
            .steering
            .drain(..)
            .chain(self.follow_ups.drain(..))
            .collect();
        all.join("\n\n")
    }
}

/// Send `text` to the agent once its running turn would end, or now when
/// it is idle. A slash command runs now either way.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct FollowUp {
    /// The agent.
    pub entity: Entity,
    /// The text typed.
    pub text: String,
}

/// Messages typed to the agent that its turn ended without sending, joined
/// in the order they were typed. A view puts them back in its input.
#[derive(Message, Clone, Debug)]
pub struct Recalled {
    /// The agent.
    pub agent: Entity,
    /// The messages.
    pub text: String,
}

/// Queues a follow-up for a busy agent, or submits it to an idle one.
pub(crate) fn on_follow_up(
    follow_up: On<FollowUp>,
    mut agents: Query<(&mut Inbox, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
) {
    let agent = follow_up.entity;
    let text = follow_up.text.trim();
    let Ok((mut inbox, busy)) = agents.get_mut(agent) else {
        return;
    };
    if text.is_empty() {
        return;
    }
    if busy && !text.starts_with('/') {
        inbox.follow_ups.push_back(text.to_owned());
    } else {
        commands.trigger(Submit {
            entity: agent,
            text: text.to_owned(),
        });
    }
}

/// Hands back what a turn that just ended did not send.
pub(crate) fn recall_on_turn_end(
    end: On<Remove<ActiveTurn>>,
    mut agents: Query<&mut Inbox>,
    mut recalled: MessageWriter<Recalled>,
) {
    let agent = end.entity;
    if let Ok(mut inbox) = agents.get_mut(agent)
        && !inbox.is_empty()
    {
        recalled.write(Recalled {
            agent,
            text: inbox.take_all(),
        });
    }
}

/// Moves the steering messages into the conversation, with their images:
/// into its last message when that is the user's, such as the tool results
/// the model waits for, so user and model keep taking turns. Whether there
/// were any.
pub(crate) fn deliver_steering(
    agent: Entity,
    inbox: &mut Inbox,
    conversation: &mut Vec<Message>,
    spec: Option<&ModelSpec>,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    if inbox.steering.is_empty() {
        return false;
    }
    for text in inbox.steering.drain(..) {
        append_user(agent, &text, conversation, spec, notices);
    }
    true
}

/// Moves the oldest follow-up into the conversation. Whether there was one.
pub(crate) fn deliver_follow_up(
    agent: Entity,
    inbox: &mut Inbox,
    conversation: &mut Vec<Message>,
    spec: Option<&ModelSpec>,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let Some(text) = inbox.follow_ups.pop_front() else {
        return false;
    };
    append_user(agent, &text, conversation, spec, notices);
    true
}

/// Appends `text` as the user's, into the last message when it is the
/// user's.
fn append_user(
    agent: Entity,
    text: &str,
    conversation: &mut Vec<Message>,
    spec: Option<&ModelSpec>,
    notices: &mut MessageWriter<Notice>,
) {
    let (message, notes) = attach::user_message(text, spec);
    for note in notes {
        notices.write(Notice::info(agent, note));
    }
    let Message::User { content: added } = message else {
        return;
    };
    match conversation.last_mut() {
        Some(Message::User { content }) => content.extend(added),
        _ => conversation.push(Message::User { content: added }),
    }
}
